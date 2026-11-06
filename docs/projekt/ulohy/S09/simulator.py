"""Simulátor akvizice turbo spin echo (TSE) v MRI (simulační úloha S09).

Rozhodnutí je POŘADÍ SBĚRU ŘÁDKŮ k-prostoru: permutace `poradi` čísel
0..N-1 (N = 64 řádků). `poradi[t]` = který řádek se nasbírá v t-tém slotu.
Řádek r odpovídá prostorové frekvenci ky = r - N/2 (řádek 32 je střed).

Časové sloty
    Sběr má S = N / ETL buzení, v každém buzení vlak ETL ech.
    Slot t patří buzení  s = t // ETL  a echu  e = t % ETL,
    čas echa TE_e = (e + 1) * ESP  (ms od buzení).

k-prostor a útlum T2
    Fantom (řez hlavou 64x64 px) se skládá z tkání j s hustotou protonů
    rho_j(x, y) a časem T2_j. Řádek ky nasbíraný v čase TE nese signál
        K(ky, kx) = sum_j exp(-TE / T2_j) * F_j(ky, kx),
    F_j = 2D Fourierova transformace mapy rho_j.
    Obraz = |inverzní 2D FFT(K)| (magnituda, tu vidí radiolog).

Cíl (ideál)
    Obraz, jako by se všechny řádky nasbíraly v čase TE_ef:
        T = |IFFT( sum_j exp(-TE_ef / T2_j) F_j )|.
    Chyba = 100 * ||obraz - T|| / ||T||  (relativní chyba v %, Frobeniova norma).

Pohyb pacienta (n_scen > 0)
    Scénář (b, dy, dx): pacient se jednou pohne po buzení b (b rovnoměrně
    z 1..S-1) o celé pixely, |dy|, |dx| <= max_posun, (dy, dx) != (0, 0).
    Řádky z buzení s >= b dostanou fázový posun exp(-2 pi i (ky dy + kx dx) / N).
    Chyba scénáře se počítá vůči té ze dvou poloh ideálu (před / po pohybu),
    ke které má obraz blíž (radiologovi vadí artefakty, ne poloha hlavy).
    Vrací se průměr přes scénáře 0..n_scen-1 daného `seed`. Stejný seed =
    stejné scénáře pro každé pořadí; scénáře 0..k-1 jsou stejné pro každé
    n_scen >= k.

Použití
    from simulator import TSE
    sim = TSE(instance=0)
    sim.chyba(poradi)                      # bez pohybu (deterministická)
    sim.chyba(poradi, n_scen=20, seed=3)   # průměr přes 20 scénářů pohybu
    sim.chyba(poradi, n_scen=20, seed=3, po_scenarich=True)  # pole chyb
    sim.scenare(20, seed=3)                # seznam scénářů (b, dy, dx)
    sim.obraz(poradi, scenar=(b, dy, dx))  # obraz k vykreslení
    sim.ideal()                            # ideální obraz T
    sim.akvizic   # počítadlo akvizic (rozpočet): bez pohybu 1, s pohybem n_scen
Parametry instance: sim.N, sim.ETL, sim.S, sim.ESP, sim.TE, sim.TE_ef,
sim.tkane (seznam slovníků nazev, T2, rho), sim.max_posun.
"""
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

TKANE = [  # nazev, T2 [ms], hustota protonu (relativni)
    ("kuze a tuk", 70.0, 0.9),
    ("kost", 40.0, 0.1),
    ("seda hmota", 100.0, 0.8),
    ("bila hmota", 75.0, 0.65),
    ("likvor", 1000.0, 1.0),
    ("leze", 160.0, 0.85),
]


def _elipsa(Y, X, cy, cx, ay, ax, uhel=0.0):
    c, s = np.cos(uhel), np.sin(uhel)
    y, x = Y - cy, X - cx
    u, v = c * x + s * y, -s * x + c * y
    return (u / ax) ** 2 + (v / ay) ** 2 <= 1.0


def fantom(N, rng):
    """Mapy tkání (J, N, N) s částečným objemem (4x nadvzorkování)."""
    m = 4
    g = (np.arange(N * m) + 0.5) / m - N / 2
    Y, X = np.meshgrid(g, g, indexing="ij")
    j = lambda: rng.uniform(0.95, 1.05)
    a = 0.42 * N
    hlava = _elipsa(Y, X, 0, 0, a * j(), 0.80 * a * j())
    lebka = _elipsa(Y, X, 0, 0, 0.92 * a, 0.92 * 0.80 * a)
    mozek = _elipsa(Y, X, 0, 0, 0.84 * a, 0.84 * 0.80 * a)
    bila = _elipsa(Y, X, 0.02 * a, 0, 0.66 * a * j(), 0.62 * a * j())
    kom = np.zeros_like(hlava)
    for strana in (-1, 1):                                    # postranní komory
        kom |= _elipsa(Y, X, -0.08 * a, strana * 0.16 * a * j(), 0.30 * a * j(),
                       0.07 * a * j(), strana * 0.35)
    kom |= _elipsa(Y, X, 0.28 * a, 0, 0.06 * a, 0.03 * a)    # třetí komora
    r, fi = rng.uniform(0.25, 0.5) * a, rng.uniform(0, 2 * np.pi)
    leze = _elipsa(Y, X, r * np.sin(fi), r * np.cos(fi), 0.11 * a * j(), 0.15 * a * j(),
                   rng.uniform(0, np.pi))
    masky = [hlava & ~lebka, lebka & ~mozek, mozek & ~bila & ~kom & ~leze,
             bila & ~kom & ~leze, kom, leze & ~kom]
    return np.array([mk.reshape(N, m, N, m).mean(axis=(1, 3)) for mk in masky])


class TSE:
    def __init__(self, instance=0, N=64, ETL=16, ESP=10.0, TE_ef=None, max_posun=3):
        rng = np.random.default_rng(1000 + instance)
        self.instance, self.N, self.ETL, self.ESP = instance, N, ETL, float(ESP)
        self.S = N // ETL
        self.max_posun = max_posun
        self.TE_ef = float(TE_ef if TE_ef is not None else rng.choice([70.0, 80.0, 90.0]))
        self._T2 = np.array([t[1] for t in TKANE]) * rng.uniform(0.9, 1.1, len(TKANE))
        rho = np.array([t[2] for t in TKANE])
        self.tkane = [dict(nazev=t[0], T2=float(a), rho=float(b)) for t, a, b in zip(TKANE, self._T2, rho)]
        self.mapy = fantom(N, rng) * rho[:, None, None]           # (J, N, N)
        self._F = fftshift(fft2(self.mapy), axes=(1, 2))         # řádek N/2 = ky 0
        self.TE = (np.arange(ETL) + 1) * self.ESP                 # časy ech [ms]
        self._T = self._magnituda(np.exp(-self.TE_ef / self._T2) @ self._F.reshape(len(TKANE), -1))[0]
        self._Tn = np.linalg.norm(self._T)
        self.akvizic = 0
        self._pamet = {}

    # --- pomocné ---
    def _magnituda(self, K):
        return np.abs(ifft2(ifftshift(K.reshape(-1, self.N, self.N), axes=(-2, -1))))

    def _kontrola(self, poradi):
        p = np.asarray(poradi)
        if p.shape != (self.N,) or not np.array_equal(np.sort(p), np.arange(self.N)):
            raise ValueError("poradi musí být permutace čísel 0..N-1")
        return p

    def _kprostor(self, p):
        """k-prostor (N, N) a buzení každého řádku pro pořadí p."""
        t = np.empty(self.N, int)
        t[p] = np.arange(self.N)                                  # slot každého řádku
        A = np.exp(-self.TE[t % self.ETL][None, :] / self._T2[:, None])   # (J, N)
        return np.einsum("jy,jyx->yx", A, self._F), t // self.ETL

    def _s_pohybem(self, K, buz, scen):
        """Magnitudy obrazů pro scénáře. Fázový posun řádků v k-prostoru =
        cyklický posun jejich příspěvku k obrazu (věta o posunu), takže stačí
        jedna inverzní FFT na buzení: obraz = |I_pred + posun(I_po, dy, dx)|."""
        maska = buz[None, :, None] == np.arange(self.S)[:, None, None]
        kum = np.cumsum(ifft2(ifftshift(K[None] * maska, axes=(-2, -1))), axis=0)
        return np.array([np.abs(kum[b - 1] + np.roll(kum[-1] - kum[b - 1], (dy, dx), axis=(0, 1)))
                         for b, dy, dx in scen])

    # --- veřejné rozhraní ---
    def scenare(self, n_scen, seed=0):
        """Seznam scénářů (b, dy, dx); scénář i závisí jen na (instance, seed, i)."""
        if ("sc", seed, n_scen) not in self._pamet:
            out = []
            for i in range(n_scen):
                r = np.random.default_rng([self.instance, seed, i])
                b = int(r.integers(1, self.S))
                dy = dx = 0
                while (dy, dx) == (0, 0):
                    dy, dx = (int(v) for v in r.integers(-self.max_posun, self.max_posun + 1, 2))
                out.append((b, dy, dx))
            T = np.array([np.roll(self._T, (dy, dx), axis=(0, 1)) for _, dy, dx in out])
            self._pamet = {("sc", seed, n_scen): (out, T)}          # pamatuje si jen poslední sadu
        return self._pamet[("sc", seed, n_scen)][0]

    def chyba(self, poradi, n_scen=0, seed=0, po_scenarich=False):
        p = self._kontrola(poradi)
        self.akvizic += max(1, n_scen)
        K, buz = self._kprostor(p)
        if n_scen == 0:
            return 100 * np.linalg.norm(self._magnituda(K)[0] - self._T) / self._Tn
        scen = self.scenare(n_scen, seed)
        Tpo = self._pamet[("sc", seed, n_scen)][1]
        M = self._s_pohybem(K, buz, scen)
        e1 = np.sqrt(((M - self._T) ** 2).sum(axis=(1, 2)))
        e2 = np.sqrt(((M - Tpo) ** 2).sum(axis=(1, 2)))
        e = 100 * np.minimum(e1, e2) / self._Tn
        return e if po_scenarich else float(e.mean())

    def obraz(self, poradi, scenar=None):
        K, buz = self._kprostor(self._kontrola(poradi))
        if scenar is None:
            return self._magnituda(K)[0]
        return self._s_pohybem(K, buz, [tuple(scenar)])[0]

    def ideal(self):
        return self._T.copy()


if __name__ == "__main__":
    import time
    sim = TSE(0)
    p = np.arange(sim.N)
    print("TE_ef", sim.TE_ef, "sekvencni poradi bez pohybu", round(sim.chyba(p), 2))
    t = time.perf_counter()
    for _ in range(50):
        sim.chyba(p, n_scen=20)
    print("s pohybem", round(sim.chyba(p, n_scen=20), 2),
          f"{(time.perf_counter() - t) / 50 * 1000:.2f} ms / vyhodnocení (20 scénářů)")
