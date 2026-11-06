"""Dekodér rozložení dílů v pásu perikardu (simulační úloha S12).

Rozhodnutí je POŘADÍ dílů (permutace indexů 0..n-1, index = pořadí v seznamu
`dily` v souboru instance) a ROTACE každého dílu ve stupních (násobek 15°,
0 <= rotace < 360; základní úloha používá jen 0, 90, 180, 270).

Pás perikardu
    Pás má šířku `sirka` (osa y, 0..sirka mm) a roli délky `delka_max`
    (osa x). Pracuje se na rastru 1 mm: buňka je obsazená, pokud do ní
    díl zasahuje aspoň trochu (test na podmřížce 0,25 mm a vrcholy). V pásu jsou
    vady (kruhy `vady` = [x, y, r]), na které se nesmí nic položit.

Dekodér (pravidlo „nejvíc vlevo, pak nejníž“)
    Díly se pokládají v daném pořadí. Každý díl s danou rotací se položí na
    rastrovou pozici s nejmenším x (a při shodě nejmenším y), kde se nepřekrývá
    s okrajem pásu, s vadami ani s již položenými díly; od dílů a vad musí
    dodržet mezeru `mezera` mm (položené díly a vady se o ni rastrově
    zvětší). Každá permutace s rotacemi je tedy přípustná, pokud se díl vejde
    do role.

    vlakna=True: cípy (díly s "vlakna": true) se navíc smějí položit jen tam,
    kde se osa cípu (po rotaci) odchyluje od směru vláken v těžišti cípu
    nejvýš o `tolerance_vlaken` stupňů. Takto pracuje řezací software, když
    obsluha zapne kontrolu vláken. S vlakna=False se vlákna ignorují
    a výsledek jen hlásí, o kolik se který cíp odchyluje.

Směr vláken
    theta(x, y) = theta0 + sum_k A_k sin(2 pi (x / lx_k + y / ly_k) + phi_k)
    [stupně, směr měřený od osy x, je to směr bez orientace: theta a theta+180
    jsou totéž]. Osa cípu s rotací r má směr r (mod 180).

Cíl
    delka = pravý okraj nejvzdálenějšího obsazeného rastrového políčka [mm]
    (celé milimetry, takže funkce má plošiny). Menší je lepší.

Použití
    from simulator import Pas
    pas = Pas()                               # hlavní instance (data/hlavni.json vedle simulátoru)
    pas = Pas("dalsi/s100.json")              # jiná instance: cesta relativně k data/ (nebo plná)
    r = pas.rozloz(poradi, rotace)            # rotace: pole délky n (index = díl)
    r["delka"], r["rozlozeni"], r["odchylka_vlaken"]
    pas.polygon(i, rotace_i, x, y)            # skutečný tvar položeného dílu
    pas.theta(x, y)                           # směr vláken [stupně]
    pas.vyhodnoceni                           # počet volání rozloz (rozpočet)

`r["rozlozeni"]` je seznam (díl, rotace, x, y) v pořadí pokládání; (x, y) je
posun dílu: skutečný tvar = polygon dílu otočený o rotaci kolem počátku,
posunutý tak, aby levý dolní roh jeho obálky byl v (x, y).
"""
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

PODMRIZKA = 4      # 4 x 4 testovací body na 1 mm buňku
DATA = Path(__file__).resolve().parent / "data"
OKNO = 1.8         # rychlé hledání v okně ~1,8 x (plocha dílů / šířka) + 60 mm, dál pomalejší cesta


def _uvnitr(body, poly):
    """Test bod v polygonu (paprsková metoda), body (m, 2), poly (k, 2)."""
    x, y = body[:, 0:1], body[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    krizi = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1)
    return krizi.sum(axis=1) % 2 == 1


def _plocha_teziste(p):
    k = p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
    a = k.sum() / 2
    return abs(a), ((p[:, 0] + np.roll(p[:, 0], -1)) @ k / (6 * a),
                    (p[:, 1] + np.roll(p[:, 1], -1)) @ k / (6 * a))


class Pas:
    def __init__(self, soubor="hlavni.json"):
        if isinstance(soubor, dict):
            self.data = soubor
        else:
            soubor = Path(soubor)
            if not soubor.is_absolute():  # relativně k data/, případně k simulátoru nebo cwd
                soubor = next((p for p in (DATA / soubor, DATA.parent / soubor) if p.exists()), soubor)
            with open(soubor, encoding="utf-8") as f:
                self.data = json.load(f)
        d = self.data
        self.n = len(d["dily"])
        self.sirka, self.L = int(d["sirka"]), int(d["delka_max"])
        self.mezera, self.tol = int(d["mezera"]), float(d["tolerance_vlaken"])
        self.vlaknovy = np.array([dl["vlakna"] for dl in d["dily"]])
        self.plocha = np.array([_plocha_teziste(np.array(dl["polygon"]))[0] for dl in d["dily"]])
        self.vyhodnoceni = 0
        self._masky, self._povoleno = {}, {}
        # Mřížka obsazenosti: řádek r = y + g, sloupec c = x + g, kde g = mezera.
        # Okraj pásu (g řádků dole i nahoře, g sloupců vlevo) je „obsazený“,
        # vady jsou obsazené kruhy; vše, co se položí, se zvětší o mezeru.
        g = self.g = self.mezera
        self.H, self.W = self.sirka + 2 * g, self.L + 2 * g
        yy, xx = np.mgrid[0:self.sirka, 0:self.L] + 0.5
        vada = np.zeros((self.sirka, self.L), bool)
        for vx, vy, vr in d["vady"]:
            vada |= (xx - vx) ** 2 + (yy - vy) ** 2 <= (vr + 0.71) ** 2
        self._zvetsi = np.ones((2 * g + 1, 2 * g + 1), bool)
        zaklad = np.zeros((self.H, self.W))
        zaklad[g:g + self.sirka, g:g + self.L] = binary_dilation(vada, self._zvetsi) if g else vada
        zaklad[:g], zaklad[self.H - g:], zaklad[:, :g] = 1, 1, 1
        self._zaklad = zaklad
        self.Lw = next_fast_len(min(int(OKNO * self.plocha.sum() / self.sirka) + 60, self.W))
        self._F0 = rfft2(zaklad[:, :self.Lw].astype(np.float32))
        # fázové posuny pro všechny možné posuny (posun v prostoru = násobení fází)
        ky, kx = np.arange(self.H), np.arange(self.Lw // 2 + 1)
        self._fy = np.exp(-2j * np.pi * np.outer(np.arange(self.H), ky) / self.H).astype(np.complex64)
        self._fx = np.exp(-2j * np.pi * np.outer(np.arange(self.Lw), kx) / self.Lw).astype(np.complex64)

    def _otoceny(self, i, rot):
        """Polygon dílu i otočený o rot stupňů, posunutý na levý dolní roh (0, 0)."""
        p = np.array(self.data["dily"][i]["polygon"], float)
        a = np.deg2rad(rot)
        p = p @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return p - p.min(axis=0)

    def polygon(self, i, rot, x, y):
        """Skutečný tvar dílu i s rotací rot položeného s posunem (x, y) [mm]."""
        return self._otoceny(i, rot) + [x, y]

    def theta(self, x, y):
        """Směr vláken v bodě (x, y) [stupně od osy x]."""
        v = self.data["vlakna"]
        t = np.full(np.shape(x), v["theta0"], float)
        for lx, ly, amp, phi in v["mody"]:
            t = t + amp * np.sin(2 * np.pi * (np.asarray(x) / lx + np.asarray(y) / ly) + phi)
        return t

    def odchylka(self, rot, x, y):
        """Úhel mezi osou cípu (směr rot) a vlákny v bodě (x, y), 0..90 stupňů."""
        d = np.mod(rot - self.theta(x, y), 180.0)
        return np.minimum(d, 180.0 - d)

    def maska(self, i, rot):
        """Rastrová maska dílu (řádky = y, sloupce = x), těžiště [mm] a předpočty FFT."""
        klic = (i, rot % 360)
        if klic not in self._masky:
            if rot % 15:
                raise ValueError("rotace musí být násobek 15 stupňů")
            p = self._otoceny(i, rot)
            w, h = int(np.ceil(p[:, 0].max() - 1e-9)), int(np.ceil(p[:, 1].max() - 1e-9))
            s = (np.arange(PODMRIZKA) + 0.5) / PODMRIZKA
            gy, gx = np.mgrid[0:h, 0:w]
            tvar = (h, w, PODMRIZKA, PODMRIZKA)
            bx = np.broadcast_to(gx[:, :, None, None] + s[None, None, None, :], tvar).ravel()
            by = np.broadcast_to(gy[:, :, None, None] + s[None, None, :, None], tvar).ravel()
            m = _uvnitr(np.c_[bx, by], p).reshape(h, w, -1).any(axis=2)
            vr = np.minimum(p.astype(int), [w - 1, h - 1])  # buňky s vrcholy (krajní body)
            m[vr[:, 1], vr[:, 0]] = True
            kus = np.pad(m, self.g)                      # díl zvětšený o mezeru
            if self.g:
                kus = binary_dilation(kus, self._zvetsi)
            Fm = np.conj(rfft2(m.astype(np.float32), s=(self.H, self.Lw)))
            Fk = rfft2(kus.astype(np.float32), s=(self.H, self.Lw))
            self._masky[klic] = (m, _plocha_teziste(p)[1], kus, Fm, Fk)
        return self._masky[klic]

    def _povolene_pozice(self, i, rot):
        """Kde smí být levý dolní roh cípu (v souřadnicích mřížky), aby vlákna
        v těžišti cípu seděla."""
        klic = (i, rot % 360)
        if klic not in self._povoleno:
            m, (cx, cy) = self.maska(i, rot)[:2]
            h, w = m.shape
            rr, cc = np.mgrid[0:self.H - h + 1, 0:self.W - w + 1]
            self._povoleno[klic] = self.odchylka(rot, cc - self.g + cx, rr - self.g + cy) <= self.tol
        return self._povoleno[klic]

    def rozloz(self, poradi, rotace, vlakna=False):
        poradi = np.asarray(poradi, int)
        rotace = np.asarray(rotace, int) % 360
        if sorted(poradi.tolist()) != list(range(self.n)) or rotace.shape != (self.n,):
            raise ValueError("poradi musí být permutace 0..n-1 a rotace pole délky n")
        self.vyhodnoceni += 1
        g = self.g
        obs = self._zaklad.copy()        # obsazenost celé role
        F = self._F0.copy()              # její 2D FFT v okně prvních Lw sloupců
        v_okne = True
        delka, rozlozeni, odch = 0, [], np.full(self.n, np.nan)
        neuspech = {"delka": np.inf, "rozlozeni": rozlozeni, "odchylka_vlaken": odch}
        for i in poradi:
            m, (cx, cy), kus, Fm, Fk = self.maska(i, rotace[i])
            h, w = m.shape
            if h > self.sirka:
                return neuspech
            pos = None
            for rychle in ((True, False) if v_okne else (False,)):
                if rychle:   # překryv se všemi pozicemi naráz: korelace přes FFT
                    c = irfft2(F * Fm, s=(self.H, self.Lw))[:self.H - h + 1, :self.Lw - w - g + 1]
                else:
                    c = fftconvolve(obs, m[::-1, ::-1].astype(float), mode="valid")
                c = c < 0.5
                if vlakna and self.vlaknovy[i]:
                    c &= self._povolene_pozice(i, rotace[i])[:, :c.shape[1]]
                sloupce = np.nonzero(c.any(axis=0))[0]
                if len(sloupce):
                    pos = (int(np.nonzero(c[:, sloupce[0]])[0][0]), int(sloupce[0]))
                    break
                v_okne = False
            if pos is None:
                return neuspech
            r, s = pos                    # levý dolní roh masky v mřížce
            kh, kw = kus.shape
            kh2, kw2 = min(kh, self.H - (r - g)), min(kw, self.W - (s - g))
            obs[r - g:r - g + kh2, s - g:s - g + kw2] += kus[:kh2, :kw2]
            if v_okne:                    # posun v prostoru = fáze ve frekvencích
                F += Fk * self._fy[r - g][:, None] * self._fx[s - g][None, :]
            x, y = s - g, r - g
            delka = max(delka, x + w)
            if self.vlaknovy[i]:
                odch[i] = self.odchylka(rotace[i], x + cx, y + cy)
            rozlozeni.append((int(i), int(rotace[i]), x, y))
        return {"delka": float(delka), "rozlozeni": rozlozeni, "odchylka_vlaken": odch}
