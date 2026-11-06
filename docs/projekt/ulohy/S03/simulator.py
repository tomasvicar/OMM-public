"""Klasický řetězec segmentace jader buněk a metriky kvality (simulační úloha S03).

Řetězec (bez učení, jen NumPy/SciPy):
  1. normalizace jasu (dělení 99,5. percentilem),
  2. odečtení pozadí: od snímku se odečte jeho silné Gaussovo rozmazání
     (sigma = `pozadi` px; při `pozadi` < 5 se krok vynechá),
  3. vyhlazení Gaussovým filtrem (sigma = `sigma` px),
  4. prahování: práh = Otsuův práh + `prah` * směrodatná odchylka snímku,
  5. vyplnění děr,
  6. distanční transformace popředí, semena = lokální maxima vzdálenosti
     v okně (2 * `semena` + 1) px (celé číslo); každá souvislá oblast dostane
     aspoň jedno semeno,
  7. zjednodušený watershed z těchto semen (pixel dostane značku nejbližšího
     semena ve své oblasti), mezi sousedními objekty dělicí čára 1 px
     (jako příkaz Watershed ve Fiji; vypíná se DELICI_CARA = False),
  8. odstranění objektů menších než `min_plocha` px (nečistoty).

Rozhraní
--------
Parametry jsou slovník (nebo vektor v pořadí NAZVY) s 5 hodnotami
v mezích MEZE; celočíselný parametr `semena` se zaokrouhlí.

    snimky, pravda = nacti("trenink")          # data/snimky-trenink.npz
    znacky = segmentuj(snimky[0], parametry)   # 0 = pozadí, 1..K = objekty
    m = metriky(znacky, pravda[0])             # dice, f1, chyba_poctu, ...
    hodnota = vyhodnot(parametry, snimky, pravda, metrika="dice")

`vyhodnot` vrací průměr zvolené metriky přes všechny snímky (pro "dice"
a "f1" čím víc, tím lépe; pro "chyba_poctu" čím míň, tím lépe). Jedno
vyhodnocení na 8 snímcích trvá zhruba 0,2 s (medián 0,17 s na jednom
jádře nevytíženého stroje, 0,3–0,45 s pod zátěží). Funkce je
deterministická.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

NAZVY = ["pozadi", "sigma", "prah", "semena", "min_plocha"]
MEZE = {
    "pozadi": (0.0, 80.0),     # px, < 5 = bez odečtení pozadí
    "sigma": (0.0, 4.0),       # px
    "prah": (-1.5, 1.0),       # násobek směrodatné odchylky
    "semena": (1, 25),         # px, celé číslo
    "min_plocha": (0.0, 400.0),  # px
}
CELA = {"semena"}
VYCHOZI = {"pozadi": 0.0, "sigma": 1.0, "prah": 0.0,
           "semena": 5, "min_plocha": 0.0}
DELICI_CARA = True   # dělicí čára 1 px mezi sousedními objekty (False = bez ní)
DATA = Path(__file__).parent / "data"


def nacti(sada="trenink"):
    """Vrátí (snimky, pravda) ze souboru data/snimky-<sada>.npz."""
    d = np.load(DATA / f"snimky-{sada}.npz")
    return d["snimky"], d["pravda"]


def _parametry(p):
    if not isinstance(p, dict):
        p = dict(zip(NAZVY, np.asarray(p, float)))
    q = {}
    for k in NAZVY:
        lo, hi = MEZE[k]
        v = float(np.clip(p[k], lo, hi))
        q[k] = int(round(v)) if k in CELA else v
    return q


def otsu(x, biny=256):
    """Otsuův práh: maximalizuje rozptyl mezi třídami histogramu."""
    h, hrany = np.histogram(x, bins=biny)
    stredy = 0.5 * (hrany[1:] + hrany[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * stredy)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return stredy[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]


def segmentuj(snimek, parametry):
    """Segmentuje jeden snímek, vrací pole značek (0 = pozadí, 1..K)."""
    p = _parametry(parametry)
    x = snimek.astype(float)
    x /= max(np.percentile(x, 99.5), 1e-9)
    if p["pozadi"] >= 5:
        # pozadí se kvůli rychlosti počítá na 4x zmenšeném snímku
        male = ndi.gaussian_filter(x[2::4, 2::4], p["pozadi"] / 4, mode="nearest")
        x = x - ndi.zoom(male, 4, order=1, grid_mode=True, mode="nearest")
    if p["sigma"] > 0:
        x = ndi.gaussian_filter(x, p["sigma"])
    popredi = x > otsu(x) + p["prah"] * x.std()
    # vyplnění děr: pozadí, které nesouvisí s okrajem snímku, je díra
    dira, _ = ndi.label(~popredi)
    okraj = np.unique(np.r_[dira[0], dira[-1], dira[:, 0], dira[:, -1]])
    popredi |= ~np.isin(dira, okraj)
    if not popredi.any():
        return np.zeros(x.shape, np.int32)
    dist = ndi.distance_transform_edt(popredi)
    # semena: lokální maxima vzdálenosti v okně (2d+1) x (2d+1)
    d = p["semena"]
    dist_s = ndi.gaussian_filter(dist, 1.0)
    vrchol = (dist_s == ndi.maximum_filter(dist_s, size=2 * d + 1)) & popredi
    oblasti, n_obl = ndi.label(popredi)
    semena, n_sem = ndi.label(vrchol)
    # oblast bez semena dostane semeno v bodě maximální vzdálenosti
    ma_semeno = np.zeros(n_obl + 1, bool)
    ma_semeno[oblasti[semena > 0]] = True
    bez = np.flatnonzero(~ma_semeno[1:]) + 1
    if len(bez):
        for pozice in ndi.maximum_position(dist, oblasti, bez):
            n_sem += 1
            semena[pozice] = n_sem
    # rozdělení oblasti mezi semena (zjednodušený watershed): každý pixel
    # dostane značku nejbližšího semena ve stejné souvislé oblasti
    obl_semene = np.zeros(n_sem + 1, int)            # oblast každého semene
    obl_semene[semena.ravel()] = oblasti.ravel()
    pocet = np.bincount(obl_semene[1:], minlength=n_obl + 1)
    jedine = np.zeros(n_obl + 1, int)                # oblast s jediným semenem
    jedine[obl_semene[1:]] = np.arange(1, n_sem + 1)
    jedine[pocet != 1] = 0
    znacky = jedine[oblasti]
    okna = ndi.find_objects(oblasti)
    for o in np.flatnonzero(pocet > 1):
        okno = okna[o - 1]
        v_obl = oblasti[okno] == o
        sem = np.where(v_obl, semena[okno], 0)
        _, (iy, ix) = ndi.distance_transform_edt(sem == 0, return_indices=True)
        znacky[okno][v_obl] = sem[iy, ix][v_obl]
    # dělicí čára 1 px: pixel, jehož pravý/dolní soused má jinou nenulovou značku
    hrana = np.zeros_like(popredi)
    for a, b, h in ((znacky[:, :-1], znacky[:, 1:], hrana[:, :-1]),
                    (znacky[:-1, :], znacky[1:, :], hrana[:-1, :])):
        h |= (a != b) & (a > 0) & (b > 0)
    if DELICI_CARA:
        znacky[hrana] = 0
    # odstranění malých objektů a přečíslování 1..K
    plochy = np.bincount(znacky.ravel())
    plochy[0] = 0
    zustava = plochy >= max(p["min_plocha"], 1)
    zustava[0] = False
    nove = np.zeros(len(plochy), np.int32)
    nove[zustava] = np.arange(1, zustava.sum() + 1)
    return nove[znacky]


def metriky(znacky, pravda, prah_iou=0.5):
    """Pixelový Dice, objektové F1 (párování při IoU > prah) a chyba počtu.

    Prah musí být aspoň 0,5: jen tehdy má každý objekt nejvýš jeden pár
    a počet dvojic nad prahem je počet zásahů (při nižším prahu by jeden
    objekt mohl mít víc párů a F1 by vyšlo přes 1).
    """
    if not prah_iou >= 0.5:
        raise ValueError("prah_iou musí být aspoň 0,5 (jinak párování není jednoznačné)")
    a, b = znacky > 0, pravda > 0
    dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    n_p, n_g = int(znacky.max()), int(pravda.max())
    # průniky všech dvojic (predikce, pravda) najednou
    prunik = np.bincount(znacky.ravel() * (n_g + 1) + pravda.ravel(),
                         minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    plocha_p = prunik.sum(1)[1:, None]
    plocha_g = prunik.sum(0)[None, 1:]
    p = prunik[1:, 1:]
    iou = p / np.maximum(plocha_p + plocha_g - p, 1)
    # IoU > 0,5 je párování vždy jednoznačné (každý objekt nejvýš jeden pár)
    tp = int((iou > prah_iou).sum())
    f1 = 2 * tp / max(n_p + n_g, 1)
    return {"dice": float(dice), "f1": float(f1),
            "chyba_poctu": abs(n_p - n_g) / max(n_g, 1),
            "pocet": n_p, "pocet_pravda": n_g, "tp": tp}


def vyhodnot(parametry, snimky, pravda, metrika="dice"):
    """Průměr metriky přes snímky pro danou sadu parametrů."""
    hodnoty = [metriky(segmentuj(s, parametry), g)[metrika]
               for s, g in zip(snimky, pravda)]
    return float(np.mean(hodnoty))


if __name__ == "__main__":
    import time
    S, G = nacti("trenink")
    t = time.perf_counter()
    m = [metriky(segmentuj(s, VYCHOZI), g) for s, g in zip(S, G)]
    print(f"výchozí parametry {VYCHOZI}")
    for k in ("dice", "f1", "chyba_poctu"):
        print(f"  {k}: {np.mean([x[k] for x in m]):.3f}")
    print(f"  čas vyhodnocení na {len(S)} snímcích: {time.perf_counter() - t:.2f} s")
