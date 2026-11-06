"""Účelová funkce úlohy S18: jak moc „vypadá česky“ text dešifrovaný klíčem.

Abeceda je 26 velkých písmen A–Z bez diakritiky, bez mezer a interpunkce
(„CH“ jsou dvě písmena C, H). Písmena se v polích kódují čísly 0–25.

Šifry
-----
* **substituce** – klíč je permutace 26 písmen ve tvaru *dešifrovacím*:
  ``klic[j]`` je otevřené písmeno, které odpovídá šifrovému písmenu
  ``ABC[j]``. Klíč ``"ABC…Z"`` text nemění.
* **Playfair** – klíč je čtverec 5 × 5 zapsaný po řádcích jako řetězec
  25 písmen (abeceda bez W; W se před šifrováním píše jako V). Text se šifruje
  po dvojicích: stejný řádek -> písmena vpravo, stejný sloupec -> písmena pod,
  jinak rohy obdélníku ve stejném řádku. Dešifrování je opačné (vlevo, nad,
  rohy). Šifrovaný text má sudou délku a ve dvojici nikdy dvě stejná písmena.

Skóre
-----
``skore = sum_k log10 P(t_k t_{k+1} t_{k+2} t_{k+3})`` přes všechny
tetragramy dešifrovaného textu ``t``; tabulka P je spočtená z jiných českých
textů (``data/tetragramy.npy``, pole 26**4 hodnot, index
``((a*26 + b)*26 + c)*26 + d``). Vyšší = českější. Skóre nezná správný klíč.

Rozhraní
--------
``nacti_instance(id)``           -> slovník {id, typ, delka, sifra}
``skore(klic, sifra)``           -> float, substituce   (1 vyhodnocení)
``skore_playfair(ctverec, sifra)``-> float, Playfair    (1 vyhodnocení)
``desifruj(klic, sifra)``, ``desifruj_playfair(ctverec, sifra)`` -> str
``POCET``                        -> počítadlo vyhodnocení skóre (lze nulovat)

``klic`` / ``ctverec`` i ``sifra`` smí být řetězec, nebo pole čísel 0–25
(rychlejší). Funkce jsou deterministické, jedno vyhodnocení trvá desítky µs.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ABC_PF = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
_TAB = None
POCET = [0]


def tabulka():
    global _TAB
    if _TAB is None:
        _TAB = np.load(DATA / "tetragramy.npy")
    return _TAB


def nacti_instance(iid=None, soubor=DATA / "instance.json"):
    """Bez argumentu vrátí seznam všech instancí, jinak jednu podle id."""
    inst = json.loads(Path(soubor).read_text(encoding="utf-8"))["instance"]
    return inst if iid is None else next(i for i in inst if i["id"] == iid)


def na_cisla(s):
    if isinstance(s, str):
        return np.frombuffer(s.encode(), dtype=np.uint8).astype(np.intp) - 65
    return np.asarray(s, dtype=np.intp)


def na_text(a):
    return (np.asarray(a, dtype=np.uint8) + 65).tobytes().decode()


def skore_textu(t):
    """Skóre otevřeného textu (pole čísel 0–25 nebo řetězec)."""
    t = na_cisla(t)
    idx = ((t[:-3] * 26 + t[1:-2]) * 26 + t[2:-1]) * 26 + t[3:]
    return float(tabulka()[idx].sum())


# ---------------------------------------------------------------- substituce
def desifruj_cisla(klic, sifra):
    return na_cisla(klic)[na_cisla(sifra)]


def desifruj(klic, sifra):
    return na_text(desifruj_cisla(klic, sifra))


def skore(klic, sifra):
    POCET[0] += 1
    return skore_textu(desifruj_cisla(klic, sifra))


# ---------------------------------------------------------------- Playfair
_PF_KOD = np.array([ABC_PF.index(z) if z in ABC_PF else -1 for z in ABC])   # A–Z -> 0–24
_PF_ZPET = na_cisla(ABC_PF)                                                # 0–24 -> A–Z
# Dešifrování dvojice závisí jen na polohách obou písmen ve čtverci (políčka
# 0–24, řádek = k // 5, sloupec = k % 5): stejný řádek -> o políčko vlevo,
# stejný sloupec -> o políčko výš, jinak rohy obdélníku. Tabulka pro všech
# 25 × 25 dvojic poloh se spočte jednou.
_r1, _c1 = np.divmod(np.repeat(np.arange(25), 25), 5)
_r2, _c2 = np.divmod(np.tile(np.arange(25), 25), 5)
_RADEK, _SLOUPEC = _r1 == _r2, (_c1 == _c2) & (_r1 != _r2)
_DEC1 = np.where(_RADEK, _r1 * 5 + (_c1 - 1) % 5,
                 np.where(_SLOUPEC, ((_r1 - 1) % 5) * 5 + _c1, _r1 * 5 + _c2))
_DEC2 = np.where(_RADEK, _r2 * 5 + (_c2 - 1) % 5,
                 np.where(_SLOUPEC, ((_r2 - 1) % 5) * 5 + _c2, _r2 * 5 + _c1))


def desifruj_playfair_cisla(ctverec, sifra):
    sq = _PF_KOD[na_cisla(ctverec)]               # políčko -> písmeno 0–24
    pos = np.empty(25, dtype=np.intp)
    pos[sq] = np.arange(25)                       # písmeno -> políčko
    p = pos[_PF_KOD[na_cisla(sifra)]]
    d = p[0::2] * 25 + p[1::2]                    # dvojice poloh -> řádek tabulky
    out = np.empty(len(p), dtype=np.intp)
    out[0::2], out[1::2] = sq[_DEC1[d]], sq[_DEC2[d]]
    return _PF_ZPET[out]


def desifruj_playfair(ctverec, sifra):
    return na_text(desifruj_playfair_cisla(ctverec, sifra))


def skore_playfair(ctverec, sifra):
    POCET[0] += 1
    return skore_textu(desifruj_playfair_cisla(ctverec, sifra))


if __name__ == "__main__":
    import time
    from string import ascii_uppercase
    inst = nacti_instance("S1000_00")
    t0 = time.perf_counter()
    for _ in range(2000):
        skore(ascii_uppercase, inst["sifra"])
    print(f"substituce: {(time.perf_counter() - t0) / 2000 * 1e6:.0f} µs/vyhodnocení (řetězce)")
    p = nacti_instance("P1000_00")
    t0 = time.perf_counter()
    for _ in range(2000):
        skore_playfair(ABC_PF, p["sifra"])
    print(f"Playfair:   {(time.perf_counter() - t0) / 2000 * 1e6:.0f} µs/vyhodnocení (řetězce)")
