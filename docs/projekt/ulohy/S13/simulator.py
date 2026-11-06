"""Klasifikátor bílých krvinek jako černá skříňka a „smítka“ na sklíčku.

Rozhraní
--------
Obrázek je pole (28, 28, 3) float v [0, 1] (řádek, sloupec, RGB), dávka
obrázků má tvar (n, 28, 28, 3). Třídy jsou TRIDY:
0 neutrofil, 1 eozinofil, 2 bazofil, 3 lymfocyt, 4 monocyt.

    obrazky, tridy = nacti_obrazky()      20 obrázků k útoku (4 na třídu) a jejich třídy

    k = Klasifikator()                    model (náhodný les ze souboru data/model.npz)
    k.pravdepodobnosti(obrazky) -> (n, 5) odhad pravděpodobnosti tříd
    k.trida(obrazky)            -> (n,)   předpovězená třída (argmax)
    k.dotazy                              kolik obrázků už model viděl
    Klasifikator(rozpocet=1000)           po 1000 obrázcích vyvolá RozpocetVycerpan

Každý obrázek v dávce je jeden dotaz, i když jde o tentýž obrázek
v jiné transformaci.

    nalep(obrazek, smitka) -> obrázek se smítky
        smitka: pole (..., m, 5), řádek = (radek, sloupec, r, g, b);
        smítko je čtvereček 2 × 2 pixely jedné barvy, (radek, sloupec) je jeho
        levý horní roh; zaokrouhlí se na celé pixely a ořízne do 0..26,
        barva musí být v [0, 1]. Smítko barvu pixelů nahradí, u překrývajících
        se smítek platí poslední. Dávka smítek (p, m, 5) dá dávku obrázků
        (p, 28, 28, 3).

    transformuj(obrazky, t) -> posunuté obrázky se změněným jasem
        t = (dy, dx, jas): posun o celé pixely (dy, dx v -2..2, okraj se doplní
        krajní hodnotou) a vynásobení jasu (0.9..1.1), oříznuto na [0, 1].
        Smítka jsou na sklíčku, takže se posunou spolu s buňkou.

    nahodne_transformace(n, seed) -> pole (n, 3) náhodných (dy, dx, jas)
        Kontrolní sada pro „fyzický“ útok je nahodne_transformace(100, seed=999);
        během hledání se na ni nesahá.

Model je deterministický. Pole v data/model.npz jsou čitelná, ale pro útok
platí pravidlo: strategie smí model jen volat, ne číst jeho stromy.
"""
from pathlib import Path

import numpy as np

TRIDY = ("neutrofil", "eozinofil", "bazofil", "lymfocyt", "monocyt")
N = 28
V = 2           # smítko je čtverec V × V pixelů
DATA = Path(__file__).parent / "data"


class RozpocetVycerpan(RuntimeError):
    pass


class Klasifikator:
    def __init__(self, rozpocet=None, soubor=DATA / "model.npz"):
        d = np.load(soubor)
        self.priznak, self.prah = d["priznak"], d["prah"]
        self.vlevo, self.vpravo, self.hodnota = d["vlevo"], d["vpravo"], d["hodnota"]
        self.hloubka = int(d["hloubka"])
        self.rozpocet = rozpocet
        self.dotazy = 0

    def _predikce(self, obrazky):
        x = np.asarray(obrazky, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        if x.shape[1:] != (N, N, 3):
            raise ValueError("obrázky musí mít tvar (n, 28, 28, 3)")
        n = len(x)
        if self.rozpocet is not None and self.dotazy + n > self.rozpocet:
            raise RozpocetVycerpan(f"rozpočet {self.rozpocet} dotazů vyčerpán")
        self.dotazy += n
        x = x.reshape(n, -1)
        T = len(self.prah)
        strom = np.arange(T)[:, None]
        uzel = np.zeros((T, n), dtype=np.int32)
        sl = np.arange(n)[None, :]
        for _ in range(self.hloubka):
            l, r = self.vlevo[strom, uzel], self.vpravo[strom, uzel]
            doleva = x[sl, self.priznak[strom, uzel]] <= self.prah[strom, uzel]
            uzel = np.where(l < 0, uzel, np.where(doleva, l, r))
        return self.hodnota[strom, uzel].mean(axis=0)          # (n, 5)

    def pravdepodobnosti(self, obrazky):
        return self._predikce(obrazky)

    def trida(self, obrazky):
        return self._predikce(obrazky).argmax(axis=1)


def nacti_obrazky():
    d = np.load(DATA / "obrazky.npz")
    return d["obrazky"], d["tridy"]


def nalep(obrazek, smitka):
    obrazek = np.asarray(obrazek, dtype=np.float32)
    s = np.asarray(smitka, dtype=float)
    if s.shape[-1] != 5:
        raise ValueError("smítko je (radek, sloupec, r, g, b)")
    davka = s.shape[:-2]
    s = s.reshape(-1, s.shape[-2], 5)
    if np.any(s[..., 2:] < 0) or np.any(s[..., 2:] > 1):
        raise ValueError("barva smítka musí být v [0, 1]")
    rad = np.clip(np.rint(s[..., 0]), 0, N - V).astype(int)
    sl = np.clip(np.rint(s[..., 1]), 0, N - V).astype(int)
    out = np.repeat(obrazek[None], len(s), axis=0)
    i = np.arange(len(s))
    for j in range(s.shape[1]):
        for a in range(V):
            for b in range(V):
                out[i, rad[:, j] + a, sl[:, j] + b] = s[:, j, 2:]
    return out.reshape(*davka, N, N, 3)


def transformuj(obrazky, t):
    dy, dx, jas = int(t[0]), int(t[1]), float(t[2])
    if abs(dy) > 2 or abs(dx) > 2:
        raise ValueError("posun nejvýš o 2 pixely")
    x = np.asarray(obrazky, dtype=np.float32)
    p = np.pad(x, [(0, 0)] * (x.ndim - 3) + [(2, 2), (2, 2), (0, 0)], mode="edge")
    p = p[..., 2 - dy:2 - dy + N, 2 - dx:2 - dx + N, :]
    return np.clip(p * jas, 0, 1)


def nahodne_transformace(n, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.integers(-2, 3, n), rng.integers(-2, 3, n), rng.uniform(0.9, 1.1, n)])
