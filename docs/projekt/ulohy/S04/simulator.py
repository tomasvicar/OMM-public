"""Simulátor repressilátoru pro úlohu S04 (Hodiny z bakterie).

Repressilátor (Elowitz a Leibler, 2000) jsou tři geny v kruhu: protein 3
potlačuje gen 1, protein 1 gen 2 a protein 2 gen 3. Model v bezrozměrném
čase tau = k * t (k je rychlost rozpadu mRNA v 1/h, t čas v hodinách):

    dm_i/dtau = -m_i + alpha / (1 + p_{i-1}^n) + ALPHA0
    dp_i/dtau = -beta * (p_i - m_i)            i = 1, 2, 3 (p_0 = p_3)

Měří se fluorescence reportéru navázaného na protein 1:  F(t) = s * p_1(t).
Měření začíná po indukci, která obvod nastaví do známého stavu X0
(mRNA nulová, hodně proteinu 1, málo proteinů 2 a 3).

Rozhraní
--------
theta : pole 5 čísel (pořadí viz NAZVY), čtyři z nich v log10:
    0  log10 alpha  maximální rychlost přepisu (síla promotoru)
    1  n            Hillův koeficient (kooperativita represe)
    2  log10 beta   poměr rychlosti rozpadu proteinu a mRNA
    3  log10 k      rychlost rozpadu mRNA [1/h], určuje, jak rychle hodiny jdou
    4  log10 s      jas reportéru (fluorescence na jednotku proteinu)
Rozsahy úlohy jsou DOLNI a HORNI.

simuluj(theta, t)       -> fluorescence F v časech t (h), pole tvaru t
simuluj(theta, t, True) -> všech šest stavů, pole (6, len(t)): m1..m3, p1..p3
chyba(theta, t, y)      -> součet čtverců rozdílů log(F) a log(y) přes body
perioda(theta)          -> perioda kmitů reportéru v hodinách (z 12–48 h),
                           NaN, když model nekmitá

Výpočet je deterministický. Selže-li integrátor nebo vyjde nekladná
fluorescence, chyba() vrátí POKUTA.
"""
import warnings

import numpy as np
from scipy.integrate import odeint

NAZVY = ["log10_alpha", "n", "log10_beta", "log10_k", "log10_s"]
DOLNI = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
HORNI = np.array([3.5, 4.0, 1.0, 0.8, 2.0])
ALPHA0 = 0.2                             # netěsnost promotoru (pevná)
X0 = [0.0, 0.0, 0.0, 20.0, 1.0, 1.0]     # stav po indukci: m1..m3, p1..p3
POKUTA = 1e6                             # hodnota chyby, když simulace selže


def _prava_strana(x, tau, alpha, n, beta):
    m1, m2, m3, p1, p2, p3 = x.tolist()
    return [
        -m1 + alpha / (1.0 + p3**n) + ALPHA0,
        -m2 + alpha / (1.0 + p1**n) + ALPHA0,
        -m3 + alpha / (1.0 + p2**n) + ALPHA0,
        -beta * (p1 - m1),
        -beta * (p2 - m2),
        -beta * (p3 - m3),
    ]


def simuluj(theta, t, vse=False):
    theta = np.asarray(theta, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha, beta, k, s = 10.0 ** theta[[0, 2, 3, 4]]
    poradi = np.argsort(t)               # integruje se od nuly v seřazených časech
    tau = np.concatenate([[0.0], k * t[poradi]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = odeint(_prava_strana, X0, tau, args=(alpha, theta[1], beta),
                   rtol=1e-5, atol=1e-7, mxstep=5000)
    vysledek = np.empty((6, len(t)))
    vysledek[:, poradi] = np.maximum(x[1:], 0.0).T
    return vysledek if vse else s * vysledek[3]


def chyba(theta, t, y):
    """Součet čtverců rozdílů logaritmů simulace a měření."""
    try:
        F = simuluj(theta, t)
    except Exception:
        return POKUTA
    if not np.all(np.isfinite(F)) or np.any(F <= 0):
        return POKUTA
    return float(np.sum((np.log(F) - np.log(y)) ** 2))


def perioda(theta):
    """Perioda reportéru [h]: průměrný odstup vzestupných průchodů log F jeho
    průměrem v okně 12–48 h. NaN, když se F v okně mění méně než 1,5krát
    nebo okno neobsahuje aspoň dva vzestupné průchody."""
    t = np.arange(0.0, 48.0001, 0.05)
    try:
        F = simuluj(theta, t)[t >= 12.0]
    except Exception:
        return float("nan")
    if not np.all(np.isfinite(F)) or F.min() <= 0 or F.max() / F.min() < 1.5:
        return float("nan")
    g = np.log(F) - np.mean(np.log(F))
    i = np.where((g[:-1] < 0) & (g[1:] >= 0))[0]
    if len(i) < 2:
        return float("nan")
    cas = 12.0 + 0.05 * (i + g[i] / (g[i] - g[i + 1]))   # lineární interpolace
    return float(np.mean(np.diff(cas)))


if __name__ == "__main__":
    import time
    theta = np.array([2.25, 2.5, 0.0, -0.1, 0.5])   # střed rozsahů, ne pravé parametry
    t = np.arange(0, 48.01, 0.5)
    t0 = time.perf_counter()
    for _ in range(50):
        F = simuluj(theta, t)
    print(f"jedno vyhodnocení: {(time.perf_counter() - t0) / 50 * 1e3:.1f} ms")
    print("perioda:", round(perioda(theta), 2), "h")
