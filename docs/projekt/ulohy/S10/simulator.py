"""Simulátor úlohy S10: větrná farma v úplavech (black box).

Model (jednotky: metry, m/s, MW, GWh za rok):

* Turbína: průměr rotoru D = 130 m, jmenovitý výkon 3,35 MW (podle referenční
  turbíny IEA 37). Výkonová křivka: pod 4 m/s nula, mezi 4 a 9,8 m/s roste
  s třetí mocninou ((v - 4)/(9,8 - 4))^3, do 25 m/s jmenovitý výkon, nad 25 m/s
  turbína vypne. Součinitel tahu C_T = 8/9 (konstantní).
* Vítr: větrná růžice = spojité rozdělení směru (směs dvou von Misesových
  rozdělení, převládající a vedlejší směr) a Weibullovo rozdělení rychlosti
  (tvar 2, měřítko 8,5 m/s, stejné pro všechny směry). Směr se počítá
  meteorologicky: odkud vítr fouká, 270° = od západu. Růžice se pro výpočet
  rozdělí na n_sektoru stejně širokých sektorů; každý sektor se počítá jako
  jediný směr uprostřed sektoru s vahou rovnou pravděpodobnosti sektoru.
* Úplav (Jensenův / Park model): turbína i zpomalí vítr v kuželi za sebou.
  Turbína j ve vzdálenosti d po proudu a c napříč je v úplavu, když
  d > 0 a |c| < r0 + k d (r0 = D/2, k = 0,075). Deficit rychlosti
      delta_ij = (1 - sqrt(1 - C_T)) / (1 + k d / r0)^2.
  Deficity od více turbín se skládají odmocninou součtu čtverců,
  rychlost turbíny j je u * (1 - delta_j).
* Roční výroba AEP = 8760 h * sum přes sektory a turbíny
  P(sektor) * E_u[ výkon(u * (1 - delta_j)) ]  (střední hodnota přes Weibulla).

Pozemek je obdélník [0, sirka] x [0, vyska]. Instance obsahuje i zakázané
zóny (kruh ochranného pásma obce a čtyřúhelník mokřadu), které se uplatní
jen v novém požadavku. Simulátor omezení NEVYNUCUJE – aep() spočítá výrobu
pro libovolné souřadnice; přípustnost hlásí funkce poruseni().

Funkce pro studenty:

    instance(seed=0) -> dict        seed 0 je hlavní instance, 1..4 další pozemky
    aep(xy, inst, n_sektoru=12) -> float      roční výroba v GWh;
        xy je pole tvaru (N, 2) se souřadnicemi turbín v metrech
    aep_turbin(xy, inst, n_sektoru=12) -> np.ndarray   výroba po turbínách
    poruseni(xy, inst, zony=True) -> dict  míry porušení omezení v metrech:
        "mimo_pozemek"  součet vzdáleností turbín mimo obdélník k jeho okraji,
        "v_zonach"      součet hloubek, do kterých turbíny zajíždějí do zón,
        "rozestup"      součet (MIN_ROZESTUP - vzdálenost) přes příliš blízké dvojice,
        "pripustne"     True, když jsou všechny tři míry <= TOLERANCE.
    v_zone(xy, inst) -> np.ndarray   hloubka průniku každé turbíny do zón (0 = venku)
    ruzice(inst, n_sektoru=12) -> (středy sektorů ve stupních, pravděpodobnosti)
    vykon(v) -> výkon turbíny v MW při rychlosti větru v (m/s)

Jedno vyhodnocení aep pro 16 turbín a 12 sektorů trvá desetiny milisekundy.
"""
from functools import lru_cache

import numpy as np

D_ROTOR = 130.0                      # m
R0 = D_ROTOR / 2
K_UPLAV = 0.075                      # rozšiřování úplavu (pevnina)
CT = 8 / 9
MIN_ROZESTUP = 4 * D_ROTOR           # m
TOLERANCE = 1.0                      # m
P_JMEN, V_IN, V_JMEN, V_OUT = 3.35, 4.0, 9.8, 25.0
WEIBULL_K, WEIBULL_A = 2.0, 8.5
HODIN = 8760.0


def vykon(v):
    """Výkonová křivka turbíny v MW."""
    v = np.asarray(v, dtype=float)
    p = np.where(v < V_JMEN, P_JMEN * ((v - V_IN) / (V_JMEN - V_IN)) ** 3, P_JMEN)
    return np.where((v < V_IN) | (v > V_OUT), 0.0, p)


# Střední výkon E_u[vykon(s * u)] jako funkce faktoru s = 1 - deficit (tabulka).
_u = np.linspace(0.0, 40.0, 8001)
_w = (WEIBULL_K / WEIBULL_A) * (_u / WEIBULL_A) ** (WEIBULL_K - 1) \
    * np.exp(-(_u / WEIBULL_A) ** WEIBULL_K) * (_u[1] - _u[0])
_S = np.linspace(0.0, 1.0, 1001)
_STREDNI = np.array([np.sum(vykon(s * _u) * _w) for s in _S])


def stredni_vykon(s):
    """E_u[vykon(s * u)] v MW pro faktor rychlosti s v [0, 1]."""
    return np.interp(s, _S, _STREDNI)


def instance(seed=0):
    """Pozemek, zakázané zóny a větrná růžice. seed 0 je hlavní instance."""
    g = np.random.default_rng(4300 + seed)
    if seed == 0:
        sirka, vyska, n = 3000.0, 2400.0, 16
        smer, smer2, vaha2 = 255.0, 140.0, 0.25
        mokrad_x, mokrad_sirka, sklon = 1950.0, 450.0, 150.0
        obec, obec_r = (0.0, 0.0), 650.0
    else:
        sirka = float(np.round(g.uniform(2700, 3300), -1))
        vyska = float(np.round(g.uniform(2100, 2700), -1))
        n = 16
        smer = float(np.round(g.uniform(225, 290)))
        smer2 = float(np.round((smer + g.uniform(150, 210)) % 360))
        vaha2 = float(np.round(g.uniform(0.15, 0.35), 2))
        mokrad_x = float(np.round(sirka * g.uniform(0.6, 0.7), -1))
        mokrad_sirka = float(np.round(g.uniform(400, 550), -1))
        sklon = float(np.round(g.uniform(-250, 250), -1))
        roh = int(g.integers(4))
        obec = ([0.0, sirka][roh % 2], [0.0, vyska][roh // 2])
        obec_r = float(np.round(g.uniform(550, 750), -1))
    # mokřad: šikmý pás přes celý pozemek (konvexní čtyřúhelník, proti směru hodin)
    h = mokrad_sirka / 2
    mokrad = np.array([[mokrad_x - h - sklon, -50.0], [mokrad_x + h - sklon, -50.0],
                       [mokrad_x + h + sklon, vyska + 50], [mokrad_x - h + sklon, vyska + 50]])
    return dict(seed=seed, sirka=sirka, vyska=vyska, n_turbin=n,
                smer=smer, smer2=smer2, vaha2=vaha2, koncentrace=4.0,
                obec=np.array(obec, dtype=float), obec_r=obec_r, mokrad=mokrad)


def ruzice(inst, n_sektoru=12):
    """Středy sektorů (stupně, odkud fouká) a jejich pravděpodobnosti."""
    return _ruzice(inst["smer"], inst["smer2"], inst["vaha2"], inst["koncentrace"], n_sektoru)


@lru_cache(maxsize=64)
def _ruzice(smer, smer2, vaha2, koncentrace, n_sektoru):
    jemne = np.arange(3600) / 10.0
    def vm(mu):
        return np.exp(koncentrace * np.cos(np.radians(jemne - mu)))
    f = (1 - vaha2) * vm(smer) / vm(smer).sum() + vaha2 * vm(smer2) / vm(smer2).sum()
    sirka = 360.0 / n_sektoru
    stredy = np.arange(n_sektoru) * sirka
    idx = np.floor(((jemne + sirka / 2) % 360) / sirka).astype(int)
    return stredy, np.bincount(idx, weights=f, minlength=n_sektoru)


def aep_turbin(xy, inst, n_sektoru=12):
    """Roční výroba jednotlivých turbín v GWh."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    stredy, p = ruzice(inst, n_sektoru)
    th = np.radians(stredy)
    # jednotkový vektor ve směru proudění (vítr fouká Z směru th)
    ex, ey = -np.sin(th), -np.cos(th)                     # (S,)
    dx = xy[None, :, 0] - xy[:, None, 0]                  # dx[i, j] = x_j - x_i
    dy = xy[None, :, 1] - xy[:, None, 1]
    d = dx[None] * ex[:, None, None] + dy[None] * ey[:, None, None]   # po proudu
    c = -dx[None] * ey[:, None, None] + dy[None] * ex[:, None, None]  # napříč
    dk = np.maximum(d, 1e-9)
    v_uplavu = (d > 0) & (np.abs(c) < R0 + K_UPLAV * dk)
    delta = np.where(v_uplavu, (1 - np.sqrt(1 - CT)) / (1 + K_UPLAV * dk / R0) ** 2, 0.0)
    deficit = np.sqrt((delta ** 2).sum(axis=1))           # (S, N): na turbínu j
    s = np.clip(1 - deficit, 0.0, 1.0)
    return HODIN * (p[:, None] * stredni_vykon(s)).sum(axis=0) / 1000.0


def aep(xy, inst, n_sektoru=12):
    """Roční výroba farmy v GWh (omezení se nekontrolují)."""
    return float(aep_turbin(xy, inst, n_sektoru).sum())


def _hloubka_konvexni(xy, poly):
    """Hloubka průniku bodů do konvexního polygonu (0 = venku)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    hrana = b - a
    normala = np.stack([hrana[:, 1], -hrana[:, 0]], axis=1)   # vnější normála (CCW)
    normala /= np.linalg.norm(normala, axis=1, keepdims=True)
    vzd = np.einsum("nek,ek->ne", xy[:, None, :] - a[None], normala)
    return np.maximum(-vzd.max(axis=1), 0.0)


def v_zone(xy, inst):
    """Hloubka průniku každé turbíny do zakázaných zón v metrech."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    obec = np.maximum(inst["obec_r"] - np.linalg.norm(xy - inst["obec"], axis=1), 0.0)
    return np.maximum(obec, _hloubka_konvexni(xy, inst["mokrad"]))


def poruseni(xy, inst, zony=True):
    """Míry porušení omezení (metry) a příznak přípustnosti."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    ven = np.maximum(-xy, 0) + np.maximum(xy - [inst["sirka"], inst["vyska"]], 0)
    mimo = float(np.linalg.norm(ven, axis=1).sum())
    zona = float(v_zone(xy, inst).sum()) if zony else 0.0
    vzd = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    iu = np.triu_indices(len(xy), 1)
    roz = float(np.maximum(MIN_ROZESTUP - vzd[iu], 0).sum())
    return dict(mimo_pozemek=mimo, v_zonach=zona, rozestup=roz,
                pripustne=bool(mimo <= TOLERANCE and zona <= TOLERANCE and roz <= TOLERANCE))
