"""Simulátor chladicí krystalizace s očkováním (úloha S15).

Vsádkový krystalizátor se ochladí za 120 min z 50 °C na 10 °C. Průběh
teploty určuje 8 čísel: teploty v časech 13,3; 26,7; ...; 106,7 min
(rovnoměrně mezi začátkem a koncem, začátek 50 °C a konec 10 °C jsou
pevné). Mezi uzly se teplota mění lineárně.

Rozhraní:

    zmer(teploty, rng, instance=0)  -> ZMĚŘENÁ střední velikost L43 [µm]
        Tak vypadá výsledek vsádky v provozu: vzorek suspenze změřený
        laserovou difrakcí, náhodná relativní chyba se směrodatnou
        odchylkou SUM_MERENI (2 %). rng = np.random.default_rng(seed).
        Optimalizace smí používat jen tohle; rozpočet počítá volání zmer.
    simuluj(teploty, instance=0)    -> skutečná L43 [µm] bez chyby měření
        Pravda jen pro vyhodnocení výsledku (v provozu ji nikdo nezná).
    simuluj(..., podrobne=True)     -> slovník s průběhy v čase
    poruseni(teploty)               -> (vzestupy [K], prekroceni_rychlosti [K])
    profil(teploty)                 -> (casy uzlu, teploty uzlu) vč. 50 a 10 °C
    parametry(instance=0)           -> látkové a kinetické parametry

`teploty` je pole 8 čísel (simulátor přijme i jiný počet uzlů m, uzly jsou
pak rovnoměrně ve 120 min). Simulátor spočítá i profil, který omezení
porušuje – přípustnost je věc toho, kdo profil navrhuje: teplota nesmí
stoupat a nesmí klesat rychleji než MAX_RYCHLOST = 0,5 K/min. Instance 0
je hlavní látka, instance 1, 2, ... jsou jiné látky.

Model (populační bilance metodou momentů, růst nezávislý na velikosti):

* rozpustnost c*(T) = c50 * exp(beta * (T - 50))       [g látky / g rozpouštědla]
* relativní přesycení s = (c - c*) / c*
* růst krystalů G = kg * s^g pro s > 0, jinak 0        [µm/min]
* nukleace B = kb * s^b * logist((s - s_met) / w)       [1 / (g rozp. · min)],
  pod metastabilní mezí s_met téměř nulová, nad ní prudce roste
* zárodky mají na začátku všechny velikost L0 a rostou stejně,
  nové krystaly z nukleace se sledují momenty m0 ... m4
* bilance látky dc/dt = -3 rho kv G (N_s L_s^2 + m2)
* L43 = m4 / m3 celé populace (hmotnostně vážená střední velikost)
Integrace RK4 s krokem 1 min, jedno vyhodnocení trvá jednotky ms.
"""
import math

import numpy as np

T_START, T_KONEC = 50.0, 10.0          # °C
DOBA = 120.0                           # min
N_UZLU = 8                             # volné uzly profilu
MAX_RYCHLOST = 0.5                     # K/min, výkon pláště
SUM_MERENI = 0.02                      # relativní chyba měření L43
KROK = 1.0                             # min, krok RK4
RHO, KV = 1.3e-12, 0.52                # g/µm^3, tvarový faktor (koule)


def parametry(instance=0):
    """Látkové a kinetické parametry instance (0 = hlavní)."""
    p = dict(c50=0.30, beta=math.log(3.0) / 40.0,  # c*(10 °C) = 0.10 g/g
             kg=150.0, g=1.5,                        # µm/min
             kb=1.0e8, b=3.0, s_met=0.05, w=0.002,
             m_seed=0.006, L0=50.0)                  # g zárodků / g rozp., µm
    if instance:
        r = np.random.default_rng(1000 + instance)
        p["beta"] *= r.uniform(0.95, 1.15)
        p["kg"] *= r.uniform(0.7, 1.4)
        p["g"] = r.uniform(1.3, 1.8)
        p["kb"] *= math.exp(r.uniform(-1.0, 1.0))
        p["s_met"] *= r.uniform(0.8, 1.1)
        p["m_seed"] *= r.uniform(0.7, 1.2)
    return {k: float(v) for k, v in p.items()}


def profil(teploty):
    """Uzly profilu včetně pevného začátku a konce: (casy, teploty)."""
    T = np.r_[T_START, np.asarray(teploty, float), T_KONEC]
    return np.linspace(0.0, DOBA, len(T)), T


def poruseni(teploty):
    """Součet vzestupů teploty a součet překročení max. poklesu za úsek [K]."""
    casy, T = profil(teploty)
    d = -np.diff(T)                                   # poklesy v úsecích
    limit = MAX_RYCHLOST * (casy[1] - casy[0])
    return float(np.clip(-d, 0, None).sum()), float(np.clip(d - limit, 0, None).sum())


def simuluj(teploty, instance=0, podrobne=False):
    """Skutečná střední velikost L43 [µm] na konci vsádky (bez chyby měření)."""
    p = parametry(instance)
    Tu = [float(v) for v in profil(teploty)[1]]
    du = DOBA / (len(Tu) - 1)
    Ns = p["m_seed"] / (RHO * KV * p["L0"] ** 3)      # počet zárodků / g rozp.

    def teplota(t):                                   # lineárně mezi uzly
        k = min(int(t / du), len(Tu) - 2)
        a = t / du - k
        return (1 - a) * Tu[k] + a * Tu[k + 1]

    def rhs(y, t):
        c, Ls, m0, m1, m2, m3, m4 = y
        cs = p["c50"] * math.exp(p["beta"] * (teplota(t) - 50.0))
        s = max((c - cs) / cs, 0.0)                   # přesycení, jen kladná část
        G = p["kg"] * s ** p["g"]                     # růst
        z = min(max((s - p["s_met"]) / p["w"], -50.0), 50.0)
        B = p["kb"] * s ** p["b"] / (1.0 + math.exp(-z))   # nukleace
        dc = -3 * RHO * KV * G * (Ns * Ls ** 2 + m2)
        return (dc, G, B, G * m0, 2 * G * m1, 3 * G * m2, 4 * G * m3), s, B

    y = (p["c50"], p["L0"], 0.0, 0.0, 0.0, 0.0, 0.0)  # c, L_s, m0..m4
    h, zaznam = KROK, []
    for i in range(int(round(DOBA / h))):
        t = i * h
        k1, s, B = rhs(y, t)
        zaznam.append((t, teplota(t), s, B))
        k2 = rhs([a + 0.5 * h * b for a, b in zip(y, k1)], t + 0.5 * h)[0]
        k3 = rhs([a + 0.5 * h * b for a, b in zip(y, k2)], t + 0.5 * h)[0]
        k4 = rhs([a + h * b for a, b in zip(y, k3)], t + h)[0]
        y = tuple(a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
                  for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4))
    c, Ls, m0, m1, m2, m3, m4 = y
    M3 = Ns * Ls ** 3 + m3
    L43 = (Ns * Ls ** 4 + m4) / M3
    if not podrobne:
        return L43
    t, T, s, B = (np.array(v) for v in zip(*zaznam))
    return dict(L43=L43, cas=t, T=T, presyceni=s, nukleace=B,
                podil_jemnych=m3 / M3,                # hmotnostní podíl z nukleace
                velikost_zarodku=Ls, pocet_novych=m0, pocet_zarodku=Ns)


def zmer(teploty, rng, instance=0):
    """Změřená L43 [µm]: skutečná hodnota s relativní chybou SUM_MERENI."""
    return simuluj(teploty, instance) * (1.0 + SUM_MERENI * rng.standard_normal())
