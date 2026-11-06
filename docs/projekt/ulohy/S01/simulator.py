"""Simulátor S01: antibiotikum proti mutantovi (simulační úloha, black box).

Režim léčby: první (nasycovací) dávka D1, pak udržovací dávky D každých tau
hodin, léčba trvá vždy DNY_LECBY dní (první dávka v čase 0). Pak se sleduje
ještě DOBA_SLEDOVANI dní. Jednotky: čas v h, koncentrace v mg/l, dávka v mg.

Model:
* Farmakokinetika: jeden kompartment, nitrožilní bolus, poločas t_half:
      C(t) = sum_i D_i / V * exp(-k (t - t_i)),   k = ln 2 / t_half.
* Bakterie: citlivé S a rezistentní mutanti R. Na jednu buňku za hodinu:
      množení   b * (1 - N/K)               (N = S + R, mutant o cena_fitness pomaleji)
      smrt      d0 + lék + imunita
      lék       Emax C^h / (EC50^h + C^h)   (mutant má EC50 mic_pomer-krát větší)
      imunita   k_imm / (1 + N/N_imm)       (malou populaci dočistí, velkou nestíhá)
  Při dělení citlivé buňky vznikne s pravděpodobností mu mutant. Na začátku
  je N0 citlivých buněk a Poisson(m0) mutantů.

Funkce:
    instance(seed)                      parametry pacienta; seed 0 = hlavní pacient
    koncentrace(D1, D, tau, t, inst)    C(t) v časech t (h), pro grafy
    toxicita(D1, D, tau, inst)          (špička, minimum před poslední dávkou), mg/l
    celkem_leku(D1, D, tau)             D1 + (počet dávek - 1) * D, mg
    deterministicky(D1, D, tau, inst)   klasický model bez mutantů (střední hodnoty);
                                        True, když citlivé klesnou pod 1 buňku
    stochasticky(D1, D, tau, inst, n_behu, seed) -> pole výsledků n_behu běhů:
        0 = vyléčeno, 1 = přežily jen citlivé (nedoléčeno), 2 = rezistence.
        Tau-leaping s krokem DT; stejný seed = stejná náhodná čísla.
        Rozpočet úlohy se počítá v bězích (součet n_behu).

Meze proměnných (podle zadání): D1 v [20, 800] mg, D v [1, 800] mg, tau
z TAU_POVOLENE. Simulátor je nehlídá: odmítne jen nekladnou dávku a tau
mimo TAU_POVOLENE.

Model je výukový, není to klinické doporučení.
"""
import numpy as np

TAU_POVOLENE = (6, 8, 12, 24)   # h
DNY_LECBY = 3
DOBA_SLEDOVANI = 4              # dní po konci léčby
DT = 0.5                        # h, krok simulace


def instance(seed=0):
    if seed == 0:
        p = dict(t_half=6.0, mic_pomer=8.0, mu=1e-10, m0=0.3)
    else:
        g = np.random.default_rng(1000 + seed)
        p = dict(t_half=float(np.round(g.uniform(3.0, 10.0), 1)),
                 mic_pomer=float(np.round(2 ** g.uniform(2, 4), 1)),
                 mu=float(10 ** np.round(g.uniform(-10.5, -9.5), 2)),
                 m0=float(np.round(g.uniform(0.1, 1.0), 2)))
    p.update(seed=seed, V=20.0, mic_S=1.0, hill=2.0, Emax=0.6, b=0.5, d0=0.1,
             cena_fitness=0.1, K=1e10, N0=1e9, k_imm=1.0, N_imm=1e4,
             spicka_max=40.0,     # mg/l, limit nejvyšší koncentrace
             minimum_max=1.5)     # mg/l, limit koncentrace před poslední dávkou
    r = p["b"] - p["d0"]
    # EC50 tak, aby při C = MIC lék přesně vyrovnal čistý růst malé populace
    p["EC50_S"] = p["mic_S"] / (r / (p["Emax"] - r)) ** (1 / p["hill"])
    p["EC50_R"] = p["EC50_S"] * p["mic_pomer"]
    return p


def _davky(D1, D, tau):
    if tau not in TAU_POVOLENE:
        raise ValueError(f"tau musí být z {TAU_POVOLENE}")
    if not (D1 > 0 and D > 0):
        raise ValueError("dávky musí být kladné")
    casy = np.arange(0, 24 * DNY_LECBY, tau, dtype=float)
    davky = np.full(len(casy), float(D))
    davky[0] = D1
    return casy, davky


def koncentrace(D1, D, tau, t, inst):
    casy, davky = _davky(D1, D, tau)
    t = np.asarray(t, float)[..., None]
    k = np.log(2) / inst["t_half"]
    C = np.where(t >= casy, davky / inst["V"] * np.exp(-k * np.clip(t - casy, 0, None)), 0.0)
    return C.sum(-1)


def toxicita(D1, D, tau, inst):
    casy, _ = _davky(D1, D, tau)
    spicka = koncentrace(D1, D, tau, casy + 1e-9, inst).max()
    minimum = koncentrace(D1, D, tau, casy[-1] - 1e-9, inst)
    return float(spicka), float(minimum)


def celkem_leku(D1, D, tau):
    return float(D1 + (24 * DNY_LECBY // tau - 1) * D)


def _ucinek(D1, D, tau, inst):
    t = (np.arange(int((DNY_LECBY + DOBA_SLEDOVANI) * 24 / DT)) + 0.5) * DT
    C = koncentrace(D1, D, tau, t, inst) ** inst["hill"]
    return (inst["Emax"] * C / (inst["EC50_S"] ** inst["hill"] + C),
            inst["Emax"] * C / (inst["EC50_R"] ** inst["hill"] + C))


def deterministicky(D1, D, tau, inst):
    S = inst["N0"]
    for ks in _ucinek(D1, D, tau, inst)[0]:
        rust = inst["b"] * (1 - S / inst["K"])
        smrt = inst["d0"] + ks + inst["k_imm"] / (1 + S / inst["N_imm"])
        S *= np.exp((rust - smrt) * DT)
        if S < 1.0:
            return True
    return False


def stochasticky(D1, D, tau, inst, n_behu=200, seed=0):
    g = np.random.default_rng(seed)
    kS, kR = _ucinek(D1, D, tau, inst)
    b, d0, K, Ni = inst["b"], inst["d0"], inst["K"], inst["N_imm"]
    bR = b * (1 - inst["cena_fitness"])
    S = np.full(n_behu, int(inst["N0"]), dtype=np.int64)
    R = g.poisson(inst["m0"], n_behu).astype(np.int64)
    konec_leku = int((24 * DNY_LECBY + 12) / DT)
    for i in range(len(kS)):
        N = S + R
        # zkratka: bez léku populace nad 100 * N_imm už jen roste, nulová zůstane nulová
        if i % 8 == 0 and (not N.any() or
                           (i > konec_leku and np.all((N == 0) | (N > 100 * Ni)))):
            break
        tlum = np.clip(1 - N / K, 0, None)
        imm = inst["k_imm"] / (1 + N / Ni)
        # nejdřív úmrtí (binomicky), pak dělení přeživších (Poisson): střední hodnota
        # je přesně S * exp((množení - smrt) * DT) jako v deterministickém modelu
        S = g.binomial(S, np.exp(-(d0 + kS[i] + imm) * DT))
        R = g.binomial(R, np.exp(-(d0 + kR[i] + imm) * DT))
        zS = g.poisson(S * np.expm1(b * tlum * DT))
        zR = g.poisson(R * np.expm1(bR * tlum * DT))
        mutanti = g.binomial(zS, inst["mu"])
        S, R = S + zS - mutanti, R + zR + mutanti
    vysledek = np.zeros(n_behu, dtype=int)
    vysledek[(S > 0) & (R == 0)] = 1
    vysledek[R > 0] = 2
    return vysledek
