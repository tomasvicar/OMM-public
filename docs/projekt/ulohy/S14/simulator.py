"""Simulátor gradientové HPLC (simulační úloha S14: rychle, nebo čistě?).

Směs látek se dělí na koloně s obrácenými fázemi; mobilní fáze je voda
a methanol, jehož podíl phi během analýzy lineárně roste (gradient).

Rozhraní
--------
    from simulator import simuluj, MEZE, NAZVY

    x = [T, F, tG, phi_start, phi_konec]
    v = simuluj(x, instance=0)

x – metoda, 5 čísel v mezích MEZE:
    T          teplota kolony [°C]                          25 až 60
    F          průtok mobilní fáze [mL/min]                 0,2 až 3,0
    tG         délka gradientu [min]                        1 až 40
    phi_start  podíl methanolu na začátku gradientu         0,05 až 0,95
    phi_konec  podíl methanolu na konci gradientu           0,05 až 0,95
Podíl methanolu jde za tG minut lineárně z phi_start na phi_konec a pak
zůstane na phi_konec. phi_start = phi_konec je analýza při stálém složení
(izokratická); phi_konec < phi_start simulátor spočítá také, v praxi se
klesající gradient nepoužívá.
instance – směs (0 = hlavní, 12 látek; 1 až 5 = jiné směsi, 10–15 látek)
sum_tR – směrodatná odchylka náhodné chyby retenčních časů [min] (0 = bez
    šumu); seed_sumu – seed této chyby.

simuluj vrátí slovník:
    'tR'     retenční časy látek [min] (pořadí látek pevné, ne podle eluce)
    'sigma'  šířky píků (směrodatné odchylky Gaussovy křivky) [min]
    'poradi' indexy látek v pořadí, v jakém vycházejí z kolony
    'Rs'     rozlišení sousedních píků v pořadí eluce (délka n - 1)
    'Rs_min' nejmenší z nich (nejhorší pár)
    'doba'   délka analýzy = konec posledního píku v pořadí eluce,
             tR + 2 sigma tohoto píku [min]
    'tlak'   nejvyšší tlak na koloně během analýzy [bar]

Model:
  retenční faktor  ln k = ln k_w(T) - S phi
                   ln k_w(T) = ln k_w(25 °C) + B (1/T - 1/T_25)   (van 't Hoff)
  látka vyjde, až urazí celou kolonu: int_0^(tR - t0) dt / (t0 k(phi(t))) = 1,
      kde phi(t) je složení na vstupu kolony (program posunutý o dwell čas tD);
      na lineárním úseku jde integrál spočítat vzorcem
  t0 = VM / F (mrtvý čas), tD = VD / F,
  šířka píku sigma = t0 (1 + k_e) / sqrt(N), k_e = k ve chvíli, kdy látka vychází,
  počet pater N z van Deemterovy rovnice (difuze roste s T a klesá s viskozitou),
  Rs = (t2 - t1) / (2 (sigma1 + sigma2)),  tlak = K F max_t viskozita(phi(t), T).
Jedno vyhodnocení trvá asi 0,3 ms.
"""
import numpy as np

NAZVY = ["T", "F", "tG", "phi_start", "phi_konec"]
MEZE = np.array([[25.0, 60.0], [0.2, 3.0], [1.0, 40.0], [0.05, 0.95], [0.05, 0.95]])

VM = 1.0        # objem mobilní fáze v koloně [mL] (kolona 100 x 4,6 mm)
VD = 0.8        # dwell objem (od směšovače po kolonu) [mL]
L_CM = 10.0     # délka kolony [cm]
DP_CM = 3e-4    # průměr částic [cm] (3 um)
K_TLAK = 170.0  # tlak [bar] na 1 mL/min a 1 mPa s
T25 = 298.15


def viskozita(phi, T):
    """Viskozita směsi voda-methanol [mPa s]; maximum asi 1,6 při phi 0,4 a 25 °C."""
    TK = T + 273.15
    lw = np.log(0.89) + 1811.0 * (1 / TK - 1 / T25)
    lm = np.log(0.55) + 975.0 * (1 / TK - 1 / T25)
    c = 3.29 + 1950.0 * (1 / TK - 1 / T25)
    return np.exp(phi * lm + (1 - phi) * lw + phi * (1 - phi) * c)


def smes(instance=0):
    """Parametry látek: ln k_w (25 °C), S a B [K]."""
    rng = np.random.default_rng(5500 + instance)
    n = 12 if instance == 0 else int(rng.integers(10, 16))
    logkw = rng.uniform(1.2, 3.6, n)
    S10 = 0.9 + 0.95 * logkw + rng.normal(0, 0.15, n)       # log10 k na jednotku phi
    B = 700.0 + 450.0 * logkw + rng.normal(0, 150.0, n)     # van 't Hoff [K], ln k
    for j, (phi_x, dS, dB) in enumerate([(0.62, 0.9, 420.0), (0.74, -0.8, -380.0)]):
        a, b = 2 * j, 2 * j + 1
        S10[b] = S10[a] + dS
        logkw[b] = logkw[a] + dS * phi_x
        B[b] = B[a] + dB
    return {"lnkw": logkw * np.log(10), "S": S10 * np.log(10), "B": B}


_SMESI = {}


def program(t, x):
    """Podíl methanolu na výstupu směšovače v čase t [min]."""
    return np.interp(t, [0.0, x[2]], [x[3], x[4]])


def simuluj(x, instance=0, sum_tR=0.0, seed_sumu=0):
    x = np.asarray(x, float)
    if x.shape != (5,):
        raise ValueError("x musí mít 5 složek: " + ", ".join(NAZVY))
    if np.any(x < MEZE[:, 0] - 1e-12) or np.any(x > MEZE[:, 1] + 1e-12):
        raise ValueError("x je mimo meze MEZE")
    if instance not in _SMESI:
        _SMESI[instance] = smes(instance)
    p = _SMESI[instance]
    T, F, tG, p0, p1 = x
    TK = T + 273.15
    lnkw = p["lnkw"] + p["B"] * (1 / TK - 1 / T25)
    S = p["S"]
    t0, tD = VM / F, VD / F
    c = np.exp(-lnkw) / t0          # rychlost postupu kolonou je c * exp(S phi)

    # 1) dwell: do času tD teče na kolonu phi_start
    zbyva = 1.0 - c * np.exp(S * p0) * tD          # část kolony, kterou látka ještě neurazila
    hotovo = zbyva <= 0
    te = np.where(hotovo, 1.0 / (c * np.exp(S * p0)), np.nan)
    phi_e = np.full(len(S), p0)
    # 2) gradient: phi = p0 + b u pro u v [0, tG]
    b = (p1 - p0) / tG
    if abs(b) < 1e-12:                               # izokraticky
        te = np.where(hotovo, te, tD + zbyva / (c * np.exp(S * p0)))
    else:
        cely = c * (np.exp(S * p1) - np.exp(S * p0)) / (S * b)   # postup za celý gradient
        behem = ~hotovo & (cely >= zbyva)
        phi_in = np.log(np.maximum(np.exp(S * p0) + np.maximum(zbyva, 0) * S * b / c,
                                   1e-300)) / S
        te = np.where(behem, tD + (phi_in - p0) / b, te)
        phi_e = np.where(behem, phi_in, phi_e)
        # 3) po gradientu drží phi_konec
        po = ~hotovo & ~behem
        te = np.where(po, tD + tG + (zbyva - cely) / (c * np.exp(S * p1)), te)
        phi_e = np.where(po, p1, phi_e)
    tR = te + t0
    k_e = np.exp(lnkw - S * phi_e)

    # počet pater: van Deemter v redukovaných jednotkách, difuze úměrná T / viskozita
    Dm = 1.0e-5 * (TK / T25) * (0.89 / viskozita(phi_e, T))     # cm^2/s
    u = L_CM / (t0 * 60.0)                                       # lineární rychlost [cm/s]
    nu = u * DP_CM / Dm
    N = L_CM / ((1.0 + 2.0 / nu + 0.08 * nu) * DP_CM)
    sigma = t0 * (1 + k_e) / np.sqrt(N)

    if sum_tR > 0:
        tR = tR + np.random.default_rng(seed_sumu).normal(0, sum_tR, len(tR))
    poradi = np.argsort(tR)
    ts, ss = tR[poradi], sigma[poradi]
    Rs = (ts[1:] - ts[:-1]) / (2 * (ss[1:] + ss[:-1]))
    doba = float(ts[-1] + 2 * ss[-1])
    phi_vstup = program(np.linspace(0.0, doba, 201) - tD, x)     # složení během analýzy
    tlak = float(K_TLAK * F * np.max(viskozita(phi_vstup, T)))
    return {"tR": tR, "sigma": sigma, "poradi": poradi, "Rs": Rs,
            "Rs_min": float(Rs.min()), "doba": doba, "tlak": tlak}
