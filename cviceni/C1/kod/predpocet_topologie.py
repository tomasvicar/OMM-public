"""Predpocet dat pro interaktivni hriste u ukazky 'topologicka optimalizace'.

Vyrobi soubor cviceni/C1/data/topologie.json, ktery se vklada primo do reportu
reporty/topologie.html jako <script type="application/json">. V prohlizeci se
nic nepocita: jedna optimalizace je 7 200 promennych a 50 reseni soustavy
o 14 762 rovnicich, tedy nekolik sekund i v Pythonu. Widget jen prepina mezi
hotovymi vysledky pro mrizku objemovych podilu.

Pro kazdy podil se uklada:
    poddajnost optimalizovaneho tvaru, poddajnost homogenni desky teze
    hmotnosti a samotny tvar podvzorkovany na sit 60x30 (cela cisla 0-99),
    a k tomu poddajnost peti RUCNE navrzenych konstrukci (prima vzpera, dva
    pasy, krizove ztuzeni, prihradovina) pri temze rozpoctu materialu.
    Rucni navrhy se neukladaji jako pole hustot - v JSON jsou jen usecky
    a tloustka, kterou si widget nakresli tyz tvar podle teze formule.

Spusteni z korene repozitare (bezi ~2 minuty):
    uv run python cviceni/C1/kod/predpocet_topologie.py

Model je tentyz jako v kod/ukazka_topologie.py (SIMP + filtr citlivosti +
optimality criteria), aby cisla v reportu, v notebooku i ve widgetu sedela:
pri podilu 0,35 musi vyjit poddajnost 933,2 na startu a 91,1 v optimu.
"""

import json
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# --- parametry, shodne s ukazka_topologie.py ---------------------------------
NELX, NELY = 120, 60
PENAL = 3.0
RMIN = 2.0
ITERACI = 50
KROK = 0.2
E0, EMIN, NU = 1.0, 1e-9, 0.3

# Mrizka objemovych podilu. Je zvolena tak, aby ke kazdemu podilu od 0,30 vys
# byla v mrizce i jeho POLOVINA (widget ukazuje, co stoji ubrat pulku materialu).
# Dolni mez 0,15 neni nahodna: pod ni jsou pruty tencí nez polomer filtru RMIN,
# SIMP kvuli tomu konci v citelne horsim lokalnim optimu (pri 0,10 dokonce horsim
# nez rucni "dve vzpery") a srovnani navrhu by uz nemerilo kvalitu navrhu, ale
# artefakt filtru.
PODILY = [0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
ZHUSTENI = 2                  # podvzorkovani tvaru: 120x60 -> 60x30
VYSTUP = "cviceni/C1/data/topologie.json"


def matice_tuhosti_elementu(nu):
    """Tuhost jednoho ctvercoveho elementu (rovinna napjatost, 4 uzly)."""
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
                  -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    return 1 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


KE = matice_tuhosti_elementu(NU)
POCET = NELX * NELY

edof = np.zeros((POCET, 8), dtype=int)
for ex in range(NELX):
    for ey in range(NELY):
        el = ey + ex * NELY
        n1 = (NELY + 1) * ex + ey
        n2 = (NELY + 1) * (ex + 1) + ey
        edof[el] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                    2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3]
iK = np.kron(edof, np.ones((8, 1))).flatten().astype(int)
jK = np.kron(edof, np.ones((1, 8))).flatten().astype(int)

# filtr citlivosti (bez nej vznika sachovnice)
r = int(np.ceil(RMIN))
iH, jH, sH = [], [], []
for i in range(NELX):
    for j in range(NELY):
        e1 = i * NELY + j
        for k in range(max(i - r + 1, 0), min(i + r, NELX)):
            for l in range(max(j - r + 1, 0), min(j + r, NELY)):
                vaha = RMIN - np.hypot(i - k, j - l)
                if vaha > 0:
                    iH.append(e1)
                    jH.append(k * NELY + l)
                    sH.append(vaha)
H = coo_matrix((sH, (iH, jH)), shape=(POCET, POCET)).tocsr()
Hs = np.array(H.sum(1)).flatten()

# okrajove podminky: vlevo vetknuti, vpravo uprostred sila dolu
ndof = 2 * (NELX + 1) * (NELY + 1)
uzel_zatizeni = (NELX + 1) * (NELY + 1) - (NELY + 1) + NELY // 2
f = np.zeros(ndof)
f[2 * uzel_zatizeni + 1] = -1.0
pevne = np.arange(0, 2 * (NELY + 1))
volne = np.setdiff1d(np.arange(ndof), pevne)


def poddajnost(x):
    """Vyresi rovnovahu K(x) u = f a vrati poddajnost a citlivost po elementech."""
    E = EMIN + x**PENAL * (E0 - EMIN)
    sK = ((KE.flatten()[np.newaxis]).T * E).flatten(order="F")
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    u = np.zeros(ndof)
    u[volne] = spsolve(K[volne, :][:, volne], f[volne])
    ue = u[edof]
    ce = (ue @ KE * ue).sum(1)
    return (E * ce).sum(), ce


def krok_optimality_criteria(x, dc, volfrac):
    """Aktualizace hustot pri presne dodrzenem rozpoctu materialu."""
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.clip(x * np.sqrt(-dc / lmid), np.maximum(0.0, x - KROK),
                       np.minimum(1.0, x + KROK))
        if xnew.sum() > volfrac * POCET:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def optimalizuj(volfrac):
    """Cely beh SIMP pro jeden objemovy podil; vrati hustoty a poddajnost."""
    x = np.full(POCET, volfrac)
    for _ in range(ITERACI):
        _, ce = poddajnost(x)
        dc = -PENAL * x**(PENAL - 1) * (E0 - EMIN) * ce
        dc = np.asarray(H @ (x * dc)) / Hs / np.maximum(1e-3, x)
        x = krok_optimality_criteria(x, dc, volfrac)
    c, _ = poddajnost(x)
    return x, c


def podvzorkuj(x):
    """Sit 120x60 -> 60x30 prumerem bloku 2x2, hodnoty jako cela cisla 0-99."""
    pole = x.reshape(NELX, NELY).T                 # radky = vyska, sloupce = sirka
    ny, nx = NELY // ZHUSTENI, NELX // ZHUSTENI
    mensi = pole.reshape(ny, ZHUSTENI, nx, ZHUSTENI).mean(axis=(1, 3))
    return np.clip(np.rint(mensi * 99), 0, 99).astype(int).flatten().tolist()


# --- rucne navrzene konstrukce k porovnani ------------------------------------
# Kazdy navrh je sada usecek v souradnicich site (x = 0..NELX zleva doprava,
# y = 0..NELY shora dolu; vetknuti je hrana x = 0, sila pusobi v bode
# (NELX, NELY/2)). Hustota bunky = pokryti pasem o tloustce t kolem usecky:
#
#     x = clip(t/2 - vzdalenost(stred bunky, usecka) + 0.5, 0, 1)
#
# Tutez formuli pocita i widget v reportu, takze nakresleny tvar je presne ten,
# kterym se pocitala poddajnost. Tloustka t se pro kazdy objemovy podil hleda
# pulenim intervalu tak, aby prumerna hustota sedla na rozpocet - srovnani je
# pak fer: vsechny konstrukce maji tutez hmotnost.
NAVRHY = [
    {"klic": "pasy", "nazev": "Dva pásy (I-nosník)",
     "popis": "nahoru a dolů, mezi pásy nic",
     "usecky": [[0, 0, 120, 0], [0, 60, 120, 60], [120, 0, 120, 60]]},
    {"klic": "vzpera", "nazev": "Přímá vzpěra",
     "popis": "od vetknutí rovnou k síle",
     "usecky": [[0, 30, 120, 30]]},
    {"klic": "prihradovina", "nazev": "Příhradovina",
     "popis": "pásy a čtyři trojúhelníky",
     "usecky": [[0, 0, 120, 0], [0, 60, 120, 60], [120, 0, 120, 60],
                [0, 0, 30, 60], [30, 60, 60, 0], [60, 0, 90, 60], [90, 60, 120, 0]]},
    {"klic": "kriz", "nazev": "Křížem ztužený rám",
     "popis": "pásy, stojka a obě úhlopříčky",
     "usecky": [[0, 0, 120, 0], [0, 60, 120, 60], [120, 0, 120, 60],
                [0, 0, 120, 60], [0, 60, 120, 0]]},
    {"klic": "vidlice", "nazev": "Dvě vzpěry k síle",
     "popis": "od obou rohů přímo do působiště",
     "usecky": [[0, 0, 120, 30], [0, 60, 120, 30]]},
]
MIN_TLOUSTKA = 2.0            # tencí prut než dvě buňky už síť neunese

# stredy elementu ve stejnem poradi, v jakem jsou hustoty (e = ey + ex*NELY)
_ex, _ey = np.divmod(np.arange(POCET), NELY)
STRED_X, STRED_Y = _ex + 0.5, _ey + 0.5


def vzdalenosti(usecky):
    """Vzdalenost stredu kazde bunky k nejblizsi z usecek navrhu."""
    d = np.full(POCET, np.inf)
    for x1, y1, x2, y2 in usecky:
        vx, vy = x2 - x1, y2 - y1
        delka2 = vx * vx + vy * vy
        t = np.clip(((STRED_X - x1) * vx + (STRED_Y - y1) * vy) / delka2, 0.0, 1.0)
        d = np.minimum(d, np.hypot(STRED_X - (x1 + t * vx), STRED_Y - (y1 + t * vy)))
    return d


def hustoty_navrhu(d, tloustka):
    """Pokryti pasem o dane tloustce - tataz formule je i ve widgetu."""
    return np.clip(tloustka / 2 - d + 0.5, 0.0, 1.0)


def tloustka_na_rozpocet(d, volfrac):
    """Puleni intervalu: jak silne pruty, aby prumerna hustota sedla na rozpocet."""
    lo, hi = 0.0, 2.0 * (NELX + NELY)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if hustoty_navrhu(d, mid).mean() < volfrac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


VZDALENOSTI = {n["klic"]: vzdalenosti(n["usecky"]) for n in NAVRHY}

c_plna, _ = poddajnost(np.ones(POCET))
data = {"nelx": NELX, "nely": NELY, "nx": NELX // ZHUSTENI, "ny": NELY // ZHUSTENI,
        "penal": PENAL, "rmin": RMIN, "iteraci": ITERACI,
        "c_plna": round(float(c_plna), 2),
        "podily": [], "c_opt": [], "c_homogenni": [], "tvary": [],
        "navrhy": [{"klic": n["klic"], "nazev": n["nazev"], "popis": n["popis"],
                    "usecky": n["usecky"], "c": [], "tloustka": [], "objem": []}
                   for n in NAVRHY]}

print(f"sit {NELX}x{NELY} = {POCET} elementu, {len(PODILY)} objemovych podilu")
print(f"plna deska (100 % materialu): poddajnost {c_plna:.2f}\n")
t0 = time.time()
for volfrac in PODILY:
    c_homog, _ = poddajnost(np.full(POCET, volfrac))   # homogenni deska teze hmotnosti
    x, c_opt = optimalizuj(volfrac)
    assert abs(x.mean() - volfrac) < 1e-3, "rozpocet materialu nesedi"
    data["podily"].append(round(volfrac, 3))
    data["c_opt"].append(round(float(c_opt), 2))
    data["c_homogenni"].append(round(float(c_homog), 2))
    data["tvary"].append(podvzorkuj(x))
    print(f"podil {volfrac:.2f}   homogenni {c_homog:9.2f}   optimum {c_opt:8.2f}"
          f"   -> {c_homog / c_opt:5.1f}x tuzsi   ({time.time() - t0:5.1f} s)")

    for zaznam, navrh in zip(data["navrhy"], NAVRHY):     # rucni navrhy nazivo
        d = VZDALENOSTI[navrh["klic"]]
        tl = tloustka_na_rozpocet(d, volfrac)
        xn = hustoty_navrhu(d, tl)
        assert abs(xn.mean() - volfrac) < 1e-4, "navrh nesedi na rozpocet"
        c_navrh, _ = poddajnost(xn)
        pouzitelny = tl >= MIN_TLOUSTKA
        zaznam["tloustka"].append(round(float(tl), 3))
        zaznam["objem"].append(round(float(xn.mean()), 4))
        zaznam["c"].append(round(float(c_navrh), 2) if pouzitelny else None)
        print(f"    {navrh['klic']:14s} tloustka {tl:5.2f}  objem {xn.mean():.4f}"
              f"  poddajnost {c_navrh:9.2f}" + ("" if pouzitelny else "   [prilis tenke, skryto]"))

os.makedirs(os.path.dirname(VYSTUP), exist_ok=True)
with open(VYSTUP, "w", encoding="utf-8") as soubor:
    json.dump(data, soubor, ensure_ascii=False, separators=(",", ":"))
print(f"\nulozeno do {VYSTUP} ({os.path.getsize(VYSTUP) / 1024:.0f} kB)")
