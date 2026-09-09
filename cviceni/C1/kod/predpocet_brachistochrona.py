"""Predpocet dat pro interaktivni hriste u ukazky 'brachistochrona'.

Vyrobi soubor cviceni/C1/data/brachistochrona.json, ktery se vklada primo
do reportu reporty/brachistochrona.html jako <script type="application/json">.
Widget v prohlizeci si tvar skluzavky kresli sam (uzivatel tahne uzly mysi)
a cas sjezdu si taky pocita sam - v prohlizeci se ale NEPOCITA optimum ani
analyticka cykloida, protoze jedno je 49rozmerna optimalizace a druhe
resi rovnici pro koncovy uhel. Oboji je tady predpocitane.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/predpocet_brachistochrona.py

Model je tentyz jako v kod/ukazka_brachistochrona.py (energie, po useckach),
aby cisla v reportu, v notebooku i ve widgetu sedela.
"""

import json
import os

import numpy as np
from scipy.optimize import brentq, minimize

G = 9.81                                          # tihove zrychleni [m/s2]
N = 51                                            # pocet uzlu vcetne krajnich
XB, YB = 2.0, -0.6                                # cilovy bod [m]; start je (0, 0)
Y_DOLNI_MEZ = -2.0
Y_HORNI_MEZ = -1e-4

# kotvy hriste: 6 klicovych bodu, 4 vnitrni se tahnou v obou osach
POCET_KOTEV = 6
Y_TAH_DOLE, Y_TAH_NAHORE = -1.2, -0.02            # meze tahani vnitrnich kotev
X_MEZERA = 0.08                                   # nejmensi povoleny rozestup kotev v x

x = np.linspace(0.0, XB, N)


def doba_sjezdu(x, y):
    """Doba sjezdu po lomene care (x, y) ze startu v klidu - viz ukazka_*.py."""
    v = np.sqrt(2.0 * G * np.maximum(y[0] - y, 0.0) + 1e-16)
    delka = np.hypot(np.diff(x), np.diff(y))
    return float(np.sum(2.0 * delka / (v[:-1] + v[1:])))


def ucelova(y_vnitrni):
    return doba_sjezdu(x, np.concatenate(([0.0], y_vnitrni, [YB])))


# ------------------------------------------------------------- numericke optimum
y_primka = np.linspace(0.0, YB, N)
cas_primka = doba_sjezdu(x, y_primka)

vysledek = minimize(ucelova, np.clip(y_primka[1:-1], Y_DOLNI_MEZ, Y_HORNI_MEZ),
                    method="L-BFGS-B", bounds=[(Y_DOLNI_MEZ, Y_HORNI_MEZ)] * (N - 2),
                    options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-12})
y_opt = np.concatenate(([0.0], vysledek.x, [YB]))
cas_opt = doba_sjezdu(x, y_opt)

# ---------------------------------------------------------- analyticka cykloida
theta_1 = brentq(lambda th: (th - np.sin(th)) / (1.0 - np.cos(th)) - XB / abs(YB),
                 1e-3, 2.0 * np.pi - 1e-3)
R = abs(YB) / (1.0 - np.cos(theta_1))
cas_cykloida = float(np.sqrt(R / G) * theta_1)

th = np.linspace(0.0, theta_1, 121)               # na vykresleni staci 121 bodu
x_cyk = R * (th - np.sin(th))
y_cyk = -R * (1.0 - np.cos(th))

odchylka = 100.0 * (cas_opt - cas_cykloida) / cas_cykloida
assert abs(odchylka) < 1.0, f"optimum se lisi od cykloidy o {odchylka:.2f} %"

# --------------------------------------------------------------------- zapis
data = {
    "poznamka": "vyrobeno skriptem cviceni/C1/kod/predpocet_brachistochrona.py",
    "g": G,
    "x_cil": XB,
    "y_cil": YB,
    "n_uzlu": N,
    "pocet_kotev": POCET_KOTEV,
    "y_tah_dole": Y_TAH_DOLE,
    "y_tah_nahore": Y_TAH_NAHORE,
    "x_mezera": X_MEZERA,
    "x": [round(float(v), 5) for v in x],
    "y_opt": [round(float(v), 5) for v in y_opt],
    "cyk_x": [round(float(v), 5) for v in x_cyk],
    "cyk_y": [round(float(v), 5) for v in y_cyk],
    "cas_opt": round(cas_opt, 5),
    "cas_cykloida": round(cas_cykloida, 5),
    "cas_primka": round(cas_primka, 5),
    "odchylka_opt_cyk": round(odchylka, 4),
    "R": round(float(R), 5),
    "theta_1": round(float(theta_1), 5),
    "iteraci": int(vysledek.nit),
}

cesta = os.path.join("cviceni", "C1", "data", "brachistochrona.json")
os.makedirs(os.path.dirname(cesta), exist_ok=True)
with open(cesta, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

print(f"primka             : {cas_primka:.4f} s")
print(f"numericke optimum  : {cas_opt:.4f} s ({vysledek.nit} iteraci)")
print(f"analyticka cykloida: {cas_cykloida:.4f} s  (odchylka {odchylka:+.3f} %)")
print(f"ulozeno do {cesta} ({os.path.getsize(cesta) / 1024:.1f} kB)")
