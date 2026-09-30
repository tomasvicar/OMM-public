"""Proložení dat přímkou v normě L1 zapsané ručně jako lineární program.

Spuštění:

    uv run python prednasky/L2/kod/lp_triky_cvxpy.py

Kalibrace senzoru: osm vzorků o známé koncentraci, u posledního se do kyvety
dostala bublina a měření je nesmyslně vysoké. Hledáme přímku y = a x + b.

Absolutní hodnota není lineární, ale minimalizace součtu absolutních hodnot
lineárním programem je: každý zbytek r_i = a x_i + b - y_i shora omezíme
pomocnou proměnnou t_i dvěma nerovnostmi

    r_i <= t_i,    -r_i <= t_i        (dohromady t_i >= |r_i|)

a minimalizujeme součet t_i. Protože se t_i objevuje jen v účelové funkci
s kladným koeficientem, solver je dotlačí dolů až na t_i = |r_i|.

Pro srovnání tutéž úlohu řešíme i metodou nejmenších čtverců, která už
lineární program není (účel je kvadratický) a která na odlehlý bod doplatí.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

# Koncentrace [mmol/l] a naměřený signál [mV]; poslední vzorek je odlehlý.
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = np.array([3.2, 5.0, 6.9, 9.1, 10.8, 12.8, 15.0, 25.0])

# --- Proložení v normě L1 zapsané jako LP s pomocnými proměnnými ---------
a = cp.Variable()                       # směrnice
b = cp.Variable()                       # posun
t = cp.Variable(len(x))                 # pomocné proměnné, t_i >= |r_i|

zbytky = cp.multiply(a, x) + b - y
lp = cp.Problem(cp.Minimize(cp.sum(t)), [zbytky <= t, -zbytky <= t])
lp.solve()

def primka(smernice: float, posun: float) -> str:
    """Zapíše přímku ve tvaru y = a x + b se správným znaménkem."""
    return f"y = {smernice:.2f} x {'+' if posun >= 0 else '-'} {abs(posun):.2f}"


a_l1, b_l1 = float(a.value), float(b.value)
print(f"stav řešiče:      {lp.status}")
print(f"L1 (LP):          {primka(a_l1, b_l1)}")
print(f"součet |r_i|:     {lp.value:.2f}")
# Kontrola, že se pomocné proměnné opravdu dotlačily na |r_i|.
odchylka = np.max(np.abs(t.value - np.abs(a_l1 * x + b_l1 - y)))
print(f"max |t_i - |r_i||: {odchylka:.2e}")

# --- Totéž metodou nejmenších čtverců (kvadratický účel, tedy už ne LP) --
a2 = cp.Variable()
b2 = cp.Variable()
mnc = cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(a2, x) + b2 - y)))
mnc.solve()

a_mnc, b_mnc = float(a2.value), float(b2.value)
print(f"nejmenší čtverce: {primka(a_mnc, b_mnc)}")

# --- Kontrola: jak by obě metody dopadly bez odlehlého vzorku ------------
xc, yc = x[:-1], y[:-1]
ac = cp.Variable()
bc = cp.Variable()
tc = cp.Variable(len(xc))
zbytky_c = cp.multiply(ac, xc) + bc - yc
cp.Problem(cp.Minimize(cp.sum(tc)), [zbytky_c <= tc, -zbytky_c <= tc]).solve()
ac_mnc, bc_mnc = np.polyfit(xc, yc, 1)
print(f"\nbez odlehlého vzorku, L1:  {primka(float(ac.value), float(bc.value))}")
print(f"bez odlehlého vzorku, MNČ: {primka(float(ac_mnc), float(bc_mnc))}")

# --- Srovnání na neodlehlé části dat ------------------------------------
print("\n x     y      L1     MNČ")
for xi, yi in zip(x, y):
    print(f"{xi:3.0f} {yi:6.1f} {a_l1 * xi + b_l1:6.2f} {a_mnc * xi + b_mnc:6.2f}")
print("\nOdlehlý bod L1 proložení ignoruje, nejmenší čtverce se za ním táhnou.")
