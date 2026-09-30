"""Kontrolní přepočet čísel hlavního příkladu prvního cvičení.

Úloha z tabulové rozcvičky: míchání infuzní směsi ze dvou roztoků glukózy.

    proměnné   x1, x2 ... objem roztoku A a B v litrech
    minimize   20*x1 + 50*x2                  (cena v Kč)
    subject to 50*x1 + 200*x2 >= 150          (glukóza, g)
               x1 + x2        >= 1.5          (objem, l)
               x1, x2         >= 0

Očekávaný výsledek: x* = (1.0, 0.5), cena 45 Kč, obě omezení aktivní,
stínové ceny 0.2 Kč/g glukózy a 10 Kč/l objemu.
"""

import cvxpy as cp
from scipy.optimize import linprog

# --- SciPy: standardní tvar A_ub @ x <= b_ub, proto obrácená znaménka ---
res = linprog(
    c=[20, 50],
    A_ub=[[-50, -200], [-1, -1]],
    b_ub=[-150, -1.5],
    bounds=[(0, None), (0, None)],
)
print("scipy :", res.x, "cena =", res.fun, "status =", res.status)

# --- CVXPY: deklarativní zápis + duální proměnné ---
x = cp.Variable(2, nonneg=True)
glukoza = 50 * x[0] + 200 * x[1] >= 150
objem = x[0] + x[1] >= 1.5
uloha = cp.Problem(cp.Minimize(20 * x[0] + 50 * x[1]), [glukoza, objem])
uloha.solve()

print("cvxpy :", x.value, "cena =", uloha.value)
print("stínové ceny: glukóza =", glukoza.dual_value, "objem =", objem.dual_value)

# --- kontrola dosazením zpět (vzor pro pravidlo „kontrola, která umí selhat") ---
x1, x2 = x.value
assert 50 * x1 + 200 * x2 >= 150 - 1e-6, "porušeno omezení na glukózu"
assert x1 + x2 >= 1.5 - 1e-6, "porušeno omezení na objem"
print("kontrola přípustnosti prošla")
