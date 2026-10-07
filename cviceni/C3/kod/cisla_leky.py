"""Zavazna cisla hlavniho prikladu tretiho cviceni (dva leky, KKT).

    uv run python cviceni/C3/kod/cisla_leky.py

Uloha:  minimize (x1 - 4)^2 + (x2 - 2)^2
        subject to x1 + x2 <= B          (spolecna zatez ledvin)
                   [x2 >= 1.5]           (jen samostatna uloha)
Resic `vyres` importuji i tabule_leky.py a uloha_samostatne.py; vsechny ostatni
skripty a notebooky cviceni se s temito cisly porovnavaji.
"""

import cvxpy as cp
import numpy as np

CIL = np.array([4.0, 2.0])     # idealni davky leku A a B [mg]
LIMIT = 4.0                    # x1 + x2 <= 4 [mg]
MIN_B = 1.5                    # samostatna uloha: x2 >= 1,5 [mg]


def vyres(limit=LIMIT, min_b=None):
    """Vrati x*, f*, multiplikatory (v poradi ledviny, min. B) a is_dcp()."""
    x = cp.Variable(2)
    omezeni = [cp.sum(x) <= limit]
    if min_b is not None:
        omezeni.append(x[1] >= min_b)
    uloha = cp.Problem(cp.Minimize(cp.sum_squares(x - CIL)), omezeni)
    uloha.solve(solver=cp.CLARABEL)
    mu = [float(o.dual_value) for o in omezeni]
    return x.value, uloha.value, mu, uloha.is_dcp()


if __name__ == "__main__":
    for b in (4.0, 5.0, 7.0):
        x, f, mu, _ = vyres(b)
        print(f"limit {b}: x = {np.round(x, 6)}, f = {f:.6f}, mu = {np.round(mu, 6)}")
    x, f, mu, _ = vyres(LIMIT, MIN_B)
    print(f"samostatna (limit 4, x2 >= 1,5): x = {np.round(x, 6)}, f = {f:.6f}, mu = {np.round(mu, 6)}")
