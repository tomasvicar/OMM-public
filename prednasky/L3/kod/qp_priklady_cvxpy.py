"""Dvě kvadratické úlohy s týmiž čísly jako na slidech: ridge a SVM (i jeho duál).

Spuštění:

    uv run python prednasky/L3/kod/qp_priklady_cvxpy.py

Obě úlohy mají stejný tvar
    min  1/2 x^T Q x + c^T x   za   lineárními omezeními,
liší se jen tím, co je Q a co znamenají řádky omezení. Čísla jsou didaktická
(hezká, ne realistická) a shodují se s obrázky v prednasky/L3/obrazky/.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def ridge() -> None:
    """Regularizace: pokuta alfa*||x||^2 versus rozpočet ||x||^2 <= t."""
    print("=" * 66)
    print("1) RIDGE — pokuta místo omezení")
    print("=" * 66)

    A = np.array([[2.0, 2.0],      # dva silně korelované příznaky
                  [1.0, 0.0],
                  [0.0, 1.0]])
    b = np.array([5.0, 4.0, 3.0])

    # a) bez pokuty: obyčejné nejmenší čtverce
    x = cp.Variable(2)
    mnc = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    mnc.solve()
    x_mnc = x.value.copy()
    print(f"bez pokuty        x* = {np.round(x_mnc, 3)}   "
          f"chyba = {mnc.value:.2f}   ||x||^2 = {x_mnc @ x_mnc:.2f}")

    # b) ridge: chyba + alfa * velikost koeficientů
    alfa = 1.0
    x = cp.Variable(2)
    hreben = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + alfa * cp.sum_squares(x)))
    hreben.solve()
    x_ridge = x.value.copy()
    t = float(x_ridge @ x_ridge)
    print(f"pokuta alfa = {alfa:.0f}   x* = {np.round(x_ridge, 3)}   "
          f"chyba = {np.sum((A @ x_ridge - b) ** 2):.2f}   ||x||^2 = {t:.2f}")

    # c) totéž jako omezení ||x||^2 <= t — musí vyjít stejné x*
    x = cp.Variable(2)
    omezeni = [cp.sum_squares(x) <= t]
    rozpocet = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), omezeni)
    rozpocet.solve()
    print(f"rozpočet t = {t:.2f}   x* = {np.round(x.value, 3)}   "
          f"chyba = {rozpocet.value:.2f}")
    # Multiplikátor omezení ||x||^2 <= t vyjde přesně roven alfa — to je ta ekvivalence.
    multiplikator = float(np.ravel(omezeni[0].dual_value)[0])
    print(f"multiplikátor omezení = {multiplikator:.3f}   (= alfa)")
    print()


def svm() -> None:
    """SVM s tvrdým odstupem: min 1/2 ||w||^2 za y_i (w^T z_i + b) >= 1."""
    print("=" * 66)
    print("2) SVM — největší odstup mezi třídami")
    print("=" * 66)

    nemocni = np.array([[3.0, 3.0], [5.0, 1.0], [4.0, 4.0],
                        [2.5, 4.5], [4.0, 2.5], [6.0, 0.5]])
    zdravi = np.array([[1.0, 1.0], [2.0, 0.0], [0.5, 1.0],
                       [0.0, 1.0], [1.5, 0.0], [0.0, 0.0]])
    Z = np.vstack([nemocni, zdravi])
    y = np.hstack([np.ones(len(nemocni)), -np.ones(len(zdravi))])

    w = cp.Variable(2)
    b = cp.Variable()
    odstupy = cp.multiply(y, Z @ w + b) >= 1
    uloha = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w)), [odstupy])
    uloha.solve()

    w_hodnota = np.asarray(w.value)
    print(f"w  = {np.round(w_hodnota, 3)}     b = {b.value:.3f}")
    print(f"účelová funkce 1/2 ||w||^2 = {uloha.value:.3f}")
    print(f"šířka odstupu 2/||w|| = {2 / np.linalg.norm(w_hodnota):.3f}")
    print("\n  bod            y   y(w^T z + b)   multiplikátor   support vektor?")
    rezervy = y * (Z @ w_hodnota + b.value)
    for bod, znak, rezerva, mu in zip(Z, y, rezervy, np.asarray(odstupy.dual_value)):
        je_sv = "ANO" if mu > 1e-6 else "—"
        print(f"  ({bod[0]:4.1f}, {bod[1]:4.1f})  {znak:+.0f}   {rezerva:8.2f}   "
              f"{mu:13.3f}   {je_sv:>7}")
    print("\nMultiplikátor je nenulový právě u bodů s rovností — komplementární")
    print("volnost, na kterou se dnes ještě podíváme u KKT podmínek.")
    print()
    svm_dual(Z, y, uloha.value)


def svm_dual(Z: np.ndarray, y: np.ndarray, p_hvezda: float) -> None:
    """Duál SVM: max sum(mu) - 1/2 ||sum_i mu_i y_i z_i||^2 za mu >= 0, sum mu_i y_i = 0.

    Odvození na slidu: lagrangián, nejlepší w = sum mu_i y_i z_i, a protože je
    lagrangián v b lineární, musí platit sum mu_i y_i = 0 (jinak je mez -inf).
    Data vstupují jen přes skalární součiny z_i^T z_j (kernel trick).
    """
    print("=" * 66)
    print("3) Duál SVM — tatáž úloha z pohledu cen")
    print("=" * 66)
    mu = cp.Variable(len(y))
    G = y[:, None] * Z                       # řádky y_i z_i
    duál = cp.Problem(cp.Maximize(cp.sum(mu) - 0.5 * cp.sum_squares(G.T @ mu)),
                      [mu >= 0, y @ mu == 0])
    duál.solve()
    w = G.T @ mu.value
    print(f"d* = {duál.value:.3f}   p* = {p_hvezda:.3f}   (silná dualita)")
    print(f"w = sum mu_i y_i z_i = {np.round(w, 3)}")
    print(f"podpůrné body (mu_i > 0): {Z[mu.value > 1e-6].tolist()}")
    # Čtyři body na okraji pásma ve dvou rozměrech: ceny mu nejsou jednoznačné,
    # jednoznačné je jen w, b a hodnota d* = p*.
    assert abs(duál.value - p_hvezda) < 1e-6
    assert np.allclose(w, [0.5, 0.5], atol=1e-4)
    print()


if __name__ == "__main__":
    ridge()
    svm()
