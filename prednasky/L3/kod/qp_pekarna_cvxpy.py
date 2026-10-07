"""Pekárna jako kvadratický program v CVXPY, včetně ověření KKT podmínek.

Spuštění:

    uv run python prednasky/L3/kod/qp_pekarna_cvxpy.py

Proti lineární úloze z druhé přednášky se změnila jen účelová funkce: čím víc
napečeme, tím víc musíme slevit, aby se to prodalo. Zisk 14 a 8 Kč z kusu proto
není pevný, ale klesá s množstvím,

    p1(x1) = 14 - 0,025 x1   [Kč/ks],      p2(x2) = 8 - 0,024 x2   [Kč/ks],

takže zisk je kvadratický. Maximalizujeme

    14 x1 + 8 x2 - 0,025 x1^2 - 0,024 x2^2   pri   A x <= b,  x >= 0,

což je ve standardním tvaru QP minimalizace 1/2 x^T Q x + c^T x.

Skript vypíše stav řešiče, optimální plán, zisk, spotřebu surovin a duální
proměnné, numericky ověří komplementaritu i stacionaritu a nakonec ověří
multiplikátor mouky tím, že úlohu vyřeší znovu s b1 = 81 kg.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

# x = (chleby, bagety) [ks/den]
Q = np.array([[0.05, 0.000],           # zakřivení zisku: 2·0,025 a 2·0,024
              [0.00, 0.048]])
c = np.array([-14.0, -8.0])            # lineární člen (minus výchozí marže) [Kč/ks]
A = np.array([[0.5, 0.2],              # mouka [kg/ks]
              [3.0, 2.0]])             # pec   [min/ks]
b = np.array([80.0, 600.0])            # zásoba: mouka [kg/den], pec [min/den]

NAZVY = ("mouka [kg]", "pec [min]")


def vyres(prava_strana: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, str]:
    """Vyřeší pekárnu pro zadanou pravou stranu a vrátí i duální hodnoty."""
    x = cp.Variable(2)
    zdroje = A @ x <= prava_strana                      # g1, g2 <= 0
    nezapornost = x >= 0                                # g3, g4 <= 0
    uloha = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x),
                       [zdroje, nezapornost])
    uloha.solve()
    return (np.asarray(x.value), float(uloha.value),
            np.asarray(zdroje.dual_value), np.asarray(nezapornost.dual_value),
            uloha.status)


x_hvezda, hodnota, mu, mu_x, stav = vyres(b)
zisk = -hodnota
spotreba = A @ x_hvezda
g = spotreba - b                       # hodnoty omezení v tvaru g(x) <= 0

print(f"stav řešiče:   {stav}")
print(f"x* =           {np.round(x_hvezda, 2)}  (chleby, bagety) [ks/den]")
print(f"zisk =         {zisk:.0f} Kč")
print(f"marže v optimu: chléb {14 - 0.025 * x_hvezda[0]:.2f} Kč, "
      f"bageta {8 - 0.024 * x_hvezda[1]:.2f} Kč")
print(f"spotřeba A x*: {np.round(spotreba, 1)}  (mez {b})")
print(f"multiplikátory mu: {np.round(mu, 2)}  (Kč za kg mouky, Kč za min pece)")

print("\nkomplementarita  mu_i · g_i(x*) = 0:")
for i, nazev in enumerate(NAZVY):
    print(f"  {nazev:<11} g = {g[i]:>7.2f}   mu = {mu[i]:>6.2f}"
          f"   součin = {mu[i] * g[i]:.2e}   → {'napnuté' if abs(g[i]) < 1e-6 else 'rezerva'}")

# Stacionarita: ∇f + Σ mu_i ∇g_i = 0, kde ∇f = Q x + c, ∇g pro A x <= b je řádek A
# a pro -x <= 0 je -I. Duální přípustnost žádá mu >= 0 u všech nerovností.
gradient_f = Q @ x_hvezda + c
zbytek = gradient_f + A.T @ mu - mu_x
print("\nstacionarita  ∇f + Aᵀmu - mu_x = 0:")
print(f"  ∇f =            {np.round(gradient_f, 3)}")
print(f"  Aᵀ mu =         {np.round(A.T @ mu, 3)}")
print(f"  zbytek =        {np.round(zbytek, 9)}   (norma {np.linalg.norm(zbytek):.2e})")
print(f"duální přípustnost: min mu = {min(mu.min(), mu_x.min()):.2e}  (musí být ≥ 0)")

# Ověření multiplikátoru mouky: uvolníme omezení o jeden kilogram.
b_vice = b.copy()
b_vice[0] += 1.0
zisk_vice = -vyres(b_vice)[1]
print("\nověření multiplikátoru mouky — b1 = 81 kg místo 80 kg:")
print(f"  zisk {zisk:.2f} Kč  →  {zisk_vice:.2f} Kč,  přírůstek {zisk_vice - zisk:.2f} Kč")
print(f"  předpověď z multiplikátoru: {mu[0]:.2f} Kč")
print("  (přírůstek je o kousek menší — účelová funkce je zakřivená, "
      "multiplikátor je jen směrnice v bodě)")
