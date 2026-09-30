"""Pekárna jako lineární program v CVXPY, včetně stínových cen.

Spuštění:

    uv run python prednasky/L2/kod/lp_pekarna_cvxpy.py

Pekárna má denně 80 kg mouky a 600 minut pece. Chléb vynese 14 Kč
(0,5 kg mouky, 3 min pece), bageta 8 Kč (0,2 kg mouky, 2 min pece).
Hledáme denní plán pečení s největším ziskem:

    maximalizuj  c^T x   pri   A x <= b,  x >= 0.

Skript vypíše stav řešiče, optimální plán, zisk, spotřebu surovin a
duální proměnné (stínové ceny) — a ověří je tím, že úlohu vyřeší znovu
s pravou stranou zvětšenou o jednotku.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

# x = (chleby, bagety) [ks/den]
c = np.array([14.0, 8.0])              # zisk za kus [Kč/ks]
A = np.array([[0.5, 0.2],              # mouka [kg/ks]
              [3.0, 2.0]])             # pec   [min/ks]
b = np.array([80.0, 600.0])            # zásoba: mouka [kg/den], pec [min/den]

NAZVY = ("mouka [kg]", "pec [min]")


def vyres(prava_strana: np.ndarray) -> cp.Problem:
    """Vyřeší pekárnu pro zadanou pravou stranu b a vrátí hotovou úlohu."""
    x = cp.Variable(2, nonneg=True)
    uloha = cp.Problem(cp.Maximize(c @ x), [A @ x <= prava_strana])
    uloha.solve()
    return uloha


uloha = vyres(b)
x_hvezda = uloha.variables()[0].value
# Duální proměnné = stínové ceny: o kolik vzroste zisk při uvolnění omezení o 1.
stinove_ceny = uloha.constraints[0].dual_value

print(f"stav řešiče:   {uloha.status}")
print(f"x* =           {np.round(x_hvezda, 2)}  (chleby, bagety)")
print(f"zisk =         {uloha.value:.0f} Kč")
print(f"spotřeba A x*: {np.round(A @ x_hvezda, 1)}  (mez {b})")
print(f"stínové ceny:  {np.round(stinove_ceny, 2)}  (Kč za kg mouky, Kč za min pece)")

print("\nověření stínových cen — přidáme jednotku k jednomu omezení:")
for i, nazev in enumerate(NAZVY):
    b_vice = b.copy()
    b_vice[i] += 1.0
    prirustek = vyres(b_vice).value - uloha.value
    print(f"  {nazev:<11} +1  →  zisk +{prirustek:.2f} Kč"
          f"   (stínová cena {stinove_ceny[i]:.2f})")
