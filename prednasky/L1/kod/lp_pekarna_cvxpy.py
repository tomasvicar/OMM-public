"""Školní lineární program z přednášky vyřešený v CVXPY.

Spuštění:

    uv run python prednasky/L1/kod/lp_pekarna_cvxpy.py

Pekárna má denně 80 kg mouky a 600 minut pece. Chléb vynese 14 Kč
(0,5 kg mouky, 3 min pece), bageta 8 Kč (0,2 kg mouky, 2 min pece).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

# x = (chleby, bagety)
c = np.array([14.0, 8.0])              # zisk za kus [Kč]
A = np.array([[0.5, 0.2],              # mouka [kg/ks]
              [3.0, 2.0]])             # pec   [min/ks]
b = np.array([80.0, 600.0])            # denní zásoba mouky a času pece

x = cp.Variable(2, nonneg=True)
uloha = cp.Problem(cp.Maximize(c @ x), [A @ x <= b])
uloha.solve()

print(f"stav řešiče:   {uloha.status}")
print(f"x* =           {np.round(x.value, 2)}  (chleby, bagety)")
print(f"zisk =         {uloha.value:.0f} Kč")
print(f"spotřeba A x*: {np.round(A @ x.value, 1)}  (mez {b})")
# Duální proměnné = stínové ceny: o kolik vzroste zisk při uvolnění omezení o 1.
print(f"stínové ceny:  {np.round(uloha.constraints[0].dual_value, 2)}")
