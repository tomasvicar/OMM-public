"""Výživa pacienta jako lineární program (Stiglerova dieta v malém).

Spuštění:

    uv run python prednasky/L2/kod/lp_dieta_cvxpy.py

Denní směs enterální výživy se míchá ze tří přípravků. Musí dodat aspoň
2000 kcal energie, 100 g bílkovin a 20 g vlákniny; chceme ji co nejlevnější.
Čísla jsou didaktická, ne klinická.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

# x = (energetický, proteinový, vlákninový) — počet dávek po 100 ml
c = np.array([10.0, 12.0, 9.0])          # cena za 100 ml [Kč]
A = np.array([[200.0, 100.0, 100.0],     # energie   [kcal / 100 ml]
              [6.0, 10.0, 4.0],          # bílkoviny [g / 100 ml]
              [0.0, 0.0, 4.0]])          # vláknina  [g / 100 ml]
b = np.array([2000.0, 100.0, 20.0])      # minimální denní dávky

zivina = ("energie [kcal]", "bílkoviny [g]", "vláknina [g]")

x = cp.Variable(3, nonneg=True)
omezeni = [A @ x >= b]
uloha = cp.Problem(cp.Minimize(c @ x), omezeni)
uloha.solve()

print(f"stav řešiče:  {uloha.status}")
print(f"rozměry:      A je {A.shape[0]}x{A.shape[1]}, b má {b.size}, c má {c.size}")
print(f"x* =          {np.round(x.value, 2)}  (dávky po 100 ml)")
print(f"cena =        {uloha.value:.0f} Kč / den")
print("kontrola omezení A x* >= b:")
for nazev, dodano, pozadovano in zip(zivina, A @ x.value, b):
    rezerva = dodano - pozadovano
    stav = "aktivní" if abs(rezerva) < 1e-6 else f"rezerva {rezerva:.1f}"
    print(f"  {nazev:<16} dodáno {dodano:8.1f}   požadováno {pozadovano:7.1f}   {stav}")
# Duální proměnné = stínové ceny: o kolik zdraží směs, když požadavek vzroste o 1.
print(f"stínové ceny: {np.round(omezeni[0].dual_value, 3)}")
