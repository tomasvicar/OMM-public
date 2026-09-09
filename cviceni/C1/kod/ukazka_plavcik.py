"""Ukazka pro prvni cviceni: plavcik a tonouci, uloha s JEDINOU promennou.

Nejjednodussi ukazka celeho bloku - cela uloha se vejde na jeden radek a cela
ucelova funkce se da nakreslit jako jedina krivka. Presto z ni vypadne
fyzikalni zakon, ktery nikdo nezadal.

    promenna     ... x, misto na brehu, kde plavcik vbehne do vody [m]
    ucelova f.   ... doba, za kterou je u tonouciho: beh po pisku + plavani
    omezeni      ... x lezi na useku brehu mezi 0 a 45 m

Pointa: podminka T'(x) = 0 je presne Snelluv zakon lomu,

    sin(theta_1) / v_1 = sin(theta_2) / v_2,

tedy tentyz vztah, kterym se lame svetlo na rozhrani dvou prostredi. Svetlo se
neláme "protoze Snell" - jde nejrychlejsi cestou a plavcik resi tutez ulohu.
Skript to overi numericky: obe strany Snella musi vyjit stejne.

Uloha je striktne konvexni (soucet dvou norem slozenych s afinnim zobrazenim),
takze minimum je jedine a solver ho najde z libovolneho startu.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/ukazka_plavcik.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

A = 25.0                                          # plavcik je 25 m od vody [m]
B = 35.0                                          # tonouci je 35 m od brehu [m]
D = 45.0                                          # vodorovny odstup obou [m]
V_PISEK = 6.0                                     # rychlost behu po pisku [m/s]
V_VODA = 1.2                                      # rychlost plavani [m/s]


def doba(x):
    """Doba, za kterou je plavcik u tonouciho, kdyz vbehne do vody v bode x."""
    return np.hypot(A, x) / V_PISEK + np.hypot(B, D - x) / V_VODA


t0 = time.perf_counter()
res = minimize_scalar(doba, bounds=(0.0, D), method="bounded",
                      options={"xatol": 1e-10})
cas_reseni = time.perf_counter() - t0
x_opt, t_opt = res.x, res.fun

# uhly od kolmice k brehu - presne to, cemu se v optice rika uhel dopadu a lomu
theta1 = np.degrees(np.arctan2(x_opt, A))
theta2 = np.degrees(np.arctan2(D - x_opt, B))
snell_pisek = np.sin(np.radians(theta1)) / V_PISEK
snell_voda = np.sin(np.radians(theta2)) / V_VODA

# srovnavaci trasy: co udela clovek bez pocitani
x_primka = D * A / (A + B)                        # rovnou na tonouciho
varianty = {
    "primka rovnou na tonouciho": x_primka,
    "kolmo do vody, pak sikmo": 0.0,
    "po brehu az naproti, pak kolmo": D,
    "optimum": x_opt,
}

print(f"reseni za {cas_reseni * 1e3:.2f} ms, {res.nfev} vyhodnoceni funkce")
print(f"optimum:  x = {x_opt:.4f} m, T = {t_opt:.4f} s")
print(f"uhly:     theta1 = {theta1:.2f} deg, theta2 = {theta2:.2f} deg")
print(f"Snell:    sin(t1)/v1 = {snell_pisek:.9f}  sin(t2)/v2 = {snell_voda:.9f}"
      f"  rozdil = {abs(snell_pisek - snell_voda):.2e}")
print(f"          sin(t1)/sin(t2) = {np.sin(np.radians(theta1)) / np.sin(np.radians(theta2)):.4f}"
      f"   v1/v2 = {V_PISEK / V_VODA:.4f}")
print()
for jmeno, x in varianty.items():
    t = doba(x)
    print(f"  {jmeno:32s} x = {x:6.2f} m   T = {t:7.3f} s   "
          f"{'—' if jmeno == 'optimum' else f'+{100 * (t / t_opt - 1):.1f} %'}")

# jak ploche je dno: o kolik se zhorsi cas, kdyz se plavcik trefi o 5 m vedle
print()
for d in (2.0, 5.0, 10.0):
    t = max(doba(x_opt - d), doba(min(x_opt + d, D)))
    print(f"  chyba {d:4.0f} m v miste vstupu   ->   +{100 * (t / t_opt - 1):.2f} % casu")

# ---------------------------------------------------------------- obrazek
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

# levy panel: geometrie
ax1.axhspan(-B - 10, 0, color="#cfe6f5")
ax1.axhspan(0, A + 10, color="#f5ead1")
ax1.axhline(0, color="#6b7280", lw=1.2)
ax1.text(D + 5, 19, "pisek\n6 m/s", fontsize=9.5, color="#8a7434")
ax1.text(D + 5, -26, "voda\n1,2 m/s", fontsize=9.5, color="#2b6a8f")

barvy = {"primka rovnou na tonouciho": "#c0392b",
         "kolmo do vody, pak sikmo": "#8e44ad",
         "po brehu az naproti, pak kolmo": "#d68910",
         "optimum": "#1a7f37"}
popisky = {}
for jmeno, x in varianty.items():
    t = doba(x)
    navic = "nejrychlejsi" if jmeno == "optimum" else f"+{100 * (t / t_opt - 1):.1f} %"
    popisky[jmeno] = f"{jmeno} — {t:.2f} s ({navic})"
    lw = 2.8 if jmeno == "optimum" else 1.5
    ax1.plot([0, x, D], [A, 0, -B], color=barvy[jmeno], lw=lw, zorder=2,
             label=popisky[jmeno])

# uhly u optima, merene od kolmice k brehu
ax1.plot([x_opt, x_opt], [-B - 10, A + 10], color="#6b7280", ls=":", lw=0.9, zorder=1)
ax1.annotate(rf"$\theta_1={theta1:.1f}^\circ$", (x_opt - 3.0, 14),
             ha="right", fontsize=11, color="#1a7f37")
ax1.annotate(rf"$\theta_2={theta2:.1f}^\circ$", (x_opt - 3.0, -18),
             ha="right", fontsize=11, color="#1a7f37")

ax1.plot(0, A, "o", ms=9, color="#111827", zorder=3)
ax1.plot(D, -B, "o", ms=9, color="#b91c1c", zorder=3)
ax1.annotate("plavcik", (0, A), xytext=(8, 6), textcoords="offset points", fontsize=10.5)
ax1.annotate("tonouci", (D, -B), xytext=(11, -4), textcoords="offset points",
             fontsize=10.5, ha="left", color="#b91c1c")
ax1.set_xlim(-10, D + 20)
ax1.set_ylim(-B - 10, A + 10)
ax1.set_aspect("equal")
ax1.set_xlabel("vzdalenost podel brehu [m]")
ax1.set_title("Kde vbehnout do vody?")

# pravy panel: ucelova funkce jedine promenne
xs = np.linspace(0, D, 600)
ax2.plot(xs, doba(xs), color="#1f4e79", lw=2.2, zorder=2)
for jmeno, x in varianty.items():
    ax2.plot(x, doba(x), "o", ms=8.5, color=barvy[jmeno], zorder=3,
             label=popisky[jmeno])
ax2.axvline(x_opt, color="#1a7f37", ls="--", lw=1.0, zorder=1)
ax2.annotate(f"minimum\n$x^\\star={x_opt:.2f}$ m\n$T^\\star={t_opt:.2f}$ s",
             (x_opt, t_opt), xytext=(-18, 40), textcoords="offset points",
             ha="right", fontsize=10.5, color="#1a7f37",
             arrowprops=dict(arrowstyle="->", color="#1a7f37"))
ax2.set_xlabel("misto vstupu do vody $x$ [m]")
ax2.set_ylabel("doba do tonouciho $T(x)$ [s]")
ax2.set_title("Ucelova funkce ma jedinou promennou")
ax2.set_ylim(36.2, 53.5)
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig("cviceni/C1/obrazky/plavcik.svg")
print("\nobrazek: cviceni/C1/obrazky/plavcik.svg")
