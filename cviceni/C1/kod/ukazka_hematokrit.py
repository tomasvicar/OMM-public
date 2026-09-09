"""Ukazka pro prvni cviceni: proc nemame hematokrit 80 %, uloha s JEDINOU promennou.

Cervene krvinky nesou kyslik, takze vic krvinek znamena vic kyslika v kazdem
mililitru krve. Jenze viskozita krve roste s hematokritem exponencialne a
prutok trubici je viskozite neprimo umerny. Dve protichudne tendence, optimum
uvnitr - a padne z nej cislo, ktere si nikdo nezadal a ktere presto sedi na
fyziologickou normu cloveka.

    promenna     ... H, hematokrit, objemovy podil cervenych krvinek v krvi [-]
    ucelova f.   ... J(H) = exp(alpha*H) / H, prevracena hodnota dodavky kysliku
    omezeni      ... 0,10 <= H <= 0,70 (rozsah, ktery ma jeste smysl merit)

Model:
    Poiseuille   ... Q = pi * dP * r^4 / (8 * mu * L), pri dane geometrii a
                     tlaku tedy Q ~ 1/mu
    viskozita    ... mu(H) = mu_p * exp(alpha*H), alpha = 2,5 (empiricky fit
                     viskozimetrie plne krve pro H v [0,2; 0,6])
    dodavka O2   ... D(H) ~ H * Q(H) ~ H * exp(-alpha*H)

Maximalizace D je minimalizace J = 1/D. Podminka J'(H) = 0 vede na alpha*H = 1,
takze optimum je analyticky H* = 1/alpha = 0,40 - bez ohledu na tlak, polomer
i viskozitu plazmy. Druha derivace J''(H) = exp(alpha*H) * (alpha^2*H^2 -
2*alpha*H + 2) / H^3 ma v zavorce kvadratiku se zapornym diskriminantem
(4*alpha^2 - 8*alpha^2 < 0), zavorka je tedy vzdy kladna a J je na H > 0
striktne konvexni: minimum je jedine.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/ukazka_hematokrit.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ALPHA = 2.5                                       # exponent viskozity krve [-]
H_MIN = 0.10                                      # dolni mez hematokritu [-]
H_MAX = 0.70                                      # horni mez hematokritu [-]


def viskozita_rel(H):
    """Viskozita krve relativne k plazme, mu(H) / mu_p."""
    return np.exp(ALPHA * H)


def prutok_rel(H):
    """Prutok trubici relativne k prutoku ciste plazmy, Q(H) / Q(0)."""
    return 1.0 / viskozita_rel(H)


def dodavka(H):
    """Dodavka kysliku (az na konstantu): D(H) = H * exp(-alpha*H)."""
    return H * prutok_rel(H)


def ucelova(H):
    """Ucelova funkce J(H) = exp(alpha*H) / H, minimalizuje se."""
    return viskozita_rel(H) / H


t0 = time.perf_counter()
res = minimize_scalar(ucelova, bounds=(H_MIN, H_MAX), method="bounded",
                      options={"xatol": 1e-12})
cas_reseni = time.perf_counter() - t0
h_opt, j_opt = res.x, res.fun
d_opt = dodavka(h_opt)

h_analyt = 1.0 / ALPHA                            # J'(H) = 0  <=>  alpha*H = 1

# konvexita: zavorka z druhe derivace musi byt na cele mrizce kladna
mrizka = np.linspace(1e-3, 1.0, 200_001)
zavorka = ALPHA**2 * mrizka**2 - 2 * ALPHA * mrizka + 2
diskriminant = (2 * ALPHA) ** 2 - 4 * ALPHA**2 * 2

print(f"reseni za {cas_reseni * 1e3:.2f} ms, {res.nfev} vyhodnoceni funkce")
print(f"optimum numericky:   H = {h_opt:.10f}   J = {j_opt:.10f}")
print(f"optimum analyticky:  H = 1/alpha = {h_analyt:.10f}")
print(f"rozdil num. - analyt.:  {abs(h_opt - h_analyt):.3e}")
print(f"konvexita: diskriminant zavorky = {diskriminant:.1f} < 0, "
      f"min zavorky na mrizce = {zavorka.min():.10f} > 0")
print(f"           => J je na H > 0 striktne konvexni, minimum je jedine")

print()
print("relativni dodavka kysliku D(H) / D(H*):")
for H in (0.20, 0.30, 0.37, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
    pomer = dodavka(H) / d_opt
    znacka = "   <- optimum" if abs(H - h_analyt) < 1e-9 else ""
    print(f"  H = {H:.2f}   D/D* = {pomer:.6f}   {100 * pomer:6.2f} %{znacka}")

print()
for H, popis in ((0.45, "typicky muz"),
                 (0.50, "strop UCI z roku 1997"),
                 (0.65, "doping krvi / polycytemie")):
    pomer = dodavka(H) / d_opt
    print(f"  H = {H:.2f}  {popis:28s} {100 * pomer:6.2f} % maxima "
          f"(ztrata {100 * (1 - pomer):.2f} %)")

# jak ploche je dno: o kolik klesne dodavka pri odchylce od optima
print()
for d in (0.02, 0.05, 0.10):
    pomer = min(dodavka(h_analyt - d), dodavka(h_analyt + d)) / d_opt
    print(f"  odchylka {d:+.2f} od optima   ->   dodavka {100 * pomer:.4f} % maxima, "
          f"pokles {100 * (1 - pomer):.4f} %")

print()
for H in (0.40, 0.45, 0.65):
    print(f"  relativni viskozita mu({H:.2f}) / mu_p = {viskozita_rel(H):.6f}")

# ---------------------------------------------------------------- obrazek
hs = np.linspace(0.05, 0.75, 700)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

# levy panel: souboj dvou efektu
# Kazdou dilci krivku normujeme na JEJI VLASTNI maximum v kresleném rozsahu.
# Bez toho se prutok a hematokrit protnou u H = 0,395, tedy opticky presne
# v optimu - a posluchac si odnese, ze optimum je prusecik obou krivek.
# Neni: optimum je maximum jejich SOUCINU a prusecik nema zadny vyznam.
ax1.plot(hs, prutok_rel(hs) / prutok_rel(hs[0]), color="#c0392b", lw=2.2,
         label=r"prutok krve $\propto e^{-\alpha H}$  (klesa)")
ax1.plot(hs, hs / hs[-1], color="#1f4e79", lw=2.2,
         label=r"kyslik na jednotku objemu $\propto H$  (roste)")
ax1.plot(hs, dodavka(hs) / d_opt, color="#1a7f37", lw=3.0,
         label=r"dodavka $D(H)=H\,e^{-\alpha H}$  (soucin)")
ax1.axvline(h_analyt, color="#1a7f37", ls="--", lw=1.0, zorder=1)
ax1.plot(h_analyt, 1.0, "o", ms=8.5, color="#1a7f37", zorder=3)
ax1.annotate(f"$H^\\star = 1/\\alpha = {h_analyt:.2f}$", (h_analyt, 1.0),
             xytext=(26, -26), textcoords="offset points", fontsize=10.5,
             color="#1a7f37", arrowprops=dict(arrowstyle="->", color="#1a7f37"))
ax1.set_xlim(0.05, 0.75)
ax1.set_ylim(0.0, 1.42)
ax1.set_xlabel("hematokrit $H$ [-]")
ax1.set_ylabel("kazda krivka normovana na sve maximum [-]")
ax1.set_title("Souboj dvou efektu")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper center", fontsize=9, framealpha=0.95)

# pravy panel: ucelova funkce jedine promenne
rel = 100 * dodavka(hs) / d_opt
ax2.axvspan(0.42, 0.50, color="#cfe0f5", alpha=0.85, zorder=0)
ax2.axvspan(0.37, 0.47, color="#f5d2e0", alpha=0.55, zorder=0)
ramecek = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
ax2.text(0.46, 39.0, "muzi 0,42-0,50", fontsize=9, color="#1f4e79", ha="center",
         bbox=ramecek, zorder=4)
ax2.text(0.42, 32.5, "zeny 0,37-0,47", fontsize=9, color="#a03060", ha="center",
         bbox=ramecek, zorder=4)

ax2.plot(hs, rel, color="#1a7f37", lw=2.6, zorder=2)
ax2.axvline(h_analyt, color="#1a7f37", ls="--", lw=1.0, zorder=1)
ax2.plot(h_analyt, 100.0, "o", ms=8.5, color="#1a7f37", zorder=3)
ax2.annotate(f"optimum $H^\\star={h_analyt:.2f}$\n100 % dodavky",
             (h_analyt, 100.0), xytext=(-74, 20), textcoords="offset points",
             fontsize=10.5, color="#1a7f37", ha="center",
             arrowprops=dict(arrowstyle="->", color="#1a7f37"))

for H, barva, popis, posun, zarovnani in (
        (0.50, "#d68910", "strop UCI 1997", (30, 20), "left"),
        (0.65, "#c0392b", "doping krvi", (-10, -52), "right")):
    y = 100 * dodavka(H) / d_opt
    ax2.plot(H, y, "o", ms=8.5, color=barva, zorder=3)
    ax2.annotate(f"{popis}\n$H={H:.2f}$: {y:.1f} % maxima", (H, y),
                 xytext=posun, textcoords="offset points", fontsize=10,
                 color=barva, ha=zarovnani,
                 arrowprops=dict(arrowstyle="->", color=barva))

ax2.set_xlim(0.05, 0.75)
ax2.set_ylim(28, 112)
ax2.set_xlabel("hematokrit $H$ [-]")
ax2.set_ylabel("dodavka kysliku $D(H)/D(H^\\star)$ [%]")
ax2.set_title("Ucelova funkce ma jedinou promennou")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("cviceni/C1/obrazky/hematokrit.svg")
print("\nobrazek: cviceni/C1/obrazky/hematokrit.svg")
