"""Tabulova uloha prvniho cviceni: optimalni hematokrit tuzkou.

Tenhle skript je JEDINY ZDROJ CISEL pro sekci "Pokracovani cviceni"
(reporty/pokracovani.html) - stacionarni bod, hodnoty na mezich, ploche dno,
druha derivace i numericke overeni konvexity. Nic z toho se do reportu nepise
od oka; co se rekne nahlas, vytiskne tenhle skript.

    promenna   H ... hematokrit, objemovy podil cervenych krvinek [-]
    maximize   D(H) = H * exp(-alpha*H)        dodavka kysliku (rel. jednotky)
    subject to 0,20 <= H <= 0,60               rozsah, kde model plati

Model je tri radky (podrobne v ../ukazka-hematokrit.md):
    Poiseuille  ... Q ~ 1/mu pri dane geometrii a tlakovem spadu
    viskozita   ... mu(H) = mu_p * exp(alpha*H), alpha = 2,5 (empiricky fit)
    dodavka O2  ... D(H) ~ H * Q(H) ~ H * exp(-alpha*H)

Proc se to na tabuli pocita pres MINUS LOGARITMU: maximalizace D je totez co
minimalizace -D, a protoze log je rostouci, i totez co minimalizace
g = -ln D = alpha*H - ln H. Podminka g'(H) = alpha - 1/H = 0 dava H* = 1/alpha
na jeden radek a g''(H) = 1/H^2 > 0 overi konvexitu na druhy. Prevracena
hodnota 1/D by tady vysla nastejno, ale otaci max na min jen pro kladnou
ucelovou funkci - minus funguje vzdycky, proto je na tabuli minus.
Skript overuje, ze maximum D a minimum g vyjdou stejne (log ani minus optimum
neposunuly) a ze g je konvexni na celem H > 0, zatimco -D uz ne.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/tabule_hematokrit.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ALPHA = 2.5                                       # exponent viskozity krve [-]
H_MIN, H_MAX = 0.20, 0.60                         # meze, kde ma model smysl [-]


def dodavka(H):
    """Dodavka kysliku (az na konstantu): D(H) = H * exp(-alpha*H)."""
    return H * np.exp(-ALPHA * H)


def g(H):
    """Ucelova funkce po prevodu na minimalizaci: g(H) = -ln D = alpha*H - ln H."""
    return ALPHA * H - np.log(H)


h_opt = 1.0 / ALPHA                               # g'(H) = alpha - 1/H = 0
print("=== stacionarni bod tuzkou ===")
print(f"g'(H)  = alpha - 1/H = 0   =>   H* = 1/alpha = {h_opt:.4f}  ({100*h_opt:.0f} %)")
print(f"g''(H) = 1/H^2 = {1/h_opt**2:.4f} > 0   =>   minimum, ne maximum ani inflexe")
print(f"D(H*) = {dodavka(h_opt):.6f}   (= 1/(alpha*e) = {1/(ALPHA*np.e):.6f})")
print(f"g(H*) = {g(h_opt):.6f}   (kontrola -ln D(H*) = {-np.log(dodavka(h_opt)):.6f})")

# --- numericka kontrola: totez pres solver, a to jak na D, tak na g ---
res_d = minimize_scalar(lambda H: -dodavka(H), bounds=(H_MIN, H_MAX), method="bounded",
                        options={"xatol": 1e-12})
res_g = minimize_scalar(g, bounds=(H_MIN, H_MAX), method="bounded",
                        options={"xatol": 1e-12})
print("\n=== kontrola solverem (tuzka vs. numerika) ===")
print(f"maximalizace D: H = {res_d.x:.10f}   odchylka od 1/alpha = {abs(res_d.x-h_opt):.2e}")
print(f"minimalizace g: H = {res_g.x:.10f}   odchylka od 1/alpha = {abs(res_g.x-h_opt):.2e}")
print("minus ani logaritmus optimum neposunuly - je to zkratka, ne jina uloha")

# --- meze: jsou aktivni, nebo ne? ---
print("\n=== jsou meze aktivni? ===")
for H, popis in ((H_MIN, "dolni mez"), (h_opt, "stacionarni bod"), (H_MAX, "horni mez")):
    print(f"{popis:16s} H = {H:.2f}   D = {dodavka(H):.6f}"
          f"   {100*dodavka(H)/dodavka(h_opt):6.2f} % maxima")
print(f"optimum lezi UVNITR intervalu ({H_MIN} < {h_opt} < {H_MAX})"
      " -> obe meze jsou NEAKTIVNI a daly by se z ulohy vyskrtnout")

# --- srovnani s realitou: cislo, ktere si nikdo nezadal ---
print("\n=== co na to fyziologie ===")
for H, popis in ((0.42, "typicka zena (0,37-0,47)"),
                 (0.45, "typicky muz (0,40-0,50)"),
                 (0.50, "strop UCI z roku 1997"),
                 (0.65, "krevni doping / polycytemie")):
    pomer = dodavka(H) / dodavka(h_opt)
    print(f"  H = {H:.2f}  {popis:28s} {100*pomer:6.2f} % maxima"
          f"   (ztrata {100*(1-pomer):.2f} %)")

# --- ploche dno: o kolik se plati odchylka od optima ---
print("\n=== jak ploche je dno (u LP ve cviceni 2 to bude jinak) ===")
for d in (0.02, 0.05, 0.10):
    pomer = min(dodavka(h_opt - d), dodavka(h_opt + d)) / dodavka(h_opt)
    print(f"  odchylka {d:+.2f} od optima -> dodavka {100*pomer:.3f} % maxima"
          f"   (ztrata {100*(1-pomer):.3f} %)")

# --- konvexita: tuzkou na g, numericky i na samotnem -D ---
print("\n=== overeni konvexity ===")
mrizka = np.linspace(1e-3, 1.0, 200_001)
g_druha = 1.0 / mrizka**2
minus_d_druha = ALPHA * np.exp(-ALPHA * mrizka) * (2 - ALPHA * mrizka)
print(f"g''(H)  = 1/H^2                    : min na mrizce = {g_druha.min():.6f} > 0")
print("=> g je na celem H > 0 striktne konvexni: stacionarni bod je JEDINY"
      " a je to globalni minimum")
print(f"(-D)''(H) = a*e^(-aH)*(2 - a*H)    : min na mrizce = {minus_d_druha.min():.6f}")
print(f"  meni znamenko v H = 2/alpha = {2/ALPHA:.2f}, takze samotne -D uz konvexni"
      " neni - logaritmus tedy neni jen kosmetika")

# kontrola definice konvexity primo: useckou nad grafem, nahodne dvojice bodu
generator = np.random.default_rng(0)
a, b = generator.uniform(0.05, 1.0, 4000), generator.uniform(0.05, 1.0, 4000)
t = generator.uniform(0.0, 1.0, 4000)
rozdil = (t * g(a) + (1 - t) * g(b)) - g(t * a + (1 - t) * b)
print(f"definice primo: min(useckou - graf) na 4000 dvojicich = {rozdil.min():.3e} >= 0")

# --- co se stane, kdyz se zameni max za min (typicka chyba) ---
print("\n=== rozbita varianta: zamena maximalizace za minimalizaci ===")
res_spatne = minimize_scalar(lambda H: dodavka(H), bounds=(H_MIN, H_MAX),
                             method="bounded", options={"xatol": 1e-12})
print(f"minimalizace D misto maximalizace -> H = {res_spatne.x:.4f}, tedy na MEZI")
print("solver nic nehlasi, vypise cislo a vypada spokojene;"
      " pozna se to jedine dosazenim zpet do zadani")

# --- kontrola dosazenim zpet ---
assert abs(h_opt - res_d.x) < 1e-6, "tuzka a solver se rozesly"
assert H_MIN <= h_opt <= H_MAX, "optimum vypadlo z rozsahu platnosti modelu"
assert dodavka(h_opt) >= dodavka(np.linspace(H_MIN, H_MAX, 100_001)).max() - 1e-12, \
    "nasel se lepsi bod nez stacionarni"
print("\nkontrola dosazenim zpet prosla")


# ---------------------------------------------------------------- obrazek ----
fig, ax1 = plt.subplots(figsize=(7.6, 5.0))
hs = np.linspace(0.08, 0.85, 700)

# --- levy panel: dodavka kysliku a stacionarni bod ---
ax1.axvspan(0.40, 0.45, color="#dcfce7", zorder=0)
ax1.annotate("fyziologicka\nnorma 40-45 %", (0.425, 0.0125), ha="center",
             fontsize=10, color="#166534")
ax1.plot(hs, dodavka(hs), color="#1f4e79", lw=2.4)
ax1.axhline(dodavka(h_opt), color="#1a7f37", ls="--", lw=1.6)
ax1.plot(h_opt, dodavka(h_opt), "o", ms=11, color="#1a7f37", zorder=5)
ax1.annotate(f"$H^\\star = 1/\\alpha = 0{{,}}40$\ntecna je vodorovna: $D'=0$",
             (h_opt, dodavka(h_opt)), xytext=(26, 22), textcoords="offset points",
             fontsize=11.5, color="#1a7f37",
             arrowprops=dict(arrowstyle="->", color="#1a7f37"))
for H, popis in ((H_MIN, "dolni mez"), (H_MAX, "horni mez")):
    ax1.axvline(H, color="#b45309", ls=":", lw=1.6)
    ax1.annotate(popis, (H, 0.006), rotation=90, ha="right", va="bottom",
                 fontsize=10, color="#b45309")
ax1.set_xlim(0.08, 0.85)
ax1.set_ylim(0, 0.187)
ax1.set_xlabel("hematokrit $H$ [-]")
ax1.set_ylabel("dodavka kysliku $D(H)=H\\,e^{-\\alpha H}$ [rel.]")
ax1.set_title("Cela uloha v jedne krivce: optimum lezi uvnitr")
ax1.grid(alpha=0.28)

fig.tight_layout()
fig.savefig("cviceni/C1/obrazky/tabule-hematokrit.svg")
print("obrazek: cviceni/C1/obrazky/tabule-hematokrit.svg")
