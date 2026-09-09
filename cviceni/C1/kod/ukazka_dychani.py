"""Ukazka pro prvni cviceni: optimalni dychani, uloha se DVEMA promennymi.

Musite provetrat 4,2 litru alveolu za minutu. Da se to udelat rychle a mele,
nebo pomalu a zhluboka - a kazda z tech voleb stoji dychaci svaly jinak. Uloha
ma dve promenne, takze se do jedne roviny vejde CELA: vrstevnice ucelove funkce
i pripustna oblast. To predchozi dve ukazky (plavcik, hematokrit) neumi, protoze
maji jedinou promennou, a ctyri nasledujici uz taky ne, protoze jich maji desitky
az tisice.

    promenne     ... f, dechova frekvence [1/min]
                     V_T, dechovy objem [l]
    ucelova f.   ... vykon dychacich svalu W = elasticka + odporova slozka
    omezeni      ... f*(V_T - V_D) >= 4,2 l/min  (alveolarni ventilace)
                     4 <= f <= 60, V_D + eps <= V_T <= 2,5

Model (Otis, Fenn & Rahn 1950):
    elasticka   ... prace na roztazeni plic za dech je 1/2 * E * V_T^2, za minutu
                    tedy 1/2 * E * V_T^2 * f - roste s objemem dechu
    odporova    ... pri sinusovem prutoku je stredni hodnota V'^2 rovna
                    (pi*f*V_T)^2/2; po prevodu R z sekund na minuty vyjde
                    (pi^2/120) * R * f^2 * V_T^2 - roste s frekvenci
    mrtvy prostor . kazdy dech provetra jen V_T - V_D, protoze V_D = 0,15 l
                    zustane v trachee a bronsich a k alveolam se nedostane

Pointa neni "fyziologie sedi v optimu" (to nese hematokrit o dve ukazky driv),
ale GEOMETRIE: ucelova funkce roste obema smery, takze optimum by samo o sobe
utikalo k nule; jedina vec, ktera ho drzi, je omezeni. Optimum proto lezi
na HRANICI pripustne oblasti a vrstevnice ucelove funkce se te hranice
DOTYKA v jedinem bode. To je Lagrangeuv multiplikator nakresleny, tri tydny
pred tim, nez se v ctvrtem cviceni pojmenuje KKT.

Kontrola, ktera umi selhat: po dosazeni aktivniho omezeni f = 4,2/(V_T - V_D)
zbyde jedna promenna a podminka dW/dV_T = 0 se da vyresit tuzkou - vyjde
kvadraticka rovnice. Kontrola projde na reseni solveru a SPADNE na podvrzenem
reseni f = 12, V_T = 0,5, ktere je pritom pripustne, lezi presne na hranici
a vypada jako z ucebnice fyziologie.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/ukazka_dychani.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

E = 10.0                                          # elastance plic [cmH2O/l]
R = 2.0                                           # odpor dychacich cest [cmH2O*s/l]
V_D = 0.15                                        # mrtvy prostor [l]
VA = 4.2                                          # pozadovana alveolarni ventilace [l/min]

F_MIN, F_MAX = 4.0, 60.0                          # meze dechove frekvence [1/min]
VT_MIN, VT_MAX = V_D + 1e-3, 2.5                  # meze dechoveho objemu [l]

CMH2O_L_NA_J = 98.0665e-3                         # 1 cmH2O*l = 98,0665 Pa * 1e-3 m^3


def vykon(f, vt, e=E, r=R):
    """Vykon dychacich svalu [cmH2O*l/min]: elasticka + odporova slozka."""
    return 0.5 * e * vt**2 * f + (np.pi**2 / 120.0) * r * f**2 * vt**2


def ventilace(f, vt, vd=V_D):
    """Alveolarni ventilace [l/min]: provetra se jen objem nad mrtvym prostorem."""
    return f * (vt - vd)


def gradient(f, vt, e=E, r=R):
    """Gradient vykonu podle (f, V_T)."""
    k = np.pi**2 / 120.0
    return np.array([0.5 * e * vt**2 + 2 * k * r * f * vt**2,
                     e * vt * f + 2 * k * r * f**2 * vt])


def analyticke_optimum(e=E, r=R, vd=V_D, va=VA):
    """Optimum tuzkou: po dosazeni aktivniho omezeni zbyde jedna promenna.

    Substituce v = V_T - vd, f = va/v prevede ucelovou funkci na
        W(v) = a*(v+vd)^2/v + b*(v+vd)^2/v^2,    a = va*e/2,  b = (pi^2/120)*r*va^2,
    a podminka dW/dv = 0 se po vykraceni kladneho cinitele (v+vd) zredukuje na
        a*v^2 - a*vd*v - 2*b*vd = 0,
    tedy na kvadratickou rovnici s jedinym kladnym korenem. Zaroven je odsud
    videt, ze reseni zavisi jen na POMERU b/a, tedy na r/e.
    """
    a = va * e / 2.0
    b = (np.pi**2 / 120.0) * r * va**2
    koreny = np.roots([a, -a * vd, -2 * b * vd])
    v = float(max(koreny[np.isreal(koreny)].real))
    return va / v, v + vd


# ------------------------------------------------------- reseni solverem
omezeni = [{"type": "ineq", "fun": lambda z: ventilace(z[0], z[1]) - VA}]
meze = [(F_MIN, F_MAX), (VT_MIN, VT_MAX)]

t0 = time.perf_counter()
res = minimize(lambda z: vykon(z[0], z[1]), x0=np.array([20.0, 0.8]),
               method="SLSQP", bounds=meze, constraints=omezeni,
               options={"ftol": 1e-12, "maxiter": 200})
cas_reseni = time.perf_counter() - t0
f_opt, vt_opt = res.x
w_opt = vykon(f_opt, vt_opt)

print(f"reseni za {cas_reseni * 1e3:.1f} ms, {res.nit} iteraci, "
      f"{res.nfev} vyhodnoceni funkce ({res.message})")
print(f"optimum:  f = {f_opt:.4f} /min,  V_T = {vt_opt:.5f} l")
print(f"          minutova ventilace f*V_T = {f_opt * vt_opt:.4f} l/min")
print(f"          alveolarni ventilace     = {ventilace(f_opt, vt_opt):.6f} l/min "
      f"(pozadavek {VA})")
print(f"          W = {w_opt:.4f} cmH2O*l/min = {w_opt * CMH2O_L_NA_J:.4f} J/min")
el = 0.5 * E * vt_opt**2 * f_opt
od = w_opt - el
print(f"          z toho elasticka {el:.4f} ({100 * el / w_opt:.1f} %), "
      f"odporova {od:.4f} ({100 * od / w_opt:.1f} %)")

vule = ventilace(f_opt, vt_opt) - VA
print(f"\nomezeni alveolarni ventilace: vule = {vule:.2e} l/min  ->  "
      f"{'AKTIVNI' if abs(vule) < 1e-6 else 'neaktivni'}")
print(f"meze f a V_T:  f = {f_opt:.3f} v ({F_MIN}; {F_MAX}), "
      f"V_T = {vt_opt:.3f} v ({VT_MIN:.3f}; {VT_MAX}) -> obe neaktivni")

# ------------------------------------------------------- multiplikator
# V optimu plati grad W = lambda * grad g, g(f,V_T) = f*(V_T - V_D). Staci tedy
# vydelit odpovidajici slozky; obe musi dat totez, coz je zaroven kontrola, ze
# vrstevnice ucelove funkce a hranice pripustne oblasti se opravdu DOTYKAJI.
gw = gradient(f_opt, vt_opt)
gg = np.array([vt_opt - V_D, f_opt])
lam_f, lam_vt = gw[0] / gg[0], gw[1] / gg[1]
print(f"\nmultiplikator z podminky grad W = lambda * grad g:")
print(f"  ze slozky f:    lambda = {lam_f:.6f}")
print(f"  ze slozky V_T:  lambda = {lam_vt:.6f}   (rozdil {abs(lam_f - lam_vt):.2e})")
lam = 0.5 * (lam_f + lam_vt)
kosinus = gw @ gg / (np.linalg.norm(gw) * np.linalg.norm(gg))
uhel = np.degrees(np.arccos(np.clip(kosinus, -1.0, 1.0)))
print(f"  uhel mezi gradienty: {uhel:.6f} deg  -> vrstevnice se hranice dotyka")

for krok in (0.1, 0.5):                           # overeni konecnou diferenci
    f2, vt2 = analyticke_optimum(va=VA + krok)
    dw = vykon(f2, vt2) - w_opt
    print(f"  pozadavek {VA} -> {VA + krok} l/min:  W roste o {dw:.4f} "
          f"(predpoved lambda*delta = {lam * krok:.4f})")

# ------------------------------------------------------- kontrola c. 1
# Dosazeni aktivniho omezeni a vyreseni tuzkou. Kontrola, ktera UMI SELHAT:
# projde na reseni solveru a spadne na pripustnem, ale spatnem reseni.
f_an, vt_an = analyticke_optimum()
print(f"\nkontrola 1 - dosazeni omezeni a vyreseni tuzkou (kvadratika v V_T - V_D):")
print(f"  analyticky:  f = {f_an:.9f} /min,  V_T = {vt_an:.9f} l")
print(f"  SLSQP:       f = {f_opt:.9f} /min,  V_T = {vt_opt:.9f} l")
print(f"  rozdil:      |df| = {abs(f_an - f_opt):.2e},  |dV_T| = {abs(vt_an - vt_opt):.2e}")


def kontrola(f, vt, tol=1e-4):
    """Vrati (prosla, popis). Testuje pripustnost A stacionaritu zaroven."""
    if ventilace(f, vt) < VA - 1e-9:
        return False, f"nepripustne, alveolarni ventilace {ventilace(f, vt):.4f} < {VA}"
    zbytek = max(abs(f - f_an), abs(vt - vt_an))
    if zbytek > tol:
        return False, f"pripustne, ale neni to optimum: odchylka {zbytek:.4f}"
    return True, f"prochazi, odchylka {zbytek:.2e}"


print("\n  test kontroly na trech resenich:")
for jmeno, (f, vt) in (("reseni SLSQP", (f_opt, vt_opt)),
                       ("podvrzene: f = 12, V_T = 0,50 (z ucebnice)", (12.0, 0.5)),
                       ("podvrzene: f = 20, V_T = 0,36", (20.0, 0.36))):
    prosla, popis = kontrola(f, vt)
    print(f"    {jmeno:44s} {'PROSLA ' if prosla else 'SPADLA '} - {popis}")
print("  obe podvrzena reseni lezi PRESNE na hranici (4,200000 l/min), takze")
print("  samotna kontrola pripustnosti je propusti - odhali je az stacionarita")

# ------------------------------------------------------- kontrola c. 2
# Hruba mrizka pres celou pripustnou oblast: nezavisla na solveru i na tuzce.
fs = np.linspace(F_MIN, 40.0, 1801)
vts = np.linspace(VT_MIN, 1.2, 1801)
FF, VV = np.meshgrid(fs, vts)
WW = vykon(FF, VV)
WW = np.where(ventilace(FF, VV) >= VA, WW, np.inf)
i = np.unravel_index(np.argmin(WW), WW.shape)
print(f"\nkontrola 2 - hruba mrizka {len(fs)}x{len(vts)} pres pripustnou oblast:")
print(f"  nejlepsi bod mrizky:  f = {FF[i]:.4f}, V_T = {VV[i]:.5f}, W = {WW[i]:.4f}")
print(f"  rozdil proti solveru: dW = {WW[i] - w_opt:.2e} cmH2O*l/min "
      f"({100 * (WW[i] / w_opt - 1):.4f} %)")
print(f"  ale v samotne frekvenci |df| = {abs(FF[i] - f_opt):.3f} /min, tedy "
      f"{abs(FF[i] - f_opt) / (fs[1] - fs[0]):.0f} kroku mrizky "
      f"({fs[1] - fs[0]:.3f} /min) - to uz je ploche dno, ne chyba")

# ------------------------------------------------------- kontrola c. 3
# Multistart: ucelova funkce neni spolecne konvexni, takze se nesmi verit
# jedinemu startu.
rng = np.random.default_rng(0)
starty = np.column_stack([rng.uniform(F_MIN, F_MAX, 200),
                          rng.uniform(VT_MIN, 1.5, 200)])
naslo, nejhorsi = 0, 0.0
for z0 in starty:
    r = minimize(lambda z: vykon(z[0], z[1]), x0=z0, method="SLSQP",
                 bounds=meze, constraints=omezeni, options={"ftol": 1e-12})
    if r.success:
        naslo += 1
        nejhorsi = max(nejhorsi, abs(r.x[0] - f_opt), abs(r.x[1] - vt_opt))
print(f"\nkontrola 3 - multistart z {len(starty)} nahodnych bodu:")
print(f"  konvergovalo {naslo}, vsechna do tehoz optima "
      f"(nejvetsi odchylka {nejhorsi:.2e})")

# ------------------------------------------------------- cena chyby
# Pri dane frekvenci je nejlevnejsi dechovy objem ten nejmensi pripustny,
# tedy V_T = V_D + 4,2/f: kdo dycha jinou frekvenci, jede po TEZE hranici.
print("\ncena chyby (pri kazde frekvenci nejlevnejsi pripustny dechovy objem):")
for f in (8.0, 10.0, 12.0, 14.385, 16.0, 20.0, 25.0, 30.0, 40.0):
    vt = V_D + VA / f
    w = vykon(f, vt)
    znacka = "   <- optimum" if abs(f - f_opt) < 0.01 else ""
    print(f"  f = {f:5.1f} /min   V_T = {vt:.3f} l   W = {w:7.4f}   "
          f"+{100 * (w / w_opt - 1):5.2f} %{znacka}")

# ------------------------------------------------------- nemoci
print("\ncizi plice: optimum sjizdi po TEZE hranici, jen jinam")
nemoci = [("zdravy", E, R), ("fibroza (C = 0,04 l/cmH2O, tedy E = 25)", 25.0, R),
          ("CHOPN (R = 10 cmH2O*s/l)", E, 10.0)]
for jmeno, e, r in nemoci:
    f, vt = analyticke_optimum(e=e, r=r)
    print(f"  {jmeno:40s} f = {f:6.2f} /min, V_T = {vt:.3f} l, "
          f"W = {vykon(f, vt, e, r):7.3f} ({vykon(f, vt, e, r) * CMH2O_L_NA_J:.3f} J/min)")

print("\noptimum zavisi jen na pomeru R/E:")
for e, r in ((10.0, 2.0), (20.0, 4.0), (5.0, 1.0)):
    f, vt = analyticke_optimum(e=e, r=r)
    print(f"  E = {e:5.1f}, R = {r:4.1f}  (R/E = {r / e:.2f})  ->  "
          f"f = {f:.6f} /min, V_T = {vt:.6f} l")

# bez mrtveho prostoru: V_T = 4,2/f, takze odporova slozka na f vubec nezavisi
# (f^2 * V_T^2 = 4,2^2 = konst) a elasticka klesa jako 1/f. Uloha proto nema
# vnitrni optimum a reseni utece az na mez frekvence.
r0 = minimize(lambda z: vykon(z[0], z[1]), x0=np.array([20.0, 0.8]), method="SLSQP",
              bounds=[(F_MIN, F_MAX), (1e-3, VT_MAX)], options={"ftol": 1e-12},
              constraints=[{"type": "ineq",
                            "fun": lambda z: ventilace(z[0], z[1], vd=0.0) - VA}])
print(f"\nbez mrtveho prostoru (V_D = 0) vnitrni optimum NEEXISTUJE:")
print(f"  SLSQP skonci na f = {r0.x[0]:.4f} /min, V_T = {r0.x[1]:.4f} l, "
      f"tedy presne na mezi f <= {F_MAX:.0f}")
print("  odporova slozka na f nezavisi (f^2*V_T^2 = 4,2^2 = konst) a elasticka")
print("  klesa jako 1/f, takze reseni utika k rychlemu melkemu dychani.")
print("  Mrtvy prostor je fyziologicky duvod, proc optimum vubec existuje.")

print("\nnamerene klidove hodnoty pro srovnani: 12-16 dechu/min, V_T ~ 0,5 l,")
print("  minutova ventilace ~ 6 l/min, prace dychani 2-3 J/min")

# ---------------------------------------------------------------- obrazek
BARVA_H = "#1f4e79"                               # hranice pripustne oblasti
BARVA_W = "#c0392b"                               # vrstevnice ucelove funkce
BARVA_O = "#1a7f37"                               # optimum
F_OD, F_DO, VT_DO = 5.0, 40.0, 1.05


def vrstevnice(f, w):
    """Vrstevnice W(f, V_T) = w vyjadrena pro V_T; kvadratika ve V_T se da odmocnit."""
    return np.sqrt(w / (0.5 * E * f + (np.pi**2 / 120.0) * R * f**2))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))

fs_kresba = np.linspace(F_OD, F_DO, 400)
hranice = V_D + VA / fs_kresba                    # f*(V_T - V_D) = 4,2

for ax in (ax1, ax2):
    ax.fill_between(fs_kresba, hranice, VT_DO, color="#cbe5d6", zorder=0)
    ax.fill_between(fs_kresba, 0.0, hranice, color="#e2e2e9", zorder=0)
    ax.plot(fs_kresba, hranice, color=BARVA_H, lw=3.4, zorder=4)
    ax.set_xlim(F_OD, F_DO)
    ax.set_ylim(0.0, VT_DO)
    ax.set_xlabel("dechova frekvence $f$ [1/min]", fontsize=14)
    ax.set_ylabel("dechovy objem $V_T$ [l]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(32, 0.90, "dost kysliku", fontsize=15, color="#256b3c", ha="center")

# ---- levy panel: nejnizsi dosazitelna vrstevnice se hranice dotkne
# Vrstevnice se kresli z uzavreneho vzorce, ne pres contour() - dotyk v jedinem
# bode by se na mrizce rozmazal a obrazek by tvrdil neco jineho nez text.
ramecek = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)
for w, popis, misto in ((14.0, "$\\dot W$ = 14", (6.0, 0.545)),
                        (w_opt, None, None),
                        (40.0, "$\\dot W$ = 40", (7.4, 0.905))):
    ax1.plot(fs_kresba, vrstevnice(fs_kresba, w), color=BARVA_W, lw=2.0,
             ls="--", zorder=3)
    if popis:
        ax1.text(misto[0], misto[1], popis, fontsize=13.5, color=BARVA_W,
                 ha="left", bbox=ramecek, zorder=7)

ax1.text(11.0, 0.06, "hypoventilace", fontsize=15, color="#55555f", ha="center")
ax1.plot(f_opt, vt_opt, "o", ms=14, color=BARVA_O, zorder=6)
ax1.annotate("optimum\n$f^\\star$ = 14,4/min\n$V_T^\\star$ = 0,44 l",
             (f_opt, vt_opt), xytext=(72, 46), textcoords="offset points",
             fontsize=14, color=BARVA_O, fontweight="bold", ha="center",
             bbox=ramecek, zorder=7,
             arrowprops=dict(arrowstyle="->", color=BARVA_O, lw=2.2))
ax1.annotate("$f\\,(V_T - 0{,}15) = 4{,}2$", (9.0, V_D + VA / 9.0),
             xytext=(96, 62), textcoords="offset points", fontsize=14,
             color=BARVA_H, fontweight="bold", ha="center", bbox=ramecek, zorder=7,
             arrowprops=dict(arrowstyle="->", color=BARVA_H, lw=1.8))
ax1.annotate("$\\dot W^\\star$ = 20,7 — nejnizsi vrstevnice,\nktera do oblasti jeste dosahne",
             (21.0, vrstevnice(21.0, w_opt)), xytext=(0, -62),
             textcoords="offset points", fontsize=13, color=BARVA_W, ha="center",
             bbox=ramecek, zorder=7,
             arrowprops=dict(arrowstyle="->", color=BARVA_W, lw=1.8))
ax1.set_title("Vrstevnice se hranice dotkne v jedinem bode", fontsize=15)

# ---- pravy panel: nemocne plice sjizdeji po teze hranici
ax2.text(26, 0.09, "hypoventilace", fontsize=15, color="#55555f", ha="center")
znacky = [("zdravy", E, R, "#1a7f37", (62, 54)),
          ("fibroza\n(tuzsi plice)", 25.0, R, "#c0392b", (60, -64)),
          ("CHOPN\n(vetsi odpor)", E, 10.0, "#8e44ad", (86, 28))]
body = []
for jmeno, e, r, barva, posun in znacky:
    f, vt = analyticke_optimum(e=e, r=r)
    body.append((f, vt))
    ax2.annotate(f"{jmeno}\n{f:.1f}/min, {vt:.2f} l".replace(".", ","), (f, vt),
                 xytext=posun, textcoords="offset points", fontsize=13.5,
                 color=barva, fontweight="bold", ha="center", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color=barva, lw=1.8))

# sipky vedou PO hranici, ne pres oblast - optimum se z hranice nikdy nehne
for (fa, _), (fb, _) in ((body[0], body[1]), (body[0], body[2])):
    kroky = np.linspace(fa, fb, 40)
    ax2.plot(kroky, V_D + VA / kroky, color="#f0a500", lw=6.0, alpha=0.85, zorder=5)
    ax2.annotate("", xy=(fb, V_D + VA / fb), xytext=(kroky[-6], V_D + VA / kroky[-6]),
                 zorder=5, arrowprops=dict(arrowstyle="-|>", color="#f0a500",
                                           lw=6.0, mutation_scale=26, shrinkB=8))
for (f, vt), (_, _, _, barva, _) in zip(body, znacky):
    ax2.plot(f, vt, "o", ms=14, color=barva, zorder=6)
ax2.set_title("Nemocne plice sjizdeji po teze hranici", fontsize=15)

fig.tight_layout()
fig.savefig("cviceni/C1/obrazky/dychani.svg")
print("\nobrazek: cviceni/C1/obrazky/dychani.svg")
