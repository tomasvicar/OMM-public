"""Tabulova uloha cviceni 2 (LP): infuzni smes ze dvou roztoku glukozy.

Puvodne to byla tabulova uloha prvniho cviceni; presunuta sem 24. 8. 2026,
protoze do uvodni hodiny patri nejdriv zaklady (optimum pres derivaci
a konvexita), ne linearni program. Podrobnosti v ../README.md.

Tenhle skript je JEDINY ZDROJ CISEL pro report
(C2/reporty/tabule-infuze.html) - vrcholy pripustne oblasti, ceny v nich, stinove
ceny, meze, ve kterych optimum zustava ve stejnem vrcholu, i obe rozbite
varianty ulohy. Nic z toho se do reportu nepise od oka; co se rekne nahlas,
vytiskne tenhle skript.

    promenne   x1, x2 ... objem roztoku A a B v litrech
    minimize   20*x1 + 50*x2                  (cena v Kc)
    subject to 50*x1 + 200*x2 >= 150          (glukoza, g)
               x1 + x2        >= 1.5          (objem, l)
               x1, x2         >= 0

Uloha je zamerne tak mala, ze se da spocitat na tabuli: optimum lezi v pruseciku
obou omezeni, vyjde na hezka cisla (1 l a 0,5 l, cena 45 Kc) a obe omezeni jsou
v nem aktivni. Zaklad (optimum a stinove ceny) prepocitava nezavisle jeste
lp_infuze_kontrola.py pres CVXPY i scipy.optimize.linprog.

Spusteni z korene repozitare:
    uv run python cviceni/C2/kod/tabule_infuze.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

CENA_A, CENA_B = 20.0, 50.0                       # Kc za litr roztoku A a B
GLU_A, GLU_B = 50.0, 200.0                        # g glukozy v litru roztoku
POZ_GLU = 150.0                                   # pozadavek na glukozu [g]
POZ_OBJ = 1.5                                     # pozadavek na objem [l]


def optimum(cena_a=CENA_A, cena_b=CENA_B, poz_glu=POZ_GLU, poz_obj=POZ_OBJ):
    """Vyresi LP a vrati (x1, x2, cena). Standardni tvar scipy je A_ub x <= b_ub."""
    res = linprog(c=[cena_a, cena_b],
                  A_ub=[[-GLU_A, -GLU_B], [-1.0, -1.0]],
                  b_ub=[-poz_glu, -poz_obj],
                  bounds=[(0, None), (0, None)])
    return res.x[0], res.x[1], res.fun


def pripustny(x1, x2, poz_glu=POZ_GLU, poz_obj=POZ_OBJ, tol=1e-9):
    return (GLU_A * x1 + GLU_B * x2 >= poz_glu - tol
            and x1 + x2 >= poz_obj - tol and x1 >= -tol and x2 >= -tol)


def cena(x1, x2, cena_a=CENA_A, cena_b=CENA_B):
    return cena_a * x1 + cena_b * x2


x1_opt, x2_opt, cena_opt = optimum()
print("=== optimum ===")
print(f"x* = ({x1_opt:.4f}; {x2_opt:.4f}) l, cena = {cena_opt:.2f} Kc")
print(f"glukoza v optimu = {GLU_A*x1_opt + GLU_B*x2_opt:.2f} g (pozadavek {POZ_GLU})")
print(f"objem v optimu   = {x1_opt + x2_opt:.2f} l  (pozadavek {POZ_OBJ})")

# --- vrcholy pripustne oblasti: prusecik obou omezeni a oba prusecky s osami ---
# prusecik: 50 x1 + 200 x2 = 150 a x1 + x2 = 1.5
x1_pr = (GLU_B * POZ_OBJ - POZ_GLU) / (GLU_B - GLU_A)
x2_pr = POZ_OBJ - x1_pr
vrcholy = {
    "jen roztok A (na ose x1)": (POZ_GLU / GLU_A, 0.0),
    "prusecik obou omezeni": (x1_pr, x2_pr),
    "jen roztok B (na ose x2)": (0.0, POZ_OBJ),
}
print("\n=== vrcholy pripustne oblasti (kandidati na optimum) ===")
for jmeno, (a, b) in vrcholy.items():
    print(f"{jmeno:26s} ({a:.4f}; {b:.4f}) l  cena {cena(a, b):6.2f} Kc"
          f"  {'pripustny' if pripustny(a, b) else 'NEPRIPUSTNY'}")

# --- smernice primek: proc optimum sedi zrovna v pruseciku ---
print("\n=== smernice v rovine (x1, x2) ===")
print(f"vrstevnice ceny : {-CENA_A/CENA_B:+.3f}")
print(f"omezeni glukoza : {-GLU_A/GLU_B:+.3f}")
print(f"omezeni objem   : {-1.0:+.3f}")
print("vrstevnice lezi smernici mezi obema omezenimi -> optimum je jejich prusecik")

# --- stinove ceny konecnou diferenci (tj. presne to, co jde spocitat na tabuli) ---
print("\n=== citlivost: co stoji prisnejsi pozadavek ===")
_, _, cena_glu = optimum(poz_glu=POZ_GLU + 10.0)
_, _, cena_obj = optimum(poz_obj=POZ_OBJ + 0.1)
print(f"glukoza {POZ_GLU:.0f} -> {POZ_GLU+10:.0f} g: cena {cena_opt:.2f} -> {cena_glu:.2f} Kc"
      f"  =>  {(cena_glu-cena_opt)/10:.2f} Kc/g")
print(f"objem   {POZ_OBJ:.1f} -> {POZ_OBJ+0.1:.1f} l: cena {cena_opt:.2f} -> {cena_obj:.2f} Kc"
      f"  =>  {(cena_obj-cena_opt)/0.1:.2f} Kc/l")

# --- meze, ve kterych optimum zustava v temze vrcholu ---
# vrchol pruseciku je optimalni, dokud smernice vrstevnice lezi mezi -1 a -0.25,
# tedy dokud pomer cen CENA_A / CENA_B lezi mezi GLU_A/GLU_B a 1
print("\n=== kdy optimum preskoci do jineho vrcholu ===")
b_dolni, b_horni = CENA_A * GLU_B / GLU_A, CENA_A / 1.0
print(f"pri cene A = {CENA_A:.0f} Kc/l zustava optimum v pruseciku, dokud"
      f" cena B lezi mezi {b_horni:.1f} a {b_dolni:.1f} Kc/l")
for cb in (15.0, 20.0, 50.0, 80.0, 95.0):
    a, b, c = optimum(cena_b=cb)
    print(f"  cena B = {cb:5.1f} Kc/l -> x* = ({a:.3f}; {b:.3f}), cena {c:6.2f} Kc")

# --- co stoji odchylka od optima (u LP neni dno ploche, je to roh) ---
print("\n=== odchylka od optima po hranici glukozy (objem roste, glukoza drzi) ===")
for t in (0.0, 0.1, 0.2, 0.5):
    a, b = x1_opt + t, x2_opt - t * GLU_A / GLU_B
    print(f"  +{t:.1f} l roztoku A: ({a:.3f}; {b:.3f})  cena {cena(a, b):6.2f} Kc"
          f"  ({(cena(a, b)/cena_opt - 1)*100:+.1f} %)  "
          f"{'pripustne' if pripustny(a, b) else 'NEPRIPUSTNE'}")
print("  opacny smer (min roztoku A) uz je nepripustny - objem klesne pod 1,5 l")

# --- dve rozbite varianty: presne to, na cem ma kontrola spadnout ---
print("\n=== rozbite varianty (vynechane omezeni) ===")
_, _, c_bez_obj = optimum(poz_obj=0.0)
_, _, c_bez_glu = optimum(poz_glu=0.0)
a1, b1, _ = optimum(poz_obj=0.0)
a2, b2, _ = optimum(poz_glu=0.0)
print(f"bez omezeni na objem : x = ({a1:.3f}; {b1:.3f}), cena {c_bez_obj:.2f} Kc"
      f" -> objem {a1+b1:.2f} l < {POZ_OBJ} l")
print(f"bez omezeni na glukozu: x = ({a2:.3f}; {b2:.3f}), cena {c_bez_glu:.2f} Kc"
      f" -> glukoza {GLU_A*a2+GLU_B*b2:.0f} g < {POZ_GLU:.0f} g")

# --- kontrola dosazenim zpet ---
assert GLU_A * x1_opt + GLU_B * x2_opt >= POZ_GLU - 1e-6, "poruseno omezeni na glukozu"
assert x1_opt + x2_opt >= POZ_OBJ - 1e-6, "poruseno omezeni na objem"
assert abs(cena_opt - 45.0) < 1e-6, "cena optima se rozesla s ocekavanim"
print("\nkontrola dosazenim zpet prosla")


# ---------------------------------------------------------------- obrazek ----
def cena_optima(poz_glu):
    """Cena optima jako funkce pozadavku na glukozu - po castech linearni."""
    return np.array([optimum(poz_glu=b)[2] for b in np.atleast_1d(poz_glu)])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.3))

# --- levy panel: pripustna oblast, vrstevnice ceny, optimum ---
X1_MAX, X2_MAX = 3.3, 1.75
ax1.add_patch(plt.Polygon(
    [(0, X2_MAX), (0, POZ_OBJ), (x1_pr, x2_pr), (POZ_GLU / GLU_A, 0),
     (X1_MAX, 0), (X1_MAX, X2_MAX)],
    facecolor="#dbeafe", edgecolor="none", zorder=0))
ax1.annotate("pripustna oblast\n(vse, co splni oba pozadavky)", (2.15, 1.15),
             ha="center", fontsize=11, color="#1e40af")

xs = np.linspace(-0.1, X1_MAX, 200)
ax1.plot(xs, (POZ_GLU - GLU_A * xs) / GLU_B, color="#b45309", lw=2.2,
         label=f"glukoza: $50x_1+200x_2={POZ_GLU:.0f}$")
ax1.plot(xs, POZ_OBJ - xs, color="#7c3aed", lw=2.2,
         label="objem: $x_1+x_2=1{,}5$")

for c, styl in ((30.0, ":"), (60.0, ":"), (75.0, ":")):
    ax1.plot(xs, (c - CENA_A * xs) / CENA_B, color="#6b7280", ls=styl, lw=1.1)
    ax1.annotate(f"{c:.0f} Kc", (0.06, c / CENA_B - 0.055), fontsize=9.5, color="#6b7280")
ax1.plot(xs, (cena_opt - CENA_A * xs) / CENA_B, color="#1a7f37", ls="--", lw=1.8,
         label=f"vrstevnice ceny {cena_opt:.0f} Kc")

def cislo(h):
    """1.0 -> "1", 0.5 -> "0,5" - kratke popisky vrcholu do obrazku."""
    return f"{h:.1f}".rstrip("0").rstrip(".").replace(".", ",") or "0"


for jmeno, (a, b) in vrcholy.items():
    ax1.plot(a, b, "o", ms=8, color="#111827", zorder=4)
    doprava = a < 2.0                             # u praveho vrcholu by popisek utekl z obrazku
    ax1.annotate(f"({cislo(a)}; {cislo(b)})\n{cena(a, b):.0f} Kc", (a, b),
                 xytext=(9 if doprava else -9, 9), textcoords="offset points",
                 ha="left" if doprava else "right", fontsize=10)
ax1.plot(x1_opt, x2_opt, "o", ms=13, mfc="none", mec="#1a7f37", mew=2.6, zorder=5)
ax1.annotate("optimum", (x1_opt, x2_opt), xytext=(-14, -30),
             textcoords="offset points", ha="right", fontsize=11.5, color="#1a7f37",
             arrowprops=dict(arrowstyle="->", color="#1a7f37"))

ax1.set_xlim(-0.05, X1_MAX)
ax1.set_ylim(-0.05, X2_MAX)
ax1.set_xlabel("roztok A (50 g/l, 20 Kc/l)  $x_1$ [l]")
ax1.set_ylabel("roztok B (200 g/l, 50 Kc/l)  $x_2$ [l]")
ax1.set_title("Optimum je roh pripustne oblasti")
ax1.grid(alpha=0.28)
ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

# --- pravy panel: cena optima jako funkce pozadavku na glukozu ---
bs = np.linspace(0, 400, 401)
ax2.plot(bs, cena_optima(bs), color="#1f4e79", lw=2.4)
for zlom in (GLU_A * POZ_OBJ, GLU_B * POZ_OBJ):
    ax2.plot(zlom, optimum(poz_glu=zlom)[2], "o", ms=7, color="#b45309", zorder=4)
ax2.plot(POZ_GLU, cena_opt, "o", ms=10, color="#1a7f37", zorder=5)
ax2.annotate(f"zadani: {POZ_GLU:.0f} g -> {cena_opt:.0f} Kc".replace(".", ","),
             (POZ_GLU, cena_opt), xytext=(14, -34), textcoords="offset points",
             fontsize=11, color="#1a7f37",
             arrowprops=dict(arrowstyle="->", color="#1a7f37"))
ax2.annotate("sklon 0,20 Kc/g\n= stinova cena glukozy", (250, optimum(poz_glu=250)[2]),
             xytext=(-6, 46), textcoords="offset points", ha="right", fontsize=10.5,
             color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax2.annotate("omezeni na glukozu\nje jeste neaktivni\n(sklon 0)", (75, 30),
             xytext=(0, 20), textcoords="offset points", ha="center",
             fontsize=10, color="#6b7280")
ax2.set_xlim(0, 400)
ax2.set_ylim(0, 105)
ax2.set_xlabel("pozadavek na glukozu [g]")
ax2.set_ylabel("cena optimalni smesi [Kc]")
ax2.set_title("Cena optima roste po castech linearne")
ax2.grid(alpha=0.28)

fig.tight_layout()
fig.savefig("cviceni/C2/obrazky/infuze.svg")
print("obrazek: cviceni/C2/obrazky/infuze.svg")
