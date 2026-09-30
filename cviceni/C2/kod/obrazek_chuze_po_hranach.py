"""Obrázek k bloku 1 cvičení 2: co dělá simplex — chůze po hranách.

Na tabuli se optimum hledá **průchodem všech vrcholů**: sepíšou se všechny
dvojice omezení, vyřeší se soustavy 2×2 a vybere se nejlevnější přípustný
vrchol. Simplex dělá totéž, ale nechodí po všech: začne v jednom vrcholu a
posouvá se **po hranách** vždy do souseda, ve kterém si polepší. Jakmile
žádný soused není lepší, končí — a tím je dokázané, že je v optimu.

Obrázek to ukazuje na téže infuzní úloze jako `tabule_infuze.py`:

    minimize   20*x1 + 50*x2                  (cena v Kč)
    subject to 50*x1 + 200*x2 >= 150          (glukóza, g)
               x1 + x2        >= 1.5          (objem, l)
               x1, x2         >= 0

    start (0; 1,5) za 75 Kč  --hrana objemu-->  (1; 0,5) za 45 Kč
    kontrola souseda (3; 0) za 60 Kč  ->  horší, tedy konec

Čísla se nepíšou ručně, počítají se ze zadání stejně jako v `tabule_infuze.py`,
ať se obrázek nerozejde s tabulí. Barvy a styl jsou schválně shodné s
`obrazky/infuze.svg`, aby oba obrázky vedle sebe ladily.

Spuštění z kořene repozitáře:
    uv run python cviceni/C2/kod/obrazek_chuze_po_hranach.py
"""

import matplotlib.pyplot as plt
plt.rcParams["svg.hashsalt"] = "chuze-po-hranach"   # stejná ID při každém běhu
import numpy as np

CENA_A, CENA_B = 20.0, 50.0                       # Kc za litr roztoku A a B
GLU_A, GLU_B = 50.0, 200.0                        # g glukozy v litru roztoku
POZ_GLU = 150.0                                   # pozadavek na glukozu [g]
POZ_OBJ = 1.5                                     # pozadavek na objem [l]


def cena(x1, x2):
    return CENA_A * x1 + CENA_B * x2


def cislo(h):
    """1.0 -> "1", 0.5 -> "0,5" - kratke popisky vrcholu do obrazku."""
    return f"{h:.2f}".rstrip("0").rstrip(".").replace(".", ",") or "0"


# --- tri vrcholy pripustne oblasti (tytez jako v tabule_infuze.py) ---
x1_pr = (GLU_B * POZ_OBJ - POZ_GLU) / (GLU_B - GLU_A)     # prusecik obou omezeni
x2_pr = POZ_OBJ - x1_pr
START = (0.0, POZ_OBJ)                 # jen roztok B
OPT = (x1_pr, x2_pr)                   # prusecik obou omezeni
SOUSED = (POZ_GLU / GLU_A, 0.0)        # jen roztok A

print("=== chuze po hranach (to, co dela simplex) ===")
print(f"start   ({cislo(START[0])}; {cislo(START[1])}) l  cena {cena(*START):.0f} Kc")
print(f"krok 1  ({cislo(OPT[0])}; {cislo(OPT[1])}) l  cena {cena(*OPT):.0f} Kc"
      f"  (po hrane omezeni na objem, polepseni o {cena(*START)-cena(*OPT):.0f} Kc)")
print(f"soused  ({cislo(SOUSED[0])}; {cislo(SOUSED[1])}) l  cena {cena(*SOUSED):.0f} Kc"
      f"  -> horsi, tedy konec")

# ---------------------------------------------------------------- obrazek ----
ZELENA, SEDA, MODRA = "#1a7f37", "#9ca3af", "#1e40af"
X1_MAX, X2_MAX = 3.65, 1.95

fig, ax = plt.subplots(figsize=(8.6, 5.6))

ax.add_patch(plt.Polygon(
    [(0, X2_MAX), START, OPT, SOUSED, (X1_MAX, 0), (X1_MAX, X2_MAX)],
    facecolor="#dbeafe", edgecolor="none", zorder=0))
ax.annotate("přípustná oblast", (2.9, 1.22), ha="center", fontsize=11, color=MODRA)

xs = np.linspace(-0.1, X1_MAX, 200)
ax.plot(xs, (POZ_GLU - GLU_A * xs) / GLU_B, color="#b45309", lw=2.2,
        label=f"glukóza: $50x_1+200x_2={POZ_GLU:.0f}$")
ax.plot(xs, POZ_OBJ - xs, color="#7c3aed", lw=2.2,
        label="objem: $x_1+x_2=1{,}5$")

# --- krok 1: po hrane omezeni na objem ze startu do optima ---
ax.annotate("", xy=OPT, xytext=START,
            arrowprops=dict(arrowstyle="-|>,head_width=0.34,head_length=0.7",
                            lw=3.0, color=ZELENA, shrinkA=13, shrinkB=15))
ax.annotate("krok 1: po hraně\n75 → 45 Kč, polepším si", (0.5, 1.0),
            xytext=(24, 26), textcoords="offset points", ha="left", fontsize=11,
            color=ZELENA, arrowprops=dict(arrowstyle="-", color=ZELENA, lw=0.9))

# --- kontrola druheho souseda: horsi, tedy konec ---
ax.annotate("", xy=SOUSED, xytext=OPT,
            arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6",
                            lw=2.0, ls=(0, (4, 3)), color=SEDA,
                            shrinkA=15, shrinkB=15))
ax.annotate("kontrola souseda:\n60 Kč > 45 Kč, tedy horší", (2.0, 0.25),
            xytext=(0, 34), textcoords="offset points", ha="center", fontsize=11,
            color="#6b7280", arrowprops=dict(arrowstyle="-", color=SEDA, lw=0.9))

# --- vrcholy: start, optimum a zamitnuty soused ---
ax.plot(*START, "o", ms=9, color="#111827", zorder=4)
ax.annotate(f"START  ({cislo(START[0])}; {cislo(START[1])})\n{cena(*START):.0f} Kč",
            START, xytext=(12, 6), textcoords="offset points", ha="left",
            fontsize=11, fontweight="bold")

ax.plot(*OPT, "o", ms=9, color="#111827", zorder=4)
ax.plot(*OPT, "o", ms=15, mfc="none", mec=ZELENA, mew=2.6, zorder=5)
ax.annotate(f"OPTIMUM  ({cislo(OPT[0])}; {cislo(OPT[1])})\n{cena(*OPT):.0f} Kč", OPT,
            xytext=(-18, -30), textcoords="offset points", ha="right", fontsize=11,
            fontweight="bold", color=ZELENA,
            arrowprops=dict(arrowstyle="->", color=ZELENA))

ax.plot(*SOUSED, "o", ms=9, color=SEDA, zorder=4)
ax.plot(*SOUSED, "x", ms=13, color="#ef4444", mew=2.4, zorder=6)
ax.annotate(f"({cislo(SOUSED[0])}; {cislo(SOUSED[1])})  {cena(*SOUSED):.0f} Kč\n"
            "zamítnuto", SOUSED,
            xytext=(0, 20), textcoords="offset points", ha="center",
            fontsize=10.5, color="#6b7280")

ax.set_xlim(-0.05, X1_MAX)
ax.set_ylim(-0.12, X2_MAX)
ax.set_xlabel("roztok A (50 g/l, 20 Kč/l)  $x_1$ [l]")
ax.set_ylabel("roztok B (200 g/l, 50 Kč/l)  $x_2$ [l]")
ax.set_title("Simplex neprohledává všechny vrcholy: jde po hranách\n"
             "a v každém kroku si polepší", fontsize=13)
ax.grid(alpha=0.28)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

fig.tight_layout()
fig.savefig("cviceni/C2/obrazky/chuze_po_hranach.svg", metadata={"Date": None})
print("obrazek: cviceni/C2/obrazky/chuze_po_hranach.svg")
