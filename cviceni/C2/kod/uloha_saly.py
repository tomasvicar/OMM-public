"""Samostatná úloha cvičení 2 (LP): alokace operačních sálů.

Zrcadlový obraz tabulové úlohy z `tabule_infuze.py`: tam minimalizace
s omezeními typu >=, tady **maximalizace s omezeními typu <=**, takže je
počátek přípustný a přípustná oblast je omezená. Studenti ji řeší sami —
formulace, CVXPY, přípustná oblast, stínové ceny.

    proměnné   x1, x2 ... počet endoprotéz kyčle a kolene za týden
    maximize   62*x1 + 55*x2                  (marže v tis. Kč)
    subject to 2.0*x1 + 2.5*x2 <= 40          (sálové hodiny)
               5.0*x1 + 4.0*x2 <= 90          (lůžkodny)
               x1, x2          >= 0

Tenhle skript je JEDINÝ ZDROJ ČÍSEL pro zadání i pro vzorové řešení.
Ověřuje čtyři věci, na kterých úloha didakticky stojí:

  1. optimum leží v průsečíku obou omezení, obě jsou aktivní;
  2. stínové ceny vyjdou na celá čísla — 6 tis. Kč za sálovou hodinu
     a 10 tis. Kč za lůžkoden;
  3. nabídka "hodina sálu navíc za 8 tis. Kč" se má odmítnout, i když jedna
     endoprotéza nese marži 62 tis. Kč — na tomhle rozdílu úloha stojí;
  4. optimum LP není celočíselné a **zaokrouhlení nevede k celočíselnému
     optimu** — to leží v úplně jiném vrcholu.

Spuštění z kořene repozitáře:
    uv run python cviceni/C2/kod/uloha_saly.py
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

MARZE = np.array([62.0, 55.0])        # tis. Kč za výkon
A = np.array([[2.0, 2.5],             # sálové hodiny na výkon
              [5.0, 4.0]])            # lůžkodny na výkon
B = np.array([40.0, 90.0])            # týdenní kapacita
NAZVY_OMEZENI = ["sálové hodiny", "lůžkodny"]


def optimum(b=B):
    """Vyřeší LP a vrátí (x, marže, stínové ceny). Maximalizace = -min(-c)."""
    res = linprog(c=-MARZE, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
    return res.x, -res.fun, -res.ineqlin.marginals


x_opt, marze_opt, duals = optimum()
print("=== optimum ===")
print(f"x* = ({x_opt[0]:.4f}; {x_opt[1]:.4f}) vykonu za tyden")
print(f"marze = {marze_opt:.2f} tis. Kc")
for nazev, radek, kapacita in zip(NAZVY_OMEZENI, A, B):
    cerpani = radek @ x_opt
    stav = "aktivni" if abs(cerpani - kapacita) < 1e-6 else "neaktivni"
    print(f"  {nazev:14s} {cerpani:6.2f} / {kapacita:5.1f}  ({stav})")

print("\n=== vrcholy pripustne oblasti (kandidati na optimum) ===")
rovnice = [(A[0], B[0], NAZVY_OMEZENI[0]),
           (A[1], B[1], NAZVY_OMEZENI[1]),
           (np.array([1.0, 0.0]), 0.0, "x1 = 0"),
           (np.array([0.0, 1.0]), 0.0, "x2 = 0")]
for i, j in itertools.combinations(range(4), 2):
    M = np.array([rovnice[i][0], rovnice[j][0]])
    if abs(np.linalg.det(M)) < 1e-9:
        continue
    x = np.linalg.solve(M, np.array([rovnice[i][1], rovnice[j][1]]))
    pripustny = (x >= -1e-9).all() and (A @ x <= B + 1e-9).all()
    popis = f"{rovnice[i][2]} x {rovnice[j][2]}"
    hodnota = f"{MARZE @ x:8.2f} tis. Kc" if pripustny else "        -        "
    print(f"  {popis:32s} ({x[0]:7.4f}; {x[1]:7.4f}) {hodnota}"
          f"  {'pripustny' if pripustny else 'NEpripustny'}")

print("\n=== stinove ceny ===")
for nazev, y in zip(NAZVY_OMEZENI, duals):
    print(f"  {nazev:14s} {y:6.2f} tis. Kc za jednotku")
NABIDKA = 8.0
print(f"\nnabidka: hodina salu navic za {NABIDKA:.0f} tis. Kc")
print(f"  hodina navic prinese {duals[0]:.0f} tis. Kc -> "
      f"{'VZIT' if duals[0] > NABIDKA else 'ODMITNOUT'}"
      f" (jeden vykon nese {MARZE[0]:.0f} tis. Kc, ale kapacitu drzi i luzka)")

print("\n=== dokdy stinova cena sálové hodiny plati ===")
for b1 in [34.0, 36.0, 40.0, 50.0, 56.0, 56.25, 58.0]:
    x, marze, y = optimum(np.array([b1, B[1]]))
    print(f"  b1 = {b1:6.2f} h: x = ({x[0]:7.4f}; {x[1]:7.4f})  "
          f"marze = {marze:7.2f}  stinove ceny = ({y[0]:5.2f}; {y[1]:5.2f})")
print("  -> mimo interval <36; 56,25> h se meni baze a stinova cena prestava platit")

print("\n=== celociselna varianta ===")
nej = max(((MARZE @ np.array([x1, x2]), x1, x2)
           for x1 in range(0, 21) for x2 in range(0, 17)
           if (A @ np.array([x1, x2]) <= B + 1e-9).all()))
print(f"  LP optimum          ({x_opt[0]:.4f}; {x_opt[1]:.4f})  {marze_opt:7.2f} tis. Kc")
for kandidat in ([int(np.floor(x_opt[0])), int(np.floor(x_opt[1]))],
                 [int(np.ceil(x_opt[0])), int(np.ceil(x_opt[1]))]):
    x = np.array(kandidat, dtype=float)
    ok = (A @ x <= B + 1e-9).all()
    hodnota = f"{MARZE @ x:7.2f} tis. Kc" if ok else "   nepripustne  "
    print(f"  zaokrouhleni {str(kandidat):8s} {hodnota}  {'' if ok else '<- porusuje kapacitu'}")
print(f"  skutecne optimum    ({nej[1]}; {nej[2]})           {nej[0]:7.2f} tis. Kc"
      f"  <- jiny vrchol, ne zaokrouhlene LP")


# ---------------------------------------------------------------- obrázek ----
# Levý panel je obvyklá přípustná oblast s vrstevnicí marže; pravý panel je ta
# didakticky cennější půlka: celočíselná mřížka, na které je vidět, že
# zaokrouhlení LP optima nedá celočíselné optimum.
ZELENA, ORANZOVA, CERVENA, SEDA = "#1a7f37", "#b45309", "#dc2626", "#6b7280"
X1_MAX, X2_MAX = 22.5, 18.5
FIALOVA, MODRA = "#7c3aed", "#1e40af"


def cislo(h):
    """14.4444 -> "14,4"; 4.0 -> "4" - kratke popisky do obrazku."""
    return f"{h:.1f}".rstrip("0").rstrip(".").replace(".", ",") or "0"


def hranice(x1):
    """Horni obalka pripustne oblasti: min pres obe omezeni."""
    return np.minimum((B[0] - A[0, 0] * x1) / A[0, 1], (B[1] - A[1, 0] * x1) / A[1, 1])


x_cele = np.array([nej[1], nej[2]], dtype=float)          # celociselne optimum
x_dolu = np.floor(x_opt)                                  # zaokrouhleni dolu
x_nahoru = np.ceil(x_opt)                                 # zaokrouhleni nahoru (nepripustne)
vrcholy_obl = [np.array([0.0, 0.0]),
               np.array([B[1] / A[1, 0], 0.0]),           # jen kycle
               x_opt,                                     # prusecik obou omezeni
               np.array([0.0, B[0] / A[0, 1]])]           # jen kolena

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.8))

# --- levý panel: přípustná oblast, vrstevnice marže, hodnoty ve vrcholech ---
for ax in (ax1, ax2):
    ax.add_patch(plt.Polygon([tuple(v) for v in vrcholy_obl],
                             facecolor="#dbeafe", edgecolor="none", zorder=0))
    ax.set_xlabel("endoprotezy kycle za tyden  $x_1$")
    ax.set_ylabel("endoprotezy kolene za tyden  $x_2$")
    ax.grid(alpha=0.28)
ax1.set_xlim(-0.4, X1_MAX)
ax1.set_ylim(-0.7, X2_MAX)
ax2.set_xlim(7.0, 21.0)          # detail rohu, aby se ctyri kandidati nepreklryvali
ax2.set_ylim(-0.9, 11.5)

xs = np.linspace(-0.4, X1_MAX, 300)
ax1.plot(xs, (B[0] - A[0, 0] * xs) / A[0, 1], color=FIALOVA, lw=2.2,
         label="salove hodiny: $2x_1+2{,}5x_2=40$")
ax1.plot(xs, (B[1] - A[1, 0] * xs) / A[1, 1], color=ORANZOVA, lw=2.2,
         label="luzkodny: $5x_1+4x_2=90$")
for hodnota in (400.0, 800.0):
    ax1.plot(xs, (hodnota - MARZE[0] * xs) / MARZE[1], color=SEDA, ls=":", lw=1.1)
    ax1.annotate(f"{hodnota:.0f}", (0.25, hodnota / MARZE[1] - 0.75),
                 fontsize=9.5, color=SEDA)
ax1.plot(xs, (marze_opt - MARZE[0] * xs) / MARZE[1], color=ZELENA, ls="--", lw=1.9,
         label=f"vrstevnice marze {marze_opt:.0f} tis. Kc")
ax1.annotate("pripustna oblast", (5.2, 3.4), fontsize=11, color=MODRA)

for v in vrcholy_obl:
    ax1.plot(v[0], v[1], "o", ms=8, color="#111827", zorder=4)
posun = {0: (10, 6), 1: (9, 10), 2: (12, 18), 3: (12, -2)}
for i, v in enumerate(vrcholy_obl):
    ax1.annotate(f"({cislo(v[0])}; {cislo(v[1])})\n{MARZE @ v:.0f} tis. Kc", tuple(v),
                 xytext=posun[i], textcoords="offset points", fontsize=10.5,
                 ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
ax1.plot(x_opt[0], x_opt[1], "o", ms=15, mfc="none", mec=ZELENA, mew=2.6, zorder=5)
ax1.annotate("optimum", (x_opt[0], x_opt[1]), xytext=(-14, -26),
             textcoords="offset points", ha="right", fontsize=11.5, color=ZELENA,
             arrowprops=dict(arrowstyle="->", color=ZELENA))
ax1.set_title(f"Optimum lezi ve vrcholu: marze {marze_opt:.0f} tis. Kc")
ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

# --- pravý panel: celočíselná mřížka a čtyři kandidáti ---
mrizka = np.array([(i, j) for i in range(0, int(B[1] / A[1, 0]) + 1)
                   for j in range(0, int(B[0] / A[0, 1]) + 1)
                   if (A @ np.array([i, j], dtype=float) <= B + 1e-9).all()])
ax2.plot(xs, hranice(xs), color=SEDA, lw=1.4, zorder=1)
ax2.plot(mrizka[:, 0], mrizka[:, 1], "o", ms=3.4, color="#64748b", zorder=2,
         label="celociselne pripustne body")
ax2.plot(x_opt[0], x_opt[1], "o", ms=14, mfc="none", mec=ZELENA, mew=2.6, zorder=6,
         label=f"LP optimum ({cislo(x_opt[0])}; {cislo(x_opt[1])}) = {marze_opt:.0f} tis. Kc")
ax2.plot(x_dolu[0], x_dolu[1], "o", ms=11, color=ORANZOVA, zorder=6,
         label=f"zaokrouhleni dolu ({cislo(x_dolu[0])}; {cislo(x_dolu[1])})"
               f" = {MARZE @ x_dolu:.0f} tis. Kc"
               f" ({MARZE @ x_dolu - marze_opt:.0f})")
ax2.plot(x_nahoru[0], x_nahoru[1], "X", ms=13, color=CERVENA, zorder=6,
         label=f"zaokrouhleni nahoru ({cislo(x_nahoru[0])}; {cislo(x_nahoru[1])})"
               f" = NEPRIPUSTNE")
ax2.plot(x_cele[0], x_cele[1], "*", ms=20, color="#111827", zorder=7,
         label=f"celociselne optimum ({cislo(x_cele[0])}; {cislo(x_cele[1])})"
               f" = {MARZE @ x_cele:.0f} tis. Kc")

ax2.annotate("", xy=(x_cele[0], x_cele[1] + 0.55), xytext=(x_opt[0], x_opt[1] - 0.55),
             arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6", lw=2.0,
                             ls=(0, (4, 3)), color="#111827",
                             connectionstyle="arc3,rad=-0.35", shrinkA=8, shrinkB=10))
ax2.annotate("celociselne optimum lezi\nv UPLNE JINEM vrcholu\n"
             f"a je o {MARZE @ x_cele - MARZE @ x_dolu:.0f} tis. Kc lepsi\n"
             "nez zaokrouhleni dolu", (10.6, 1.9), ha="left", va="center",
             fontsize=10.5, color="#111827",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1d5db", alpha=0.95))
ax2.annotate(f"{A[0] @ x_nahoru:.1f} h > {B[0]:.0f} h".replace(".", ","),
             (x_nahoru[0], x_nahoru[1]), xytext=(14, 6), textcoords="offset points",
             ha="left", fontsize=10, color=CERVENA)
ax2.set_title("Zaokrouhlene LP neni celociselne optimum (detail rohu oblasti)")
ax2.legend(loc="upper right", fontsize=9, framealpha=0.96)

fig.tight_layout()
fig.savefig("cviceni/C2/obrazky/saly.svg")
print("\nobrazek: cviceni/C2/obrazky/saly.svg")
