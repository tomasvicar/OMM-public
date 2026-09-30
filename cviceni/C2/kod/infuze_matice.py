"""Rozšíření tabulové úlohy na matici: parenterální výživa, 6 přípravků × 4 živiny.

Formulace je **totožná** s úlohou ze dvou roztoků (`tabule_infuze.py`) — cena
jako skalární součin, živiny jako maticová nerovnost. Mění se jediné: rozměry
polí. Přípustná oblast se přestane dát nakreslit, kód zůstane na řádek stejný.

    proměnné   x ... objem každého přípravku v litrech (6 čísel)
    minimize   c^T x                          (cena v Kč)
    subject to A x >= b  (energie, bílkoviny, draslík)
               A x <= b  (horní mez draslíku)
               objem == 2.0 l
               x >= 0

Kromě optima skript ukazuje tři věci, kvůli kterým je v cvičení zařazený:

  1. **neaktivní omezení má nulový multiplikátor** (horní mez draslíku
     a strop lipidů) — komplementární volnost tři týdny před KKT;
  2. **CVXPY a `scipy.optimize.linprog` dají totéž**, jen linprog vyžaduje
     převod na standardní tvar (omezení >= se násobí -1);
  3. **model bez omezení na minimální podíl tuků vrátí roztok bez lipidů** —
     o 88 Kč levnější a klinicky nepoužitelný. Chybějící omezení tady není
     umělé, je to tatáž chyba, na které stojí červený praporek.

Spuštění z kořene repozitáře:
    uv run python cviceni/C2/kod/infuze_matice.py
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

NAZVY = ["Glc 10%", "Glc 40%", "AK 10%", "Lipid 20%", "KCl 7,45%", "voda"]
CENA = np.array([40.0, 120.0, 350.0, 500.0, 30.0, 15.0])      # Kč za litr

# řádky: energie [kcal/l], bílkoviny [g/l], draslík [mmol/l], objem [l/l]
A = np.array([
    [400.0, 1600.0, 400.0, 2000.0,    0.0, 0.0],
    [  0.0,    0.0, 100.0,    0.0,    0.0, 0.0],
    [  0.0,    0.0,   0.0,    0.0, 1000.0, 0.0],
    [  1.0,    1.0,   1.0,    1.0,    1.0, 1.0],
])
ENERGIE, BILKOVINY, DRASLIK_MIN, DRASLIK_MAX, OBJEM = 1800.0, 70.0, 60.0, 100.0, 2.0
LIPID_MAX = 0.5      # l, technologický strop tukové emulze


def vyres(lipid_min=0.0):
    """Vrátí (x, cena, duály) pro danou dolní mez objemu tukové emulze."""
    x = cp.Variable(6, nonneg=True)
    omezeni = {
        "energie >= 1800 kcal": A[0] @ x >= ENERGIE,
        "bilkoviny >= 70 g": A[1] @ x >= BILKOVINY,
        "draslik >= 60 mmol": A[2] @ x >= DRASLIK_MIN,
        "draslik <= 100 mmol": A[2] @ x <= DRASLIK_MAX,
        "objem == 2 l": A[3] @ x == OBJEM,
        "lipid <= 0,5 l": x[3] <= LIPID_MAX,
    }
    if lipid_min > 0:
        omezeni["lipid >= 0,25 l"] = x[3] >= lipid_min
    uloha = cp.Problem(cp.Minimize(CENA @ x), list(omezeni.values()))
    uloha.solve()
    duals = {k: float(np.atleast_1d(v.dual_value)[0]) for k, v in omezeni.items()}
    return x.value, uloha.value, duals


x_opt, cena_opt, duals = vyres()
print("=== optimum (model tak, jak ho student napise poprve) ===")
for nazev, objem in zip(NAZVY, x_opt):
    print(f"  {nazev:10s} {objem:7.4f} l")
print(f"cena = {cena_opt:.2f} Kc")
print(f"  energie   {A[0] @ x_opt:8.1f} kcal (pozadavek {ENERGIE})")
print(f"  bilkoviny {A[1] @ x_opt:8.1f} g    (pozadavek {BILKOVINY})")
print(f"  draslik   {A[2] @ x_opt:8.1f} mmol (pozadavek {DRASLIK_MIN}-{DRASLIK_MAX})")
print(f"  objem     {A[3] @ x_opt:8.2f} l    (pozadavek {OBJEM})")

print("\n=== stinove ceny (nula = omezeni neni aktivni) ===")
for nazev, y in duals.items():
    print(f"  {nazev:22s} {y:9.4f}")
print("  pozor: u rovnosti vraci CVXPY dual s opacnym znamenkem nez u nerovnosti;")
print("  velikost 13,33 Kc/l plati, znamenko je konvence knihovny")

print("\n=== tataz uloha pres scipy.optimize.linprog (standardni tvar) ===")
res = linprog(
    c=CENA,
    A_ub=np.vstack([-A[0], -A[1], -A[2], A[2], np.eye(6)[3]]),
    b_ub=np.array([-ENERGIE, -BILKOVINY, -DRASLIK_MIN, DRASLIK_MAX, LIPID_MAX]),
    A_eq=A[3][None, :], b_eq=[OBJEM],
    bounds=[(0, None)] * 6,
)
print(f"  x = {np.round(res.x, 4)}")
print(f"  cena = {res.fun:.2f} Kc, status = {res.status} ({res.message.strip()})")
print(f"  rozdil proti CVXPY: {abs(res.fun - cena_opt):.2e} Kc")

print("\n=== co dela chybejici omezeni ===")
x_tuk, cena_tuk, _ = vyres(lipid_min=0.25)
print(f"  bez minimalniho podilu tuku: {cena_opt:.2f} Kc, lipid {x_opt[3]:.3f} l")
print(f"  s omezenim lipid >= 0,25 l : {cena_tuk:.2f} Kc, lipid {x_tuk[3]:.3f} l")
print(f"  levnejsi model je o {cena_tuk - cena_opt:.2f} Kc levnejsi a klinicky nepouzitelny")


# ---------------------------------------------------------------- obrázek ----
# Obrázek pro blok 4 a pro červený praporek: co s vakem udělá vynechané omezení
# na minimální podíl tuků. Levnější řešení srazí lipidovou emulzi přesně na
# nulu — a to je na skupinovém sloupcovém grafu vidět dřív, než se čtou čísla.
CERVENA, MODRA = "#dc2626", "#1f4e79"


def kc(h, mist=2):
    """0.3867 -> "0,39" - cisla do popisku s desetinnou carkou."""
    return f"{h:.{mist}f}".replace(".", ",")


fig, ax = plt.subplots(figsize=(11.0, 5.8))
poz = np.arange(len(NAZVY))
sirka = 0.38
ax.bar(poz - sirka / 2, x_opt, sirka, color=CERVENA, alpha=0.9,
       label=f"bez omezeni na tuky  ({kc(cena_opt)} Kc)")
ax.bar(poz + sirka / 2, x_tuk, sirka, color=MODRA, alpha=0.9,
       label=f"s omezenim lipid >= 0,25 l  ({kc(cena_tuk)} Kc)")

for p, (v_bez, v_s) in enumerate(zip(x_opt, x_tuk)):
    if p != 3:                                    # u lipidu mluvi cervena anotace
        ax.annotate(kc(v_bez), (p - sirka / 2, v_bez), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color="#374151")
    ax.annotate(kc(v_s), (p + sirka / 2, v_s), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9.5, color="#374151")

ax.annotate("levnejsi vak neobsahuje ZADNY tuk (0,00 l)", (3 - sirka / 2, 0.012),
            xytext=(3 - sirka / 2, 0.88), ha="center", fontsize=11.5, color=CERVENA,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=CERVENA, lw=1.9, shrinkB=2))

ax.set_xticks(poz)
ax.set_xticklabels(NAZVY, fontsize=11)
ax.set_ylim(0, 1.2)
ax.set_ylabel("objem ve dvoulitrovem vaku [l]")
ax.set_xlabel("pripravek")
ax.grid(axis="y", alpha=0.28)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
fig.suptitle("Chybejici omezeni na minimalni podil tuku srazi lipidovou emulzi na nulu",
             fontsize=13.5, y=0.975)
ax.set_title(f"rozdil {kc(cena_tuk - cena_opt)} Kc je cena za klinicky pouzitelny vak"
             " - solver hlasi optimal v obou pripadech", fontsize=11, color="#374151")

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("cviceni/C2/obrazky/matice.svg")
print("\nobrazek: cviceni/C2/obrazky/matice.svg")
