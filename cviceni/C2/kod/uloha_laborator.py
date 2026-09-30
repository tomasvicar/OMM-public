"""Bonusová úloha cvičení 2 (LP): rozvržení vzorků mezi analyzátory v laboratoři.

Nesbírá se a nehodnotí se — je pro ty, kdo skončí samostatnou úlohu dřív,
a pro každého, kdo si chce po cvičení zkusit ještě jednu formulaci. Na hodinu
se nevejde.

Proti tabulové úloze i proti sálům je tahle **maticová**: tři analyzátory ×
čtyři typy vyšetření = **dvanáct proměnných**, a tedy dvanáctirozměrná
přípustná oblast, kterou nejde nakreslit. Tím navazuje na blok 4 (parenterální
výživa): formulace se nemění, jen tvary polí — a obrázek zmizí.

    proměnné   X[i,j] ... kolik vzorků typu j se za den udělá na analyzátoru i
    minimize   sum_ij CENA[i,j] * X[i,j]              (reagencie, Kč/den)
    subject to sum_j CAS[i,j] * X[i,j] <= KAPACITA[i] (provozní minuty stroje)
               sum_i X[i,j]            >= POPTAVKA[j] (došlé vzorky se musí udělat)
               X[0,0] <= 500, X[1,1] <= 150           (denní zásoba dvou reagencií)
               X[i,j] = 0 tam, kde analyzátor vyšetření neumí
               X >= 0

**Ne každý analyzátor umí každé vyšetření** — to je na úloze to zajímavé:
velký biochemický analyzátor A neumí imunochemii (troponin, TSH), malý záložní
C neumí TSH. Matice `UMI` proto není samá jednička a čtyři z dvanácti
proměnných jsou pevně nulové.

Úloha je postavená tak, aby stínové ceny odpověděly na dvě provozní otázky
a aby vyšly na celá čísla:

  1. **Který analyzátor má smysl posílit?** V optimu je plně vytížený jenom
     analyzátor B (320/320 min); A i C mají volno. Hodina navíc na B ušetří
     360 Kč za den, hodina navíc na A nebo C nic — a to i přesto, že A je
     vytížený na 400 ze 420 minut.
  2. **Kterou reagencii doobjednat?** Obě zásoby jsou v optimu vyčerpané, ale
     nestejně cenné: kazeta na troponin pro analyzátor B ušetří 24 Kč, kazeta
     na základní biochemii pro analyzátor A jen 10 Kč.

Poptávka je zapsaná jako `>=` (musí se udělat aspoň všechny došlé vzorky).
V optimu je splněná s rovností — víc vzorků, než jich přišlo, dělat nikdo
nebude, protože to jen stojí reagencie. Nerovnost je tu proto, že u ní vyjde
duál s kladným znaménkem; u rovnosti vrací CVXPY opačné, viz `infuze_matice.py`.

Optimum a duály přepočítává nezávisle ještě `scipy.optimize.linprog`; rozdíl
obou cest skript vytiskne.

Spuštění z kořene repozitáře:
    uv run python cviceni/C2/kod/uloha_laborator.py
"""

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

ANALYZATORY = ["A velky biochem.", "B integrovany", "C zalozni maly"]
VYSETRENI = ["biochemie", "troponin", "TSH", "CRP"]

# které vyšetření který analyzátor vůbec umí (A neumí imunochemii, C neumí TSH)
UMI = np.array([
    [True,  False, False, True],
    [True,  True,  True,  True],
    [True,  True,  False, True],
])

# strojový čas na jeden vzorek [min]; nuly jsou tam, kde se stroj nepoužije
CAS = np.array([
    [0.5, 0.0, 0.0, 0.5],
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 0.0, 2.0],
])

# cena reagencií na jeden vzorek [Kč]
CENA = np.array([
    [10.0,   0.0,  0.0, 20.0],
    [14.0,  80.0, 60.0, 26.0],
    [20.0, 110.0,  0.0, 34.0],
])

KAPACITA = np.array([420.0, 320.0, 300.0])      # provozních minut za den
POPTAVKA = np.array([600.0, 180.0, 120.0, 300.0])   # došlých vzorků za den

# denní zásoba dvou reagencií, které se objednávají po kazetách
ZASOBA = {(0, 0): 500.0, (1, 1): 150.0}         # (analyzátor, vyšetření) -> vzorků
POPIS_ZASOBY = {(0, 0): "kazety biochemie pro A", (1, 1): "kazety troponin pro B"}


def vyres(kapacita=KAPACITA, zasoba=None):
    """Vyřeší LP v CVXPY. Vrátí (X, cena, status, duály kapacit, duály poptávky,
    duály zásob)."""
    zasoba = ZASOBA if zasoba is None else zasoba
    X = cp.Variable((3, 4), nonneg=True)
    om_kapacita = [cp.sum(cp.multiply(CAS[i], X[i])) <= kapacita[i] for i in range(3)]
    om_poptavka = [cp.sum(X[:, j]) >= POPTAVKA[j] for j in range(4)]
    om_zasoba = [X[i, j] <= mez for (i, j), mez in zasoba.items()]
    uloha = cp.Problem(cp.Minimize(cp.sum(cp.multiply(CENA, X))),
                       [X[~UMI] == 0] + om_kapacita + om_poptavka + om_zasoba)
    uloha.solve()
    dual = lambda om: np.array([float(np.atleast_1d(o.dual_value)[0]) for o in om])
    return (X.value, uloha.value, uloha.status,
            dual(om_kapacita), dual(om_poptavka), dual(om_zasoba))


def linprog_kontrola(kapacita=KAPACITA):
    """Táž úloha ve standardním tvaru scipy (A_ub x <= b_ub). Nezávislý přepočet."""
    radky, prava = [], []
    for i in range(3):                                   # kapacita: <= rovnou
        r = np.zeros((3, 4)); r[i] = CAS[i]
        radky.append(r.ravel()); prava.append(kapacita[i])
    for j in range(4):                                   # poptávka: >= se násobí -1
        r = np.zeros((3, 4)); r[:, j] = -1.0
        radky.append(r.ravel()); prava.append(-POPTAVKA[j])
    for (i, j), mez in ZASOBA.items():
        r = np.zeros((3, 4)); r[i, j] = 1.0
        radky.append(r.ravel()); prava.append(mez)
    # co analyzátor neumí, má horní mez nula
    meze = [(0.0, None) if umi else (0.0, 0.0) for umi in UMI.ravel()]
    res = linprog(c=CENA.ravel(), A_ub=np.array(radky), b_ub=np.array(prava),
                  bounds=meze)
    return res.x.reshape(3, 4), res.fun, res.status, -res.ineqlin.marginals


def tabulka(matice, format_bunky, hlavicka):
    """Vytiskne 3x4 tabulku analyzátor x vyšetření; kde stroj neumí, je pomlčka."""
    print(f"  {hlavicka:18s}" + "".join(f"{v:>12s}" for v in VYSETRENI))
    for i, nazev in enumerate(ANALYZATORY):
        bunky = [format_bunky.format(matice[i, j]) if UMI[i, j] else "-"
                 for j in range(4)]
        print(f"  {nazev:18s}" + "".join(f"{b:>12s}" for b in bunky))


X_opt, cena_opt, status, y_kap, y_pop, y_zas = vyres()

print("=== zadani ===")
tabulka(CAS, "{:.1f} min", "cas na vzorek")
print()
tabulka(CENA, "{:.0f} Kc", "cena reagencii")
print()
print("  kapacita za den   " + "".join(f"{ANALYZATORY[i].split()[0]}: {KAPACITA[i]:.0f} min   "
                                       for i in range(3)))
print("  poptavka za den   " + "".join(f"{VYSETRENI[j]}: {POPTAVKA[j]:.0f}   "
                                       for j in range(4)))
print("  zasoba reagencii  " + "".join(f"{POPIS_ZASOBY[k]}: {v:.0f} vzorku   "
                                       for k, v in ZASOBA.items()))

print("\n=== optimum ===")
print(f"status = {status}")
tabulka(X_opt, "{:.0f}", "vzorku za den")
print(f"\ncelkova cena reagencii = {cena_opt:.2f} Kc za den")

print("\n=== vytizeni analyzatoru ===")
for i, nazev in enumerate(ANALYZATORY):
    minut = CAS[i] @ X_opt[i]
    stav = "PLNE VYTIZEN" if abs(minut - KAPACITA[i]) < 1e-6 else "volno"
    print(f"  {nazev:18s} {minut:6.1f} / {KAPACITA[i]:5.0f} min "
          f" ({100*minut/KAPACITA[i]:5.1f} %)  {stav}")

print("\n=== stinove ceny ===")
print("  kapacita analyzatoru [Kc za minutu / za hodinu]")
for i, nazev in enumerate(ANALYZATORY):
    print(f"    {nazev:18s} {y_kap[i]:7.2f} Kc/min   {60*y_kap[i]:8.2f} Kc/h")
print("  poptavka [Kc za jeden vzorek navic]")
for j, nazev in enumerate(VYSETRENI):
    print(f"    {nazev:18s} {y_pop[j]:7.2f} Kc/vzorek")
print("  zasoba reagencii [Kc za jednu kazetu navic]")
for k, y in zip(ZASOBA, y_zas):
    print(f"    {POPIS_ZASOBY[k]:18s} {y:7.2f} Kc/kazeta")

print("\n=== interpretace ===")
nej_kap = int(np.argmax(y_kap))
zbytek = [i for i in range(3) if i != nej_kap]
print(f"Hodina navic na analyzatoru {ANALYZATORY[nej_kap].split()[0]} usetri"
      f" {60*y_kap[nej_kap]:.0f} Kc za den: kazda uvolnena minuta prevezme jeden"
      f" vzorek biochemie z drazsiho stroje C.")
print(f"Na analyzatorech {' a '.join(ANALYZATORY[i].split()[0] for i in zbytek)}"
      f" neusetri hodina navic nic - a to i u A, ktery je vytizeny na"
      f" {100*(CAS[0] @ X_opt[0])/KAPACITA[0]:.0f} %. A nedrzi cas, ale reagencie:"
      f" kazety dosly driv nez minuty, takze cas navic lezi ladem.")
nej_zas = int(np.argmax(y_zas))
klic = list(ZASOBA)[nej_zas]
print(f"Doobjednat se vyplati predevsim {POPIS_ZASOBY[klic]}: kazda dalsi kazeta"
      f" usetri {y_zas[nej_zas]:.0f} Kc, u druhe reagencie jen"
      f" {min(y_zas):.0f} Kc.")

print("\n=== dokdy stinove ceny plati ===")
print("  kapacita B [min] ->  cena [Kc/den]   stinova cena [Kc/min]")
for kap_b in [260.0, 280.0, 300.0, 320.0, 360.0, 380.0, 400.0]:
    _, cena, _, yk, _, _ = vyres(kapacita=np.array([KAPACITA[0], kap_b, KAPACITA[2]]))
    print(f"    {kap_b:6.0f}            {cena:9.2f}          {yk[1]:6.2f}")
print("  -> 6 Kc/min plati na intervalu <270; 370> min; mimo nej se meni baze")

print("\n=== nezavisly prepocet pres scipy.optimize.linprog ===")
X_sp, cena_sp, status_sp, duals_sp = linprog_kontrola()
duals_cvxpy = np.concatenate([y_kap, y_pop, y_zas])
print(f"  status = {status_sp} (0 = optimal)")
print(f"  cena   = {cena_sp:.2f} Kc")
print(f"  rozdil cen proti CVXPY      : {abs(cena_sp - cena_opt):.2e} Kc")
print(f"  nejvetsi rozdil v rozvrzeni : {np.abs(X_sp - X_opt).max():.2e} vzorku")
print(f"  nejvetsi rozdil v dualech   : {np.abs(duals_sp - duals_cvxpy).max():.2e}")

# --- kontrola dosazením zpět proti zadání, ne proti modelu -------------------
assert status == "optimal", "uloha neni pripustna nebo je neomezena"
for i in range(3):
    assert CAS[i] @ X_opt[i] <= KAPACITA[i] + 1e-6, f"prekrocena kapacita {i}"
for j in range(4):
    assert X_opt[:, j].sum() >= POPTAVKA[j] - 1e-6, f"neudelane vzorky typu {j}"
for (i, j), mez in ZASOBA.items():
    assert X_opt[i, j] <= mez + 1e-6, f"prekrocena zasoba reagencie {(i, j)}"
assert np.abs(X_opt[~UMI]).max() < 1e-6, "vzorek prirazen stroji, ktery ho neumi"
assert (X_opt >= -1e-6).all(), "zaporny pocet vzorku"
print("\nkontrola dosazenim zpet prosla")
