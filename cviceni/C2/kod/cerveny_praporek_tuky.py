"""ZÁMĚRNĚ ROZBITÝ skript pro blok 6 cvičení 2 — červený praporek.

**Neopravovat.** Vada v tomhle souboru je didaktický materiál, ne chyba.
Studenti skript dostanou, spustí ho a mají *doložit*, proč se výsledku nedá
věřit — najít konkrétní číslo, které porušuje zadání, ne napsat „zdá se mi to
divné". Správná verze téže úlohy je v `infuze_matice.py`.

## Jaká je vada

Úloha je maticové rozšíření z bloku 4 (parenterální výživa, 6 přípravků ×
4 živiny). V seznamu omezení **chybí dolní mez tukové emulze** — tedy
`x[3] >= 0,25 l`. Lipidová emulze je v tabulce zdaleka nejdražší položka
(500 Kč/l) a energii z ní jde nahradit glukózou, takže optimalizátor ji
vyhodí úplně: vak vyjde **bez tuků**, tedy bez esenciálních mastných kyselin,
což je při delší parenterální výživě klinicky nepoužitelné.

Horní mez lipidů (`x[3] <= 0,5 l`) v modelu naopak *je*. Působí to tedy, jako
by se na tuky myslelo — a zároveň je to stopa: strop na položku, které v optimu
vyjde nula, je omezení, které nikdy nemohlo nic udržet.

    správně (s dolní mezí tuků):   453,00 Kč, lipid 0,250 l
    tenhle skript (bez ní):        364,67 Kč, lipid 0,000 l
    rozdíl:                         88,33 Kč ve prospěch nepoužitelného vaku

## Proč skript vypadá důvěryhodně

Vypíše `status = optimal`, kompletní složení vaku, cenu, stínové ceny **a
provede kontrolu dosazením zpět, která projde**. Projde ovšem proto, že
kontroluje právě ta omezení, která jsou v modelu — a to omezení, které chybí
v modelu, chybí i v kontrole. To je pointa celého bloku:

    kontrola proti modelu není kontrola; kontroluje se proti zadání.

Úplný seznam klinických požadavků na vak (včetně minimálního podílu tuků)
je v **zadání cvičení**, ne v tomhle souboru. Kdo kontroluje jen to, co vidí
v kódu, vadu nikdy nenajde.

## Na co se ptát

Vyučující se ptá tří studentů na to, **jak** vadu našli, ne co je v ní
špatně. Kdo je hotov dřív, dostane druhou otázku: jak by se taková chyba
dala odhalit automaticky?

Spuštění z kořene repozitáře:
    uv run python cviceni/C2/kod/cerveny_praporek_tuky.py
    uv run python cviceni/C2/kod/cerveny_praporek_tuky.py --reseni   # vada odhalena
"""

import sys

import cvxpy as cp
import numpy as np

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
LIPID_MIN = 0.25     # l, používá se jen v části pod přepínačem --reseni


def sestav(lipid_min=None):
    """Vrátí slovník omezení {popis: výraz} pro danou dolní mez tukové emulze.

    Úplný seznam klinických požadavků, proti kterému se má hotový vak
    kontrolovat, je v **zadání cvičení** — ne tady v kódu.
    """
    x = cp.Variable(6, nonneg=True)
    omezeni = {
        "energie >= 1800 kcal": A[0] @ x >= ENERGIE,
        "bilkoviny >= 70 g": A[1] @ x >= BILKOVINY,
        "draslik >= 60 mmol": A[2] @ x >= DRASLIK_MIN,
        "draslik <= 100 mmol": A[2] @ x <= DRASLIK_MAX,
        "objem == 2 l": A[3] @ x == OBJEM,
        "lipid <= 0,5 l": x[3] <= LIPID_MAX,
    }
    if lipid_min is not None:
        omezeni["lipid >= 0,25 l"] = x[3] >= lipid_min
    return x, omezeni


def vyres(lipid_min=None):
    """Vyřeší LP a vrátí (x, cena, status, duály, omezení)."""
    x, omezeni = sestav(lipid_min)
    uloha = cp.Problem(cp.Minimize(CENA @ x), list(omezeni.values()))
    uloha.solve()
    duals = {k: float(np.atleast_1d(v.dual_value)[0]) for k, v in omezeni.items()}
    return x.value, uloha.value, uloha.status, duals, omezeni


x_opt, cena_opt, status, duals, omezeni = vyres()

print("=== parenteralni vyziva: navrh vaku na 24 hodin ===")
print("6 pripravku, ctyri zivinove pozadavky; uplny seznam pozadavku je v zadani")
print(f"\nstatus = {status}")
print("\nslozeni vaku:")
for nazev, objem in zip(NAZVY, x_opt):
    print(f"  {nazev:10s} {objem:7.4f} l")
print(f"\ncena = {cena_opt:.2f} Kc za vak")

print("\n=== co v tom vaku je ===")
print(f"  energie   {A[0] @ x_opt:8.1f} kcal")
print(f"  bilkoviny {A[1] @ x_opt:8.1f} g")
print(f"  draslik   {A[2] @ x_opt:8.1f} mmol")
print(f"  objem     {A[3] @ x_opt:8.2f} l")

print("\n=== stinove ceny (nula = omezeni neni aktivni) ===")
for nazev, y in duals.items():
    print(f"  {nazev:22s} {y:9.4f}")

# --- kontrola dosazením zpět -------------------------------------------------
# Projde každé omezení modelu a dosadí do něj nalezené x. Vypadá to jako
# poctivá kontrola a formálně to poctivá kontrola je — jen se kontroluje proti
# modelu, ne proti zadání. Viz docstring.
print("\n=== kontrola dosazenim zpet ===")
TOL = 1e-6
proslo = 0
for nazev, vyraz in omezeni.items():
    zbytek = float(np.atleast_1d(vyraz.violation()).max())
    ok = zbytek <= TOL
    proslo += ok
    print(f"  {nazev:22s} odchylka {zbytek:9.2e}  {'OK' if ok else 'PORUSENO'}")
print(f"kontrola dosazenim zpet: {proslo}/{len(omezeni)} omezeni splneno")
if proslo == len(omezeni):
    print("vsechna omezeni modelu jsou splnena, reseni je optimalni")


# ------------------------------------------------------- jen pro vyučujícího --
if "--reseni" in sys.argv:
    x_ok, cena_ok, _, _, _ = vyres(lipid_min=LIPID_MIN)
    print("\n" + "=" * 70)
    print("=== RESENI: co v modelu chybi ===")
    print("V seznamu omezeni neni dolni mez tukove emulze. Lipidy jsou nejdrazsi")
    print("polozka (500 Kc/l) a energii z nich lze nahradit glukozou, takze je")
    print("optimalizator vyhodi uplne - vak vyjde bez esencialnich mastnych kyselin.")
    print("\n  chybejici omezeni:  x[3] >= 0,25 l  (Lipid 20%)")
    print(f"\n{'':22s} {'lipid [l]':>10s} {'cena [Kc]':>10s}")
    print(f"  {'model ze skriptu':20s} {x_opt[3]:10.4f} {cena_opt:10.2f}")
    print(f"  {'s dolni mezi tuku':20s} {x_ok[3]:10.4f} {cena_ok:10.2f}")
    print(f"\nrozdil: {cena_ok - cena_opt:.2f} Kc - o tolik je rozbity model levnejsi")
    print("a za tu usporu dostane pacient vak, ktery mu nesmi byt podan.")
    print("\nProc to kontrola dosazenim zpet nenasla: prochazi omezeni MODELU.")
    print("Omezeni, ktere v modelu chybi, chybi i v kontrole. Kontrolovat se musi")
    print("proti uplnemu seznamu pozadavku ze zadani, ne proti vlastnimu modelu.")
