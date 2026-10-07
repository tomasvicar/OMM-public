"""ZÁMĚRNĚ CHYBNÉ „hotové“ řešení pro červený praporek třetího cvičení.

**Neopravovat.** Vada v tomhle souboru je didaktický materiál, ne chyba.
Studenti výsledek dostanou jako hotový. Tentokrát výsledek splňuje VŠECHNA
omezení ze zadání. Mají doložit, že to přesto není nejlepší odhad, jaký data
dovolují. Čísla a obrázky počítá `praporek_orezani.py`.

## Úloha

Složení vzorku krve ze 4 typů buněk (T-lymfocyty, NK buňky, monocyty,
neutrofily) z exprese 6 genů. Exprese směsi je vážený průměr expresí čistých
typů (matice A), váhami jsou podíly buněk:

    minimize   f(x) = ||A x - b||^2
    subject to x >= 0,  sum(x) = 1

## Jaká je vada

Kolega spočítal nejmenší čtverce BEZ omezení (`np.linalg.lstsq`), vyšly mu
NK buňky −8,3 %. Záporné číslo nastavil na nulu a zbytek přenormoval na
100 %. Omezení tedy nejsou v modelu, jen „opravená“ po výpočtu. Výsledek je
přípustný, ale ne optimální:

    tenhle skript:  x = (0,411; 0; 0,134; 0,455),  f = 266,2
    správně (QP):   x = (0,366; 0; 0,149; 0,485),  f = 138,6   (o 48 % menší)

## Proč vypadá důvěryhodně

- vlastní kontrola „nic záporného, součet 100 %“ projde a projde i přísná
  kontrola proti zadání: omezení jsou opravdu splněná;
- ořezání jednoho malého záporného čísla vypadá jako nevinná kosmetika.

Důkaz bez solveru je KKT stacionarita: na nenulových podílech musí mít
gradient 2 A^T (A x − b) všude stejnou hodnotu. U kolegy vyjde
(111; 2435; −6276; −5213), na třech nenulových složkách rozdíl přes 6000.

Spuštění z kořene repozitáře:
    uv run python cviceni/C3/kod/cerveny_praporek_orezani.py
    uv run python cviceni/C3/kod/cerveny_praporek_orezani.py --reseni   # vada odhalena
"""

import sys

import numpy as np

TYPY = ["T-lymfocyty", "NK buňky", "monocyty", "neutrofily"]
# řádky geny CD3E, NKG7, GNLY, CD14, FCGR3B, LYZ; sloupce typy buněk
A = np.array([[100, 10, 1, 1],
              [40, 120, 2, 2],
              [20, 80, 1, 1],
              [2, 2, 150, 10],
              [1, 20, 5, 200],
              [5, 5, 300, 120]], dtype=float)
b = np.array([45, 10, 2, 30, 100, 105], dtype=float)     # exprese ve směsi

# ---- řešení kolegy: nejmenší čtverce, pak „oprava“ ----
x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
x_kolega = np.clip(x_ls, 0, None)          # záporný podíl nedává smysl -> 0
x_kolega = x_kolega / x_kolega.sum()       # a ať to dá dohromady 100 %
f_kolega = float(np.sum((A @ x_kolega - b) ** 2))

for typ, podil in zip(TYPY, x_kolega):
    print(f"{typ:12s} {100 * podil:5.1f} %")
print(f"chyba fitu f = {f_kolega:.1f}")
assert np.all(x_kolega >= 0), "záporný podíl"
assert abs(x_kolega.sum() - 1) < 1e-9, "součet není 100 %"
print("kontrola: nic záporného, součet 100 % ✓  ->  hotovo")

if "--reseni" in sys.argv:
    import cvxpy as cp

    print("\n--- odhalení ---")
    print(f"nejmenší čtverce bez omezení: {np.round(100 * x_ls, 1)} %, "
          f"součet {100 * x_ls.sum():.1f} %")

    # cesta 1: vyřešit správnou úlohu, omezení patří do modelu
    x = cp.Variable(4)
    omezeni = [x >= 0, cp.sum(x) == 1]
    uloha = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), omezeni)
    uloha.solve(solver=cp.CLARABEL)
    print(f"správně:  x = {np.round(100 * x.value, 1)} %, f = {uloha.value:.1f}")
    print(f"kolega:   x = {np.round(100 * x_kolega, 1)} %, f = {f_kolega:.1f}  "
          f"(o {100 * (f_kolega / uloha.value - 1):.0f} % horší)")

    # cesta 2: KKT test bez solveru, grad f + lambda*1 - mu = 0, mu >= 0, mu_i x_i = 0
    # -> na nenulových podílech musí být gradient stejný (= -lambda)
    for nazev, xx in [("kolega", x_kolega), ("optimum", x.value)]:
        g = 2 * A.T @ (A @ xx - b)
        volne = xx > 1e-6
        print(f"KKT {nazev:8s} grad f = {np.round(g, 0)}, rozdíl na nenulových "
              f"složkách {np.ptp(g[volne]):.0f}")
    print(f"duály CVXPY: lambda = {omezeni[1].dual_value:.1f}, "
          f"mu = {np.round(omezeni[0].dual_value, 1)}")
