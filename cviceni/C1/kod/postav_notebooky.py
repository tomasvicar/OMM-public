"""Generuje Colab notebooky v cviceni/C1/notebooks/: sedm k ukazkam
a jeden studentsky (hematokrit-student) do tabuloveho bloku prvniho cviceni.

POZOR: notebooky jsou generovane, tenhle skript je jejich jediny zdroj pravdy.
Rucni uprava .ipynb se pri dalsim spusteni prepise - opravovat se ma tady.
Duvod, proc to generator dela: vsech sedm ukazek ma zamerne stejnou stavbu,
a ta se rucne pres sedm souboru neudrzi.

Kazdy notebook ma stejny rytmus:

    markdown  # Nazev / ## Zadani slovy / ## Formulace / ## Od zadani ke kodu
    kod       instalace cvxpy (jen tam, kde se pouziva)
    kod       importy (+ pripadna priprava dat)
    markdown  ## Model a reseni
    kod       parametry s @param, sestaveni ulohy, reseni, vypis
    markdown  ## Kontrola, ktera umi selhat
    kod       nezavisle overeni s assertem
    markdown  ## Obrazek
    kod       graf
    markdown  ## Na co se zeptat kodu

Spousti se z korene repozitare:

    uv run python cviceni/C1/kod/postav_notebooky.py               # zapise .ipynb
    uv run python cviceni/C1/kod/postav_notebooky.py --zkontroluj  # + spusti kod
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CILOVY_ADRESAR = Path("cviceni/C1/notebooks")
GITHUB = ("https://github.com/tomasvicar/OMM-public/"
          "blob/master/cviceni/C1/kod")

INSTALACE = """try:
    import cvxpy as cp
except ImportError:
    %pip install -q cvxpy
    import cvxpy as cp"""


def zdroj(text: str) -> list[str]:
    """Text na seznam radku ve formatu, ktery ceka nbformat."""
    radky = text.strip("\n").split("\n")
    return [r + "\n" for r in radky[:-1]] + [radky[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": zdroj(text)}


def kod(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": zdroj(text)}


def uloz(jmeno: str, bunky: list[dict]) -> None:
    notebook = {
        "cells": bunky,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    cesta = CILOVY_ADRESAR / f"{jmeno}.ipynb"
    cesta.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"zapsano {cesta}")


def paticka(skript: str) -> str:
    return (f"Plná verze téhle ukázky, ze které notebook vychází, je ve skriptu "
            f"[`kod/{skript}.py`]({GITHUB}/{skript}.py) v repozitáři předmětu.")


# ---------------------------------------------------------------- plavcik ---

uloz("plavcik", [
    md(r"""
# Plavčík a tonoucí

## Zadání slovy

> Plavčík stojí na pláži 25 m od vody. Tonoucí je 45 m stranou a 35 m od břehu.
> Po písku plavčík běží 6 m/s, ale plave jen 1,2 m/s — pětkrát pomaleji.
>
> Ve kterém místě má vběhnout do vody, aby byl u tonoucího **co nejdřív**?

Nejkratší cesta je přímka, jenže ta vede zbytečně dlouho vodou. Běžet po břehu
až naproti tonoucímu a plavat kolmo zase znamená naběhat spoustu metrů navíc.
Optimum leží mezi tím — a mnohem blíž „naproti tonoucímu“, než by člověk čekal.

## Formulace

Břeh je vodorovná osa. Plavčík stojí $A$ metrů nad ní, tonoucí $B$ metrů pod ní
a vodorovně jsou od sebe $D$. Proměnná je jediná: souřadnice $x$ místa, kde
plavčík vběhne do vody. Doba je dráha po písku dělená rychlostí běhu plus dráha
ve vodě dělená rychlostí plavání, obojí Pythagorova věta.

$$
\begin{aligned}
\text{minimize}_{x}\quad & T(x)=\frac{\lVert (x,\,A)\rVert_2}{v_1}
  +\frac{\lVert (D-x,\,B)\rVert_2}{v_2} && \text{doba do tonouciho (s)}\\
\text{subject to}\quad & 0 \le x \le D && \text{usek brehu (m)}
\end{aligned}
$$

Data: $A = 25$ m, $B = 35$ m, $D = 45$ m, $v_1 = 6$ m/s, $v_2 = 1{,}2$ m/s.

Účelová funkce je součet dvou eukleidovských norem složených s afinním
zobrazením, takže je **konvexní**: minimum je jediné a solver ho najde
z libovolného startu. CVXPY ji proto vezme přesně tak, jak je zapsaná výš.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| proměnná $x$ | `x = cp.Variable()` |
| $\lVert (x,\,A)\rVert_2 / v_1$ | `cp.norm(cp.hstack([x, A])) / v1` |
| $\lVert (D-x,\,B)\rVert_2 / v_2$ | `cp.norm(cp.hstack([D - x, B])) / v2` |
| $\min T(x)$ | `cp.Minimize(beh + plavani)` |
| $0 \le x \le D$ | `[x >= 0, x <= D]` |

""" + paticka("ukazka_plavcik")),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
"""),
    md("## Model a řešení"),
    kod("""
# vzdálenosti v metrech, rychlosti běhu po písku a plavání v m/s
A = 25.0   # @param {type:"number"}
B = 35.0   # @param {type:"number"}
D = 45.0   # @param {type:"number"}
v1 = 6.0   # @param {type:"number"}
v2 = 1.2   # @param {type:"number"}

x = cp.Variable()                              # kde vběhnout do vody
beh = cp.norm(cp.hstack([x, A])) / v1          # dráha po písku / rychlost běhu
plavani = cp.norm(cp.hstack([D - x, B])) / v2  # dráha ve vodě / rychlost plavání
uloha = cp.Problem(cp.Minimize(beh + plavani), [x >= 0, x <= D])
# dno je ploché, tak se solveru předepíše přísnější tolerance, než má ve výchozím stavu
uloha.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)

x_opt, t_opt = float(x.value), float(uloha.value)
print(f"úloha je konvexní podle DCP: {uloha.is_dcp()}, stav: {uloha.status}")
print(f"optimum: x* = {x_opt:.2f} m, T* = {t_opt:.2f} s")
"""),
    md("""
## Kontrola, která umí selhat

Podmínka $T'(x) = 0$ se dá vyřešit tužkou a vyjde z ní **Snellův zákon**: poměr
sinů úhlů od kolmice k břehu se rovná poměru rychlostí. Solver o optice nic
neví, takže je to nezávislé měřítko — a kontrola má smysl jen tehdy, když umí
spadnout, proto ji pustíme i na řešení posunuté o metr.
"""),
    kod("""
theta1, theta2 = np.arctan2(x_opt, A), np.arctan2(D - x_opt, B)
pomer_sinu, pomer_rychlosti = np.sin(theta1) / np.sin(theta2), v1 / v2
print(f"poměr sinů {pomer_sinu:.4f} vs. poměr rychlostí {pomer_rychlosti:.4f}")
assert abs(pomer_sinu - pomer_rychlosti) < 1e-3, "optimum neodpovídá Snellovu zákonu"

t1, t2 = np.arctan2(x_opt + 1, A), np.arctan2(D - x_opt - 1, B)
print(f"o metr vedle: poměr sinů {np.sin(t1) / np.sin(t2):.4f} — tam kontrola spadne")
"""),
    md("## Obrázek"),
    kod("""
plt.plot([0, x_opt, D], [A, 0, -B], "-o", label=f"nejrychlejší trasa, T* = {t_opt:.1f} s")
plt.axhline(0, color="gray", lw=1, label="břeh (nahoře písek, dole voda)")
plt.gca().set_aspect("equal")
plt.xlabel("vzdálenost podél břehu [m]")
plt.ylabel("vzdálenost od břehu [m]")
plt.title("Kde vběhnout do vody?")
plt.legend(loc="lower left")
plt.show()
"""),
    md(r"""
## Na co se zeptat kódu

1. Nastavte stejné rychlosti ($v_1 = v_2$). Co udělá trasa a co poměr sinů?
2. Zpomalte plavání na $v_2 = 0{,}4$ m/s. Kterého z omezení $0 \le x \le D$ se
   řešení skoro dotkne?
3. Zapište účelovou funkci špatně — minimalizujte dráhu místo doby. Solver vrátí
   číslo i tak; jak by se na chybu přišlo bez znalosti správné odpovědi?
"""),
])


# ------------------------------------------------------------- hematokrit ---

uloz("hematokrit", [
    md(r"""
# Optimální hematokrit

## Zadání slovy

> Hematokrit $H$ je objemový podíl červených krvinek v krvi. Krvinky nesou
> kyslík, takže čím vyšší hematokrit, tím víc kyslíku v každém mililitru krve.
>
> **Proč tedy nemáme hematokrit 80 %?**

Protože s hematokritem roste **viskozita krve**, a to exponenciálně. Průtok
trubicí je podle Poiseuilleova zákona viskozitě nepřímo úměrný, takže hustší
krev teče pomaleji. Dodávka kyslíku do tkání je součin obou věcí — kolik kyslíku
nese mililitr krve krát kolik mililitrů proteče. Jedna část součinu roste, druhá
klesá, takže **optimum leží uvnitř** rozsahu, ne na jeho okraji.

## Formulace

Proměnná je jediná, hematokrit $H$, a model má tři řádky:

- Poiseuille: $Q \sim \Delta P\,r^4/\mu$, tedy při dané geometrii a tlakovém
  spádu $Q \sim 1/\mu$;
- viskozita: $\mu(H)=\mu_p\,e^{\alpha H}$ s $\alpha = 2{,}5$ — empirický fit
  viskozimetrie plné krve pro $H$ zhruba od $0{,}2$ do $0{,}6$;
- dodávka kyslíku: $D(H)\sim H\,Q(H)\sim H\,e^{-\alpha H}$.

$$
\begin{aligned}
\text{maximize}_{H}\quad & D(H) = H\,e^{-\alpha H}
  && \text{dodavka kysliku (rel. jednotky)}\\
\text{subject to}\quad & H_{\min} \le H \le H_{\max}
  && \text{rozsah, ktery ma smysl merit}
\end{aligned}
$$

Data: $\alpha = 2{,}5$, $H_{\min} = 0{,}10$, $H_{\max} = 0{,}70$.

Takhle zapsaná účelová funkce konkávní **není**, a CVXPY by ji nevzal. Logaritmus
je ale rostoucí funkce, takže nemění polohu maxima — a $\log D(H) = \log H
- \alpha H$ už konkávní je (součet konkávního $\log H$ a lineárního členu). Do
solveru tedy jde logaritmus a výsledné $H^\star$ je totéž.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| proměnná $H > 0$ | `H = cp.Variable(pos=True)` |
| $\log D(H) = \log H - \alpha H$ | `cp.log(H) - alpha * H` |
| $\max D(H)$ | `cp.Maximize(...)` |
| $H_{\min} \le H \le H_{\max}$ | `[H >= H_min, H <= H_max]` |
| hodnota $D$ v optimu | `np.exp(uloha.value)` (solver vrátil logaritmus) |

""" + paticka("ukazka_hematokrit")),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
"""),
    md("## Model a řešení"),
    kod("""
alpha = 2.5   # @param {type:"slider", min:1.5, max:4.0, step:0.1}
H_min = 0.10  # @param {type:"slider", min:0.05, max:0.30, step:0.01}
H_max = 0.70  # @param {type:"slider", min:0.35, max:0.90, step:0.01}

H = cp.Variable(pos=True)
uloha = cp.Problem(cp.Maximize(cp.log(H) - alpha * H), [H >= H_min, H <= H_max])
uloha.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
h_opt = float(H.value)

print(f"stav řešení: {uloha.status}")
print(f"optimum: H* = {h_opt:.4f}")
print(f"dodávka v optimu: D = {np.exp(uloha.value):.5f}")
"""),
    md(r"""
## Kontrola, která umí selhat

Podmínka $D'(H) = 0$ se dá vyřešit tužkou: vede na $\alpha H = 1$, tedy
$H^\star = 1/\alpha$ — bez ohledu na tlak, poloměr cévy i viskozitu plazmy.
Solver o té derivaci nic neví, takže je to nezávislé měřítko.
"""),
    kod("""
h_teor = 1.0 / alpha  # z podmínky D'(H) = 0 vyjde alpha*H = 1
print(f"analyticky 1/alpha = {h_teor:.6f}, rozdíl proti solveru {abs(h_opt - h_teor):.2e}")
assert abs(h_opt - h_teor) < 1e-6, "optimum se neshoduje s analytickým řešením"

# kontrola umí spadnout: klidová hodnota u mužů je přípustná, ale optimum to není
dodavka = lambda h: h * np.exp(-alpha * h)
print(f"H = 0,45 je přípustné, ale dodávka je o "
      f"{100 * (1 - dodavka(0.45) / dodavka(h_opt)):.2f} % nižší — tam kontrola spadne")
"""),
    md("## Obrázek"),
    kod("""
hs = np.linspace(H_min, H_max, 400)
plt.plot(hs, dodavka(hs), label=r"$D(H) = H\\,e^{-\\alpha H}$")
plt.axvline(h_opt, ls="--", label=f"optimum $H^\\\\star$ = {h_opt:.2f}")
plt.xlabel("hematokrit $H$ [-]")
plt.ylabel("dodávka kyslíku [rel. jednotky]")
plt.title("Optimální hematokrit")
plt.legend()
plt.show()
"""),
    md(r"""
## Na co se zeptat kódu

1. Změňte $\alpha$ na 2,0 nebo 3,5 — kam se posune optimum a proč vyjde vždycky
   $1/\alpha$?
2. Stáhněte $H_{\max}$ pod $1/\alpha$ (třeba na 0,35) — které omezení se stane
   aktivním a proč kontrola spadne, i když solver počítá správně?
3. Optimum modelu vyjde 0,40, norma u mužů je 0,42–0,50. Kolik procent dodávky ta
   odchylka stojí?
"""),
])


# ------------------------------------------------- hematokrit pro studenty ---
# Notebook do prvniho cviceni: zadani + syntaxe CVXPY na trech malych
# prikladech, ktere s hematokritem nesouvisi, a pak PRAZDNA bunka. Formulaci
# si student napise sam - to je cely ukol. Reseni tady zamerne neni.

uloz("hematokrit-student", [
    md(r"""
# Hematokrit v CVXPY — napiš si to sám

## Zadání

> Hematokrit $H$ je objemový podíl červených krvinek v krvi. Krvinky nesou
> kyslík, takže čím vyšší hematokrit, tím víc kyslíku v každém mililitru krve.
> S hematokritem ale roste viskozita, takže krev teče pomaleji.
>
> **Při jakém hematokritu se do tkání dostane nejvíc kyslíku?**

$$
\begin{aligned}
\text{maximize}_{H}\quad & D(H) = H\,e^{-\alpha H}
  && \text{dodavka kysliku (rel. jednotky)}\\
\text{subject to}\quad & 0{,}20 \le H \le 0{,}60
  && \text{rozsah, kde plati fit viskozity}
\end{aligned}
$$

Data: $\alpha = 2{,}5$.

**Úkol:** zapsat tuhle úlohu v CVXPY, vyřešit ji a porovnat výsledek s tím, co
vyšlo na tabuli. Níž je přehled syntaxe na třech malých příkladech, které
s hematokritem nesouvisí — jsou tam proto, aby se formulace dala **odvodit**,
ne opsat.
"""),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import numpy as np
"""),
    md("""
## Co je CVXPY

Knihovna pro **konvexní** optimalizaci. Úloha se v ní nepíše jako algoritmus, ale
skoro jako na papíře — proměnná, účelová funkce, seznam omezení. CVXPY z toho
sám pozná, jestli je úloha konvexní (`is_dcp()`), přeloží ji do standardního
tvaru, pustí na ni numerický solver (CLARABEL, SCS, OSQP…) a vrátí řešení
i jeho stav. **Není to obecný optimalizátor:** co neumí označit za konvexní, to
odmítne.

## Syntaxe na třech příkladech

Každá úloha má v CVXPY tytéž čtyři kusy: **proměnnou**, **účelovou funkci**,
**seznam omezení** a **složení do `cp.Problem`**. Pak už se jen volá `solve()`.
"""),
    md(r"""
**1) Jedna proměnná a meze.** Nejlepší nepřípustný bod je $x = 3$, jenže ten do
mezí nespadne — optimum proto skončí **na mezi** a ta je pak *aktivní*.

$$
\begin{aligned}
\text{minimize}_{x}\quad & (x-3)^2\\
\text{subject to}\quad & 0 \le x \le 2
\end{aligned}
$$
"""),
    kod("""
# 1) jedna proměnná, meze -- a mez, která je v optimu aktivní
x = cp.Variable()                          # jedno reálné číslo, které solver hledá
ucel = cp.Minimize(cp.square(x - 3))       # co se minimalizuje
omezeni = [x >= 0, x <= 2]                 # seznam podmínek

uloha = cp.Problem(ucel, omezeni)
print("je to konvexní (DCP)?", uloha.is_dcp())
uloha.solve()
print(f"stav: {uloha.status},  x* = {float(x.value):.4f},  účelová funkce = {uloha.value:.4f}")
print("optimum sedí na horní mezi -- ta mez je AKTIVNÍ")
"""),
    md(r"""
**2) Maximalizace a logaritmus.** Účelová funkce je konkávní, takže se smí
maximalizovat rovnou; `pos=True` říká, že proměnná je kladná, jinak by logaritmus
neměl smysl.

$$
\begin{aligned}
\text{maximize}_{y}\quad & \ln y - y\\
\text{subject to}\quad & 0 < y \le 5
\end{aligned}
$$
"""),
    kod("""
# 2) maximalizace a logaritmus; pos=True říká, že proměnná je kladná
y = cp.Variable(pos=True)
uloha = cp.Problem(cp.Maximize(cp.log(y) - y), [y <= 5])
uloha.solve()
print(f"y* = {float(y.value):.4f}   (maximalizovat f je totéž co minimalizovat -f)")
"""),
    md(r"""
**3) Vektorová proměnná a rovnice.** Proměnná nemusí být jedno číslo; omezení
může být i rovnost. Hledá se nejbližší bod k $(1,2,3)$ mezi těmi, které mají
nezáporné složky se součtem 3.

$$
\begin{aligned}
\text{minimize}_{\mathbf z \in \mathbb R^3}\quad
  & \lVert \mathbf z - (1,2,3)\rVert_2^2\\
\text{subject to}\quad & \textstyle\sum_i z_i = 3,\qquad \mathbf z \ge 0
\end{aligned}
$$
"""),
    kod("""
# 3) víc proměnných najednou: proměnná může být vektor a omezení rovnice
z = cp.Variable(3)
uloha = cp.Problem(cp.Minimize(cp.sum_squares(z - np.array([1.0, 2.0, 3.0]))),
                   [cp.sum(z) == 3, z >= 0])
uloha.solve()
print("z* =", np.round(z.value, 4), "  součet =", round(float(np.sum(z.value)), 4))
"""),
    md(r"""
### Přehled

| co potřebuju | jak se to píše |
|---|---|
| proměnná | `cp.Variable()`, `cp.Variable(pos=True)`, `cp.Variable(n)` |
| účelová funkce | `cp.Minimize(vyraz)` nebo `cp.Maximize(vyraz)` |
| omezení | seznam: `[x >= 0, x <= 1, cp.sum(x) == 1]` |
| úloha | `uloha = cp.Problem(ucel, omezeni)` |
| je to konvexní? | `uloha.is_dcp()` |
| vyřešit | `uloha.solve()` |
| výsledky | `x.value`, `uloha.value`, `uloha.status` |
| užitečné funkce | `cp.square`, `cp.sqrt`, `cp.log`, `cp.exp`, `cp.norm`, `cp.sum_squares` |

Když CVXPY zápis odmítne s `DCPError`, **není to porucha**: je to táž věta jako
na tabuli — v tomhle tvaru se úloha řešit nedá a musí se přepsat.
"""),
    md(r"""
## Pravidla skládání: podle čeho CVXPY pozná konvexní zápis

CVXPY nederivuje. Skládá výraz z dílů, o kterých ví, jestli jsou konvexní, nebo
konkávní — a podle několika pravidel z toho odvodí, co je celek. Proto některé
zápisy odmítne, ačkoli by je NumPy spolkl.

| pravidlo | příklad zápisu |
|---|---|
| součet konvexních je konvexní | `cp.square(x) + cp.abs(x)` |
| kladný násobek konvexní je konvexní | `3 * cp.square(x)` |
| maximum z konvexních je konvexní | `cp.maximum(x, cp.square(x))` |
| konvexní složená s **afinní** je konvexní | `cp.square(2 * x - 1)` |
| lineární je konvexní **i** konkávní | `2 * x + 1` |
| normy jsou konvexní | `cp.norm(x)` |
| mínus konkávní je konvexní | `-cp.log(x)` |
| ✗ součin dvou výrazů s proměnnou | `x * y`, `x * cp.exp(x)` |
| ✗ minimum ze dvou konvexních | `cp.minimum(cp.square(x), cp.abs(x))` |

**Na hematokritu:** účelová funkce v tom tvaru, do kterého se převedla na tabuli,
je $\alpha H - \ln H$, tedy **lineární člen plus $-\ln H$**. Lineární je konvexní,
$\ln$ je konkávní a mínus konkávní je konvexní — a součet dvou konvexních je
konvexní. Proto tenhle zápis projde. Původní tvar $H\,e^{-\alpha H}$ je naproti
tomu **součin dvou výrazů s proměnnou** a žádné pravidlo na něj nesedí.

Každý výraz se v CVXPY dá zeptat sám:
"""),
    kod("""
t = cp.Variable(pos=True)
vyrazy = (("2.5 * t          (lineární)", 2.5 * t),
          ("-cp.log(t)       (mínus konkávní)", -cp.log(t)),
          ("2.5*t - cp.log(t)  (součet obou)", 2.5 * t - cp.log(t)),
          ("t * cp.exp(-t)   (součin s proměnnou)", t * cp.exp(-t)))
for popis, vyraz in vyrazy:
    print(f"{popis:38s} konvexní? {str(vyraz.is_convex()):5s} konkávní? {vyraz.is_concave()}")
"""),
    md("""
## Úkol: hematokrit

Buňka je schválně prázdná. Postup je v komentářích, kód je na vás.
"""),
    kod("""
ALPHA = 2.5
H_MIN, H_MAX = 0.20, 0.60

# 1) proměnná H
# 2) účelová funkce -- pozor, v jakém tvaru ji CVXPY vezme (viz tabule)
# 3) seznam omezení
# 4) uloha = cp.Problem(...), vypsat uloha.is_dcp() a zavolat uloha.solve()
# 5) vypsat H.value a porovnat s ručním výsledkem 1 / ALPHA
"""),
    md(r"""
## Až to poběží

1. Sedí $H^\star$ s tím, co vyšlo tužkou? Na kolik desetinných míst?
2. Je některá z mezí aktivní? Co by se stalo, kdyby se z úlohy vyškrtly?
3. Změňte $\alpha$ na 3,0. Kam se optimum posune a proč zrovna tam?
4. Prohoďte schválně `Minimize` a `Maximize`. Solver doběhne a vypíše číslo —
   podle čeho se pozná, že je to nesmysl?
"""),
    md(r"""
---

# 🔒 PRO UČITELE — před rozdáním studentům smazat

**Tenhle nadpis a dvě buňky pod ním jsou řešení.** Nic jiného v notebooku na ně
neodkazuje, takže se dají smazat bez následků: v Colabu *Edit → Delete cell*,
v Jupyteru klávesou `dd`.

Účelová funkce se zapíše v tom tvaru, do kterého se převedla na tabuli —
$g(H) = -\ln D(H) = \alpha H - \ln H$. Druhá buňka ukazuje, co se stane
s původním tvarem: CVXPY ho odmítne, a je to táž věta jako na tabuli.
"""),
    kod("""
# PRO UČITELE: řešení úlohy výš.
H = cp.Variable(pos=True)                       # kladná proměnná, aby šel použít log
uloha = cp.Problem(cp.Minimize(ALPHA * H - cp.log(H)),
                   [H >= H_MIN, H <= H_MAX])
print("je to konvexní (DCP)?", uloha.is_dcp())
uloha.solve()

print(f"stav: {uloha.status}")
print(f"H* z CVXPY = {float(H.value):.6f}")
print(f"H* tužkou  = {1 / ALPHA:.6f}   (rozdíl {abs(float(H.value) - 1 / ALPHA):.2e})")
print(f"obě meze jsou neaktivní: {H_MIN} < {float(H.value):.3f} < {H_MAX}")
"""),
    kod("""
# PRO UČITELE: dvě chyby, které se ve cvičení objeví -- a které CVXPY samo odmítne.
pokusy = (("původní tvar   max H * exp(-alfa*H)", lambda: cp.Maximize(H * cp.exp(-ALPHA * H))),
          ("prohozené min/max   max alfa*H - log(H)", lambda: cp.Maximize(ALPHA * H - cp.log(H))))
for popis, sestav in pokusy:
    try:
        cp.Problem(sestav(), [H >= H_MIN, H <= H_MAX]).solve()
        print(f"{popis}: prošlo (?!)")
    except Exception as chyba:
        print(f"{popis}  ->  {type(chyba).__name__}")

print("\\nOba zápisy CVXPY zastaví už při kontrole DCP, tedy dřív, než se něco spočítá.")
print("Obecný solver (scipy) by je spolkl a vrátil číslo -- proto se kontroluje dosazením zpět.")
"""),
])


# ---------------------------------------------------------------- dychani ---

uloz("dychani", [
    md(r"""
# Optimální dýchání: frekvence a dechový objem

## Zadání slovy

> Musíte provětrat 4,2 litru alveolů za minutu. Můžete dýchat rychle a mělce,
> nebo pomalu a zhluboka — a obojí ten požadavek splní.
>
> **Které z toho stojí dýchací svaly nejmíň?**

Hluboký nádech je drahý, protože se musí roztáhnout plíce: elastická práce roste
s druhou mocninou dechového objemu. Rychlé dýchání je drahé taky, protože vzduch
musí proletět dýchacími cestami rychleji a odpor se platí druhou mocninou
průtoku. Kdyby šlo jen o tohle, nejlevnější by bylo nedýchat vůbec. Jediná věc,
která tomu brání, je požadavek na ventilaci — a proto optimum leží **na jeho
hranici**, ne uvnitř.

## Formulace

Proměnné jsou dvě: dechová frekvence $f$ [1/min] a dechový objem $V_T$ [l].
Model je Otisův, Fennův a Rahnův z roku 1950.

$$
\begin{aligned}
\text{minimize}_{f,\,V_T}\quad & \dot W = \tfrac12 E\,V_T^2 f
  + \frac{\pi^2}{120}R\,f^2 V_T^2
  && \text{vykon dychacich svalu (cmH}_2\text{O}\cdot\text{l/min)}\\
\text{subject to}\quad & f\,(V_T - V_D)\ \ge\ \dot V_A
  && \text{alveolarni ventilace (l/min)}\\
& 4 \le f \le 60,\qquad V_D < V_T \le 2{,}5
  && \text{frekvence (1/min), objem (l)}
\end{aligned}
$$

Data: $E = 10$ cmH₂O/l (elastance plic), $R = 2$ cmH₂O·s/l (odpor dýchacích
cest), $V_D = 0{,}15$ l (mrtvý prostor), $\dot V_A = 4{,}2$ l/min. Tři řádky
modelu:

- **elastická složka:** práce na roztažení plic je za jeden dech
  $\tfrac12 E V_T^2$, za minutu tedy $\tfrac12 E V_T^2 f$ — roste s hloubkou;
- **odporová složka:** při sinusovém průtoku je střední hodnota $\dot V^2$ rovna
  $(\pi f V_T)^2/2$; po převodu $R$ ze sekund na minuty vyjde
  $\tfrac{\pi^2}{120}R f^2 V_T^2$ — roste s rychlostí;
- **mrtvý prostor:** z každého dechu se k alveolům dostane jen $V_T - V_D$,
  protože 0,15 l zůstane v průdušnici a průduškách. Proto v omezení není
  $f\,V_T$, ale $f\,(V_T - V_D)$.

Takhle zapsaná úloha konvexní **není** — v účelové funkci je součin $f^2V_T^2$.
Přípustná oblast sama konvexní je (leží nad grafem konvexní funkce $V_D + \dot V_A/f$),
ale omezení $f\,v \ge \dot V_A$ je součin dvou proměnných a v tomhle tvaru ho DCP
nevezme. Po substituci $V_T = V_D + v$ jsou ale
všechny členy účelové funkce i omezení **monomy s kladnými koeficienty**, tedy
posynomy, a to je přesně definice **geometrického programu**. CVXPY takové úlohy
umí: proměnné se založí jako `pos=True` a řeší se s `gp=True`. Uvnitř se to
zlogaritmuje a stane se z toho konvexní úloha — my to zapíšeme tak, jak stojí
v modelu.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| $f > 0$, $v = V_T - V_D > 0$ | `f = cp.Variable(pos=True)`, `v = cp.Variable(pos=True)` |
| $V_T = V_D + v$ | `V_T = v + V_D` |
| $\tfrac12 E V_T^2 f + \tfrac{\pi^2}{120}R f^2V_T^2$ | `0.5 * E * cp.square(V_T) * f + K * R * cp.square(f) * cp.square(V_T)` |
| $f\,(V_T - V_D) \ge \dot V_A$ | `f * v >= VA` |
| meze $f$ a $V_T$ | `f >= 4, f <= 60, v <= 2.5 - V_D` |
| „je to geometrický program“ | `uloha.is_dgp()`, `uloha.solve(gp=True)` |

Posuvníky odpovídají nemocem: **fibróza** je tuhá plíce, tedy velké $E$;
**CHOPN** je ucpaná dýchací cesta, tedy velké $R$.

""" + paticka("ukazka_dychani")),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
"""),
    md("## Model a řešení"),
    kod("""
E = 10.0    # @param {type:"slider", min:5.0, max:30.0, step:0.5}
R = 2.0     # @param {type:"slider", min:0.5, max:15.0, step:0.5}
V_D = 0.15  # @param {type:"slider", min:0.0, max:0.40, step:0.01}
VA = 4.2    # @param {type:"number"}

K = np.pi**2 / 120  # převede odpor R ze sekund na minuty

# substituce V_T = V_D + v, v > 0: pak je všechno posynom v (f, v)
f = cp.Variable(pos=True)
v = cp.Variable(pos=True)
V_T = v + V_D
W = 0.5 * E * cp.square(V_T) * f + K * R * cp.square(f) * cp.square(V_T)
omezeni = [f * v >= VA, f >= 4, f <= 60, v <= 2.5 - V_D]
uloha = cp.Problem(cp.Minimize(W), omezeni)

print(f"je to geometrický program (DGP)?  {uloha.is_dgp()}")
print(f"je to běžná konvexní úloha (DCP)? {uloha.is_dcp()}")

uloha.solve(gp=True)
f_opt, vt_opt, w_opt = float(f.value), float(v.value) + V_D, float(uloha.value)

print(f"\\nstav řešení: {uloha.status}  ({uloha.solver_stats.solver_name}, "
      f"{1e3 * uloha.solver_stats.solve_time:.2f} ms)")
print(f"optimum:  f* = {f_opt:.4f} /min,  V_T* = {vt_opt:.5f} l")
print(f"alveolární ventilace {f_opt * (vt_opt - V_D):.6f} l/min (požadavek {VA})")
print(f"výkon W* = {w_opt:.4f} cmH2O*l/min = {w_opt * 0.0980665:.4f} J/min")
"""),
    md(r"""
## Kontrola, která umí selhat

Účelová funkce je ve $V_T$ rostoucí, takže při každé frekvenci je nejlevnější
**nejmenší přípustný** objem — omezení tedy musí být aktivní. Dosazením
$f = \dot V_A/v$ pak zmizí jedna proměnná a podmínka
$\mathrm d\dot W/\mathrm dv = 0$ se po vykrácení kladného činitele $(v + V_D)$
zredukuje na **kvadratickou rovnici**

$$a\,v^2 - a\,V_D\,v - 2b\,V_D = 0,\qquad
  a = \tfrac12 E\dot V_A,\quad b = \tfrac{\pi^2}{120}R\dot V_A^2 .$$

Solver o té rovnici nic neví, takže je to nezávislé měřítko. A kontrola má smysl
jen tehdy, když umí spadnout — pustíme ji proto i na podvržené řešení
$f = 12$/min, $V_T = 0{,}50$ l, které je přípustné na šest desetinných míst
a je to učebnicová klidová hodnota.
"""),
    kod("""
a = 0.5 * E * VA
b = K * R * VA**2
koreny = np.roots([a, -a * V_D, -2 * b * V_D])
v_an = float(max(koreny[np.isreal(koreny)].real))
f_an, vt_an = VA / v_an, v_an + V_D
print(f"tužkou:  f = {f_an:.9f} /min,  V_T = {vt_an:.9f} l")
print(f"CVXPY:   f = {f_opt:.9f} /min,  V_T = {vt_opt:.9f} l")
assert max(abs(f_opt - f_an), abs(vt_opt - vt_an)) < 1e-3, "optimum nesedí na kořen kvadratiky"

# kontrola umí spadnout: podvržené řešení je přípustné, ale optimum to není
ff, vv = 12.0, 0.50
print(f"\\npodvržené f = {ff:.0f} /min, V_T = {vv:.2f} l:"
      f" ventilace {ff * (vv - V_D):.6f} l/min, tedy přípustné,")
print(f"ale od optima je daleko ({max(abs(ff - f_an), abs(vv - vt_an)):.3f}) —"
      f" tam kontrola spadne. Samotná přípustnost by ho propustila.")
"""),
    md("""
## Obrázek

Dvě proměnné jsou poslední počet, u kterého se do jednoho obrázku vejde
**přípustná oblast i krajina účelové funkce**. Vrstevnice se dá vyjádřit
uzavřeně — účelová funkce je ve $V_T$ kvadratická, takže stačí odmocnit.
"""),
    kod("""
fs = np.linspace(5, 40, 400)
hranice = V_D + VA / fs


def vrstevnice(ff, w):
    return np.sqrt(w / (0.5 * E * ff + K * R * ff**2))


plt.figure(figsize=(7.5, 5))
plt.fill_between(fs, hranice, 1.05, color="#e6f2e9")
plt.fill_between(fs, 0, hranice, color="#e4e4ea")
plt.plot(fs, hranice, color="#1f4e79", lw=3, label=f"$f\\\\,(V_T - V_D) = {VA}$  (hranice)")
for w in (0.68 * w_opt, 1.9 * w_opt):  # jedna vrstevnice pod optimem, jedna nad
    plt.plot(fs, vrstevnice(fs, w), color="#c0392b", lw=1.5, ls=":")
plt.plot(fs, vrstevnice(fs, w_opt), color="#c0392b", lw=2.2, ls="--",
         label=f"vrstevnice $\\\\dot W$ = {w_opt:.1f} se hranice dotýká")
plt.plot(f_opt, vt_opt, "o", ms=12, color="#1a7f37",
         label=f"optimum: {f_opt:.1f}/min, {vt_opt:.2f} l")
plt.xlim(5, 40)
plt.ylim(0, 1.05)
plt.xlabel("dechová frekvence $f$ [1/min]")
plt.ylabel("dechový objem $V_T$ [l]")
plt.title("Optimum leží na hranici a vrstevnice se jí dotýká")
plt.legend(loc="upper right")
plt.show()
"""),
    md(r"""
## Na co se zeptat kódu

1. **Fibróza.** Nastavte $E = 25$ (tuhé plíce). Kam se optimum posune po hranici
   a jak se to jmenuje u lůžka?
2. **CHOPN.** Vraťte $E = 10$ a nastavte $R = 10$. Optimum sjede na opačnou
   stranu — proč?
3. **Mrtvý prostor.** Nastavte $V_D = 0$. Kde skončí solver a proč tam vnitřní
   optimum neexistuje? (Nápověda: při $V_D = 0$ je $f^2V_T^2 = \dot V_A^2$
   konstantní.)
4. **Špatná účelová funkce.** Minimalizujte místo výkonu jen elastickou složku.
   Solver vrátí číslo i tak — jak by se na chybu přišlo bez znalosti správné
   odpovědi?
"""),
])


# --------------------------------------------------------- brachistochrona ---

uloz("brachistochrona", [
    md(r"""
# Brachistochrona — nejrychlejší skluzavka

## Zadání slovy

> Máme dva body: start vlevo nahoře a cíl 2 m vpravo a 0,6 m níž. Mezi ně se má
> postavit skluzavka. Kulička se pustí bez počátečního impulzu, jede jen tíhou,
> tření se zanedbá.
>
> **Jaký tvar skluzavky dopraví kuličku do cíle za nejkratší čas?**

Přímka je nejkratší, ale ne nejrychlejší. Kdo klesne strmě hned na začátku, má
zbytek cesty rychlost navíc — a zaplatí za to delší dráhou. Optimum je někde
mezi, a spadne dokonce **pod úroveň cíle**, aby na konci zase stoupalo.

## Formulace

Dráha se rozseká na $n$ úseků v pevných $x$-ových uzlech. Proměnné jsou jen
výšky $y_1,\dots,y_{n-1}$ vnitřních uzlů — o „křivce“ není v zápisu ani slovo.
Rychlost plyne ze zachování energie, $v(y)=\sqrt{2g\,(y_0-y)}$, a na každé
úsečce je zrychlení konstantní, takže doba průjezdu úsečky je přesně její délka
dělená **průměrnou** rychlostí.

$$
\begin{aligned}
\text{minimize}_{\mathbf y}\quad & T(\mathbf y)=\sum_{i=0}^{n-1}
  \frac{2\sqrt{(x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2}}{v_i+v_{i+1}}
  && \text{doba sjezdu (s)}\\
\text{where}\quad & v_i=\sqrt{2g\,(y_0-y_i)}
  && \text{zachovani energie}\\
\text{subject to}\quad & y_0=0,\qquad y_n=-0{,}6
  && \text{pevne krajni body (m)}\\
& -2 \le y_i \le -10^{-4},\quad i=1,\dots,n-1
  && \text{box}
\end{aligned}
$$

Horní mez $-10^{-4}$ ošetřuje **singularitu na startu**: v bodě $y=y_0$ je
rychlost nulová a $1/v$ diverguje. Průměrná rychlost na úsečce ale nulová není,
takže se integruje přes ni, ne přes okamžitou rychlost.

Účelová funkce není konvexní ani hladce zapsatelná pro CVXPY, takže se řeší
`scipy.optimize.minimize` metodou L-BFGS-B jako **blackbox**: solver o fyzice
nic neví, jen si funkci opakovaně vyhodnotí a gradient si udělá numericky.
Startuje se z přímky.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| pevné uzly $x_0,\dots,x_n$ | `x = np.linspace(0.0, X_CIL, POCET_UZLU)` |
| proměnné $y_1,\dots,y_{n-1}$ | `vysledek.x` (vnitřní uzly) |
| doplnění pevných konců | `cela_draha(y_vnitrni)` |
| $T(\mathbf y)$ | `doba_sjezdu(y)` |
| box $-2 \le y_i \le -10^{-4}$ | `bounds=[(-2.0, -1e-4)] * (POCET_UZLU - 2)` |
| start z přímky | `y_primka[1:-1]` |

""" + paticka("ukazka_brachistochrona")),
    kod("""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize
"""),
    md("## Model a řešení"),
    kod("""
G = 9.81  # tíhové zrychlení [m/s2]
POCET_UZLU = 51  # @param {type:"slider", min:11, max:101, step:10}
X_CIL = 2.0  # @param {type:"slider", min:0.5, max:4.0, step:0.1}
Y_CIL = -0.6  # @param {type:"slider", min:-1.5, max:-0.1, step:0.1}

x = np.linspace(0.0, X_CIL, POCET_UZLU)  # x-ové uzly jsou pevné


def doba_sjezdu(y):
    # v = sqrt(2 g (y0 - y)); na úsečce je zrychlení konstantní, tak doba = délka / prům. rychlost
    v = np.sqrt(2.0 * G * np.maximum(y[0] - y, 0.0) + 1e-16)
    return float(np.sum(2.0 * np.hypot(np.diff(x), np.diff(y)) / (v[:-1] + v[1:])))


def cela_draha(y_vnitrni):
    return np.concatenate(([0.0], y_vnitrni, [Y_CIL]))  # krajní body jsou pevné


y_primka = np.linspace(0.0, Y_CIL, POCET_UZLU)
vysledek = minimize(lambda y: doba_sjezdu(cela_draha(y)), y_primka[1:-1],
                    method="L-BFGS-B", bounds=[(-2.0, -1e-4)] * (POCET_UZLU - 2))
y_opt = cela_draha(vysledek.x)
cas_primka, cas_opt = doba_sjezdu(y_primka), doba_sjezdu(y_opt)

print(f"proměnných: {POCET_UZLU - 2}")
print(f"po přímce: {cas_primka:.4f} s")
print(f"optimum:   {cas_opt:.4f} s  (o {100 * (1 - cas_opt / cas_primka):.0f} % rychleji)")
"""),
    md("""
## Kontrola, která umí selhat

Úloha má od Johanna Bernoulliho známé přesné řešení: je to **cykloida**. Podíl
jejích dvou parametrických rovnic vyřadí poloměr $R$ a zbude jediná rovnice pro
koncový úhel, kterou dopočítá `brentq`. Diskretizace na konečně mnoha uzlech
musí vyjít o kousek **hůř** než přesná křivka, ale ne o víc než procento —
kdyby vyšla lépe, počítá se něco jiného.
"""),
    kod("""
theta = brentq(lambda t: (t - np.sin(t)) / (1.0 - np.cos(t)) - X_CIL / abs(Y_CIL),
               1e-3, 2.0 * np.pi - 1e-3)
R = abs(Y_CIL) / (1.0 - np.cos(theta))
cas_cykloida = float(np.sqrt(R / G) * theta)  # přesná doba sjezdu po cykloidě
odchylka = 100.0 * (cas_opt / cas_cykloida - 1.0)

print(f"analytická cykloida: {cas_cykloida:.4f} s, odchylka optima {odchylka:+.2f} %")
assert 0.0 < odchylka < 1.0, "optimum se neshoduje se známým řešením"
"""),
    md("## Obrázek"),
    kod("""
t = np.linspace(0.0, theta, 400)
plt.plot(x, y_primka, label=f"přímka — {cas_primka:.3f} s")
plt.plot(x, y_opt, label=f"optimum — {cas_opt:.3f} s")
plt.plot(R * (t - np.sin(t)), -R * (1.0 - np.cos(t)),
         label=f"analytická cykloida — {cas_cykloida:.3f} s")
plt.xlabel("vodorovná vzdálenost [m]")
plt.ylabel("výška [m]")
plt.title("Nejrychlejší skluzavka mezi dvěma body")
plt.legend()
plt.show()
"""),
    md("""
## Na co se zeptat kódu

1. Zdvojnásobte `POCET_UZLU` — jak se změní odchylka od cykloidy?
2. Posuňte cíl níž než dál (`Y_CIL = -1.5`, `X_CIL = 0.5`). Klesne optimum pořád
   pod úroveň cíle?
3. Napište v `doba_sjezdu` rychlost bez odmocniny. Solver doběhne a vypíše číslo
   — chytí to kontrola?
"""),
])


# ----------------------------------------------------------- radioterapie ---

uloz("radioterapie", [
    md(r"""
# Plánování radioterapie

## Zadání slovy

> V řezu pacientem leží nádor a hned vedle něj mícha, kterou nesmíme přezářit.
> Ozařovač umí vyslat úzký svazek z osmi úhlů kolem pacienta, v každém úhlu ve
> 12 příčných polohách, a u každého z těch 96 svazků se dá nastavit intenzita.
> Dávka se v tkáni sčítá.
>
> **Jak nastavit 96 intenzit, aby nádor dostal předepsaných 60 Gy a mícha
> i okolní zdravá tkáň co nejméně?**

Fyzika je celá v **matici dávek** $D$: prvek $D_{ij}$ říká, kolik dávky dostane
voxel $i$ od svazku $j$ s jednotkovou intenzitou. Je to Gaussův profil napříč
svazkem (σ = 2,6 voxelu, takže se sousední polohy překrývají) krát exponenciální
útlum s hloubkou. Rozdělení dávky v celém řezu je pak **lineární** funkce
nastavení, $\mathbf d = D\mathbf w$ — a právě proto je z toho zvládnutelná úloha.

## Formulace

Nechť $T$ je množina voxelů nádoru, $R$ voxelů míchy (rizikový orgán) a $Z$
voxelů zdravé tkáně.

$$
\begin{aligned}
\text{minimize}_{\mathbf w}\quad & 3\sum_{i \in R} (D\mathbf w)_i^2
  \;+\; \sum_{i \in Z} (D\mathbf w)_i^2 && \text{davka mimo nador}\\
\text{subject to}\quad & 60 \le (D\mathbf w)_i \le 69,\quad i \in T
  && \text{predpis do nadoru (Gy)}\\
& \mathbf w \ge \mathbf 0 && \text{intenzity nejsou zaporne}
\end{aligned}
$$

Proměnné jsou intenzity svazků $\mathbf w \in \mathbb{R}^{96}$, účelová funkce
je vážený součet čtverců dávky mimo nádor a omezení jsou lineární. Jde tedy
o **konvexní kvadratický program**: každé lokální minimum je globální a solver
vrací certifikát optimality. Váha 3 u míchy není fyzika, ale klinická preference
— je to první místo, kde se dá s modelem hrát.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| matice dávek $D$ | `D` — řádky voxely, sloupce svazky |
| množiny $T$, $R$, $Z$ | masky `nador`, `micha`, `zdrava` |
| $\mathbf w \ge \mathbf 0$ | `w = cp.Variable(D.shape[1], nonneg=True)` |
| $(D\mathbf w)_i$ pro $i \in T$ | `d_nador = D[nador.ravel()] @ w` |
| $3\sum_R(\cdot)^2 + \sum_Z(\cdot)^2$ | `VAHA_MICHA * cp.sum_squares(d_micha) + cp.sum_squares(d_zdrava)` |
| $60 \le (D\mathbf w)_i \le 69$ | `[d_nador >= PREDPIS, d_nador <= TOLERANCE * PREDPIS]` |

""" + paticka("ukazka_radioterapie")),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
"""),
    md("""
## Řez pacientem a matice dávek

Nejdřív vznikne geometrie: kruhové tělo, v něm nádor a kousek vedle mícha. Pak
se pro každý z 96 svazků spočítá, kolik dávky dá do každého voxelu — to je
matice $D$. Do optimalizace pak jde už jen ta matice, o anatomii solver nic neví.
"""),
    kod("""
N = 64  # rozlišení řezu
yy, xx = np.mgrid[0:N, 0:N]
c = (N - 1) / 2
telo = np.hypot(xx - c, yy - c) < 30
nador = np.hypot(xx - c - 6, yy - c + 4) < 7
micha = np.hypot(xx - c + 8, yy - c - 6) < 4
zdrava = telo & ~nador & ~micha

uhel = np.deg2rad(np.arange(0, 360, 45))[:, None, None, None]  # 8 úhlů ozařovače
posun = np.linspace(-18, 18, 12)[None, :, None, None]  # 12 poloh svazku v úhlu
pricne = (xx - c) * np.cos(uhel) + (yy - c) * np.sin(uhel)  # napříč svazkem
hloubka = -(xx - c) * np.sin(uhel) + (yy - c) * np.cos(uhel)  # podél svazku
D = np.exp(-0.5 * ((pricne - posun) / 2.6) ** 2) * np.exp(-0.02 * (hloubka + 32)) * telo
D = D.reshape(-1, N * N).T  # dávka do voxelu i od svazku j s jednotkovou intenzitou
print(f"matice dávek D: {D.shape[0]} voxelů × {D.shape[1]} svazků")
"""),
    md("## Model a řešení"),
    kod("""
VAHA_MICHA = 3  # @param {type:"slider", min:0, max:100, step:1}
PREDPIS = 60  # @param {type:"slider", min:30, max:80, step:1}
TOLERANCE = 1.15  # @param {type:"slider", min:1.05, max:1.4, step:0.05}

w = cp.Variable(D.shape[1], nonneg=True)  # intenzity svazků, nezáporné
d_nador, d_micha, d_zdrava = D[nador.ravel()] @ w, D[micha.ravel()] @ w, D[zdrava.ravel()] @ w
uloha = cp.Problem(
    cp.Minimize(VAHA_MICHA * cp.sum_squares(d_micha) + cp.sum_squares(d_zdrava)),
    [d_nador >= PREDPIS, d_nador <= TOLERANCE * PREDPIS],
)
uloha.solve(solver=cp.CLARABEL)

print(f"stav řešení: {uloha.status}")
print(f"zapnutých svazků: {(w.value > 1e-3 * w.value.max()).sum()} z {D.shape[1]}")
"""),
    md("""
## Kontrola, která umí selhat

Solver vrací intenzity, ne dávku — tak se dávka spočítá **znovu** z vrácených
intenzit a ověří se, že celý nádor opravdu padne do předepsaného pásma.
Srovnávacím plánem je „rovnoměrné ozáření“: všechny svazky stejně silné,
naškálované tak, aby nádor dostal tutéž minimální dávku. Ten je přípustný taky,
jen do míchy pošle násobně víc.
"""),
    kod("""
davka = D @ w.value  # dávka přepočítaná zpět z vráceného řešení
rovno = D @ np.ones(D.shape[1])
rovno *= PREDPIS / rovno[nador.ravel()].min()  # stejné minimum v nádoru

print(f"nádor {davka[nador.ravel()].min():.1f}–{davka[nador.ravel()].max():.1f} Gy "
      f"(předpis {PREDPIS:.0f}–{TOLERANCE * PREDPIS:.0f} Gy)")
print(f"mícha max: rovnoměrně {rovno[micha.ravel()].max():.1f} Gy, "
      f"optimalizace {davka[micha.ravel()].max():.1f} Gy")

assert davka[nador.ravel()].min() >= PREDPIS - 1e-6
assert davka[nador.ravel()].max() <= TOLERANCE * PREDPIS + 1e-6
print("kontrola dosazením zpět: celý nádor je v předepsaném pásmu")
"""),
    md("""
## Obrázek

Dose-volume histogram: pro každou dávku na vodorovné ose říká, kolik procent
objemu dané tkáně jí dostane aspoň tolik. Ideál je svislá čára u nádoru
a co nejvíc vlevo stlačená křivka u míchy.
"""),
    kod("""
for maska, popis in [(nador, "nádor"), (micha, "mícha"), (zdrava, "zdravá tkáň")]:
    objem = np.linspace(0, 100, int(maska.sum()))
    cara, = plt.plot(np.sort(davka[maska.ravel()])[::-1], objem, label=popis)
    plt.plot(np.sort(rovno[maska.ravel()])[::-1], objem, "--", color=cara.get_color())
plt.xlabel("dávka [Gy]")
plt.ylabel("podíl objemu s aspoň touto dávkou [%]")
plt.title("Dose-volume histogram (plně optimalizace, čárkovaně rovnoměrně)")
plt.legend()
plt.show()
"""),
    md("""
## Na co se zeptat kódu

1. Nastavte váhu míchy na 0 a pak na 100 — o kolik klesne dávka v míše a komu se
   to připíše?
2. Zúžení tolerance na 1,05 nádor zhomogenizuje. Kolik za to zaplatí mícha?
3. Co se stane, když dávku v nádoru místo předepsání *maximalizujete*?
"""),
])


# ----------------------------------------------------------------- raketa ---

uloz("raketa", [
    md(r"""
# Měkké přistání rakety

## Zadání slovy

> Raketa padá z výšky 1 500 m rychlostí přes 100 m/s a má přistát přesně na
> plošině — s nulovou rychlostí, aniž by proletěla pod zem nebo vybočila
> z přistávacího kužele. Motor nejde vypnout a znovu zapálit; smí jen škrtit
> mezi 5 a 30 kN. Paliva je málo, takže se ho má spotřebovat co nejméně.
>
> **Jak má počítač řídit tah motoru v každé z 60 vteřin sestupu?**

Není to školní vymyšlenost: přesně tuhle úlohu řeší algoritmus **G-FOLD**
(Açıkmeşe a Ploen), který běží na palubě přistávajících raket. Důvod, proč se
smí pustit v reálném čase pár set metrů nad zemí, je ten, že je **konvexní** —
solver doběhne v předvídatelném čase a nemůže „uvíznout“.

## Formulace

Čas se rozseká na $N = 60$ kroků po $\Delta t = 1$ s. Proměnné jsou polohy
$\mathbf p_k$, rychlosti $\mathbf v_k$ a vektory tahu $\mathbf T_k$ ve všech
krocích, plus pomocná $\sigma_k$ — dohromady přes 350 čísel.

$$
\begin{aligned}
\text{minimize}\quad & \textstyle\sum_k \sigma_k\,\Delta t && \text{palivo}\\
\text{subject to}\quad
& \mathbf v_{k+1}=\mathbf v_k+\Delta t\left(\tfrac{\mathbf T_k}{m}-\mathbf g\right)
  && \text{Newton}\\
& \mathbf p_{k+1}=\mathbf p_k+\Delta t\,\tfrac{\mathbf v_k+\mathbf v_{k+1}}{2}
  && \text{poloha z rychlosti}\\
& \mathbf p_0,\mathbf v_0\ \text{zadane},\qquad
  \mathbf p_N=\mathbf 0,\ \mathbf v_N=\mathbf 0 && \text{mekke pristani}\\
& p_{k,2}\ \ge\ \tfrac12\,|p_{k,1}| && \text{pristavaci kuzel}\\
& \lVert \mathbf T_k\rVert_2\le\sigma_k,\qquad
  T_{\min}\le\sigma_k\le T_{\max} && \text{skrceni motoru (N)}
\end{aligned}
$$

Data: $m = 2000$ kg, $\mathbf p_0 = (1200,\ 1500)$ m, $\mathbf v_0 = (-60,\ -80)$
m/s, $T_{\min} = 5$ kN, $T_{\max} = 30$ kN.

Poslední řádek je pointa pro celý předmět. Skutečné omezení motoru je
$\lVert\mathbf T_k\rVert \ge T_{\min}$ — a to je **nekonvexní**, protože zakazuje
malé tahy uprostřed přípustné koule. Zavedením pomocné proměnné $\sigma_k$, které
se meze předepíšou a norma se k ní přiváže shora, vznikne konvexní úloha (SOCP)
a dá se **dokázat, že optimum je totožné** (*lossless convexification*). Těžké
tedy nebylo úlohu vyřešit, ale zapsat.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| $\mathbf p_k,\ \mathbf v_k$ | `p = cp.Variable((N + 1, 2))`, `v = cp.Variable((N + 1, 2))` |
| $\mathbf T_k,\ \sigma_k$ | `T = cp.Variable((N, 2))`, `sigma = cp.Variable(N)` |
| Newton a poloha z rychlosti | smyčka `for k in range(N)` |
| měkké přistání | `p[N] == 0, v[N] == 0` |
| přistávací kužel | `p[:, 1] >= 0.5 * cp.abs(p[:, 0])` |
| $\lVert\mathbf T_k\rVert\le\sigma_k$ | `cp.norm(T, axis=1) <= sigma` |
| $\sum_k \sigma_k\Delta t$ | `cp.Minimize(cp.sum(sigma) * dt)` |

""" + paticka("ukazka_raketa")),
    kod(INSTALACE),
    kod("""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
"""),
    md("## Model a řešení"),
    kod("""
N, dt = 60, 1.0                       # počet kroků a délka kroku [s]
g, m = 9.81, 2000.0                   # tíhové zrychlení [m/s2], hmotnost rakety [kg]
gv = np.array([0.0, g])
Tmax = 30000.0  # @param {type:"slider", min:20000, max:45000, step:1000}
Tmin = 5000.0                         # motor nejde vypnout ani škrtit pod minimum
p0 = np.array([1200.0, 1500.0])       # počáteční poloha [m]
v0 = np.array([-60.0, -80.0])         # počáteční rychlost [m/s]

p = cp.Variable((N + 1, 2))
v = cp.Variable((N + 1, 2))
T = cp.Variable((N, 2))
sigma = cp.Variable(N)                # pomocná horní mez velikosti tahu

omezeni = [p[0] == p0, v[0] == v0, p[N] == 0, v[N] == 0,   # start a měkké dosednutí
           p[:, 1] >= 0.5 * cp.abs(p[:, 0]),               # přistávací kužel
           cp.norm(T, axis=1) <= sigma, sigma >= Tmin, sigma <= Tmax]
for k in range(N):                    # diskretizovaná Newtonova rovnice
    omezeni += [v[k + 1] == v[k] + dt * (T[k] / m - gv),
                p[k + 1] == p[k] + dt * (v[k] + v[k + 1]) / 2]

uloha = cp.Problem(cp.Minimize(cp.sum(sigma) * dt), omezeni)
uloha.solve()
tah = np.linalg.norm(T.value, axis=1)

print(f"stav řešení: {uloha.status}")
print(f"spotřebovaný impuls: {uloha.value / 1e6:.2f} MN*s")
print(f"velikost tahu: {tah.min():.0f} až {tah.max():.0f} N (meze {Tmin:.0f}-{Tmax:.0f})")
"""),
    md(r"""
## Kontrola, která umí selhat

Solver vrátil rovnou trajektorii i tahy, jenže trajektorie je jeho vlastní
proměnná — kdyby byla dynamika zapsaná špatně, vyšlo by to „správně“ i tak.
Nezávislá kontrola proto vezme **jen nalezené tahy**, odsimuluje s nimi let od
startu bez solveru a podívá se, kde raketa doopravdy skončí. Zároveň se ověří
i to, že se nikde neporušila spodní mez tahu, kvůli které se úloha
konvexifikovala.
"""),
    kod("""
v_sim = np.vstack([v0, v0 + dt * np.cumsum(T.value / m - gv, axis=0)])
p_sim = np.vstack([p0, p0 + dt * np.cumsum((v_sim[:-1] + v_sim[1:]) / 2, axis=0)])
print(f"koncová poloha {np.linalg.norm(p_sim[-1]):.2e} m, "
      f"koncová rychlost {np.linalg.norm(v_sim[-1]):.2e} m/s")
assert np.linalg.norm(p_sim[-1]) < 1e-2 and np.linalg.norm(v_sim[-1]) < 1e-2

print(f"nejmenší tah {tah.min():.0f} N vs. minimum motoru {Tmin:.0f} N")
assert tah.min() >= Tmin - 1e-3, "lossless convexification by tady selhala"
"""),
    md("## Obrázek"),
    kod("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(p_sim[:, 0], p_sim[:, 1], "-o", ms=3, label="trajektorie")
ax1.set(xlabel="vodorovná vzdálenost [m]", ylabel="výška [m]", title="Sestup rakety")
ax1.legend()
ax2.step(np.arange(N) * dt, tah / 1000, where="post", label="velikost tahu")
ax2.set(xlabel="čas [s]", ylabel="tah [kN]", title="Optimální řízení motoru")
ax2.legend()
plt.show()
"""),
    md(r"""
## Na co se zeptat kódu

1. Průběh tahu vyšel *bang-bang* (maximum – minimum – maximum), nikdo to tak
   nezadal. Zkuste minimalizovat `cp.sum_squares(sigma)` místo `cp.sum(sigma)` —
   co se s profilem stane a co to fyzikálně znamená?
2. Stáhněte $T_{\max}$ posuvníkem pod zhruba 1,3násobek váhy rakety
   ($mg = 19{,}6$ kN): solver vrátí `infeasible`, tedy důkaz, že žádné takové
   řízení neexistuje — a to je taky výsledek, ne chyba.
3. Vyhoďte omezení přistávacího kužele. Účelová funkce vyjde *lépe* — proč, a jak
   by se bez znalosti správné odpovědi poznalo, že se počítá jiná úloha?
"""),
])


# -------------------------------------------------------------- topologie ---

uloz("topologie", [
    md(r"""
# Topologická optimalizace výztuhy

## Zadání slovy

> Odlehčená výztuha — držák implantátu nebo rám vnějšího fixátoru. Vlevo je
> pevně přichycená ke kosti, vpravo na ni působí síla. Obdélníkový prostor, do
> kterého se smí stavět, je daný anatomií. Materiálu je ale rozpočet: smí se
> použít nejvýše 35 % objemu toho obdélníku, protože zbytek je hmotnost navíc.
>
> **Kam ten materiál dát, aby byla výztuha co nejtužší?**

Přesně takhle dnes vznikají 3D tištěné titanové implantáty a nosné díly
v letectví: konstruktér zadá zástavbový prostor, uchycení, zatížení a rozpočet
hmotnosti — a tvar dopočítá optimalizace. Výsledek pak často vypadá jako kost,
protože kost řeší tutéž úlohu.

## Formulace

Deska se rozdělí na $N$ čtvercových elementů. Proměnná $x_e$ je **hustota
materiálu** v elementu $e$: $0$ je díra, $1$ plný titan.

$$
\begin{aligned}
\text{minimize}_{\mathbf x}\quad & c(\mathbf x)=\mathbf f^{\top}\mathbf u
  =\sum_{e=1}^{N} E_e(x_e)\,\mathbf u_e^{\top}K_0\,\mathbf u_e
  && \text{compliance}\\
\text{subject to}\quad & K(\mathbf x)\,\mathbf u=\mathbf f
  && \text{rovnovaha (MKP)}\\
& \frac{1}{N}\sum_{e=1}^{N} x_e \le \texttt{VOLFRAC}
  && \text{rozpocet materialu}\\
& 0\le x_e\le 1,\qquad
  E_e(x_e)=E_{\min}+x_e^{\,p}\,(E_0-E_{\min})
  && \text{SIMP}
\end{aligned}
$$

Účelová funkce $c$ je **poddajnost** (*compliance*), tedy práce, kterou síla
vykoná na posunutí; malá poddajnost znamená velkou tuhost. Omezení
$K(\mathbf x)\mathbf u=\mathbf f$ je fyzika — soustava rovnic z metody konečných
prvků, která se musí vyřešit v každé iteraci znovu, protože matice tuhosti $K$
závisí na proměnných.

Exponent $p = 3$ je trik metody **SIMP**: poloviční hustota dá jen osminovou
tuhost, takže se šedé mezihodnoty nevyplatí a řešení samo sklouzne k „buď
materiál, nebo díra“. Tím se ale úloha stane **nekonvexní** a konvexní solver ji
nevezme; hustoty se místo toho aktualizují metodou *optimality criteria*, což je
jednoduchá formule odvozená z KKT podmínek. Ke spádu se navíc přidává **filtr**
přes okolí o poloměru `RMIN` elementů — bez něj vyjde šachovnice, která je
v diskretizaci uměle tuhá.

## Od zadání ke kódu

| v zadání | v kódu |
|---|---|
| hustoty $x_e$ | pole `x` tvaru (NELY, NELX) |
| $E_e(x_e)$, SIMP | `E = EMIN + x.ravel()**PENAL * (E0 - EMIN)` |
| $K(\mathbf x)\mathbf u = \mathbf f$ | `u[volne] = spsolve(K[volne][:, volne], f[volne])` |
| $c(\mathbf x)=\mathbf f^\top\mathbf u$ | návratová hodnota `poddajnost(x)` |
| $\partial c/\partial x_e$ | `dc = -PENAL * x**(PENAL - 1) * (E0 - EMIN) * ce` |
| rozpočet materiálu | půlení násobitele v `krok_oc()` |

""" + paticka("ukazka_topologie")),
    kod("""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
"""),
    md("""
## Síť konečných prvků

Tahle buňka je standardní příprava MKP (matice tuhosti čtvercového elementu,
očíslování posunů, okrajové podmínky, jádro filtru) — je to řemeslo, které se
v každé implementaci opisuje stejně. Optimalizace sama začíná až v další buňce.
"""),
    kod("""
VOLFRAC = 0.35  # @param {type:"slider", min:0.1, max:0.7, step:0.05}
ITERACI = 30  # @param {type:"slider", min:5, max:60, step:5}
RMIN = 2.0  # @param {type:"slider", min:1, max:4, step:0.5}
PENAL = 3.0  # @param {type:"slider", min:1, max:5, step:0.5}
NELX, NELY, E0, EMIN, NU, KROK = 60, 30, 1.0, 1e-9, 0.3, 0.2
POCET, ndof = NELX * NELY, 2 * (NELX + 1) * (NELY + 1)

k = np.array([1/2 - NU/6, 1/8 + NU/8, -1/4 - NU/12, -1/8 + 3*NU/8,
              -1/4 + NU/12, -1/8 - NU/8, NU/6, 1/8 - 3*NU/8])
KE = k[np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1],
                 [3, 6, 5, 0, 7, 2, 1, 4], [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
                 [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]])] / (1 - NU**2)

e = np.arange(POCET)
rohy = ((e // NELX) * (NELX + 1) + e % NELX)[:, None] + [0, 1, NELX + 2, NELX + 1]
edof = np.repeat(2 * rohy, 2, 1); edof[:, 1::2] += 1      # dva posuny na uzel
iK, jK = np.repeat(edof, 8, 1).ravel(), np.tile(edof, 8).ravel()

f = np.zeros(ndof); f[2 * ((NELY // 2) * (NELX + 1) + NELX) + 1] = -1.0   # síla vpravo dolů
volne = np.setdiff1d(np.arange(ndof), 2*np.arange(NELY+1)[:, None]*(NELX+1) + [0, 1])
d = np.arange(-int(RMIN), int(RMIN) + 1)
jadro = np.maximum(0.0, RMIN - np.hypot(*np.meshgrid(d, d)))   # kuželové váhy filtru
Hs = convolve2d(np.ones((NELY, NELX)), jadro, mode="same")
print(f"{POCET} elementů, {ndof} stupňů volnosti, rozpočet {100 * VOLFRAC:.0f} % materiálu")
"""),
    md("""
## Model a řešení

Jedna iterace má tři kroky: vyřešit rovnováhu při současném rozložení materiálu,
spočítat citlivost poddajnosti na každou hustotu a materiál podle ní přerozdělit
tak, aby se rozpočet vyčerpal přesně.
"""),
    kod("""
def poddajnost(x):
    E = EMIN + x.ravel()**PENAL * (E0 - EMIN)             # SIMP: šedé hustoty se penalizují
    K = coo_matrix(((KE.ravel()[:, None] * E).ravel(order="F"), (iK, jK)),
                   shape=(ndof, ndof)).tocsc()
    u = np.zeros(ndof)
    u[volne] = spsolve(K[volne][:, volne], f[volne])
    ce = (u[edof] @ KE * u[edof]).sum(1)                  # energie napjatosti elementu
    return (E * ce).sum(), ce.reshape(NELY, NELX)


def krok_oc(x, dc):
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:           # půlení: stínová cena materiálu
        lm = 0.5 * (l1 + l2)
        xn = np.clip(x * np.sqrt(-dc / lm), np.maximum(0.0, x - KROK), np.minimum(1.0, x + KROK))
        l1, l2 = (lm, l2) if xn.mean() > VOLFRAC else (l1, lm)
    return xn


x = np.full((NELY, NELX), VOLFRAC)                        # start: materiál rovnoměrně
c_rovnomerna = poddajnost(x)[0]
for it in range(ITERACI):
    c, ce = poddajnost(x)                                             # 1) rovnováha K(x) u = f
    dc = -PENAL * x**(PENAL - 1) * (E0 - EMIN) * ce                   # 2) citlivost poddajnosti
    dc = convolve2d(x * dc, jadro, mode="same") / Hs / np.maximum(1e-3, x)   # filtr citlivosti
    x = krok_oc(x, dc)                                                # 3) přerozdělení materiálu
c_opt, _ = poddajnost(x)

print(f"rovnoměrná deska: poddajnost {c_rovnomerna:.1f}")
print(f"optimalizovaný tvar téže hmotnosti: {c_opt:.1f} → {c_rovnomerna / c_opt:.1f}× tužší")
"""),
    md("""
## Kontrola, která umí selhat

Optimality criteria není solver s certifikátem, takže se ověřuje ručně: rozpočet
materiálu musí být vyčerpaný přesně (jinak by šlo přidat materiál a být tužší)
a optimalizovaný tvar musí být tužší než rovnoměrná deska téže hmotnosti.
"""),
    kod("""
assert abs(x.mean() - VOLFRAC) < 1e-3, f"rozpočet nesedí: {x.mean():.4f}"
assert c_opt < c_rovnomerna, "optimalizace nezlepšila ani rovnoměrnou desku"
print(f"průměrná hustota {x.mean():.4f}, rozpočet {VOLFRAC:.2f} — omezení je aktivní")
"""),
    md("## Obrázek"),
    kod("""
plt.imshow(x, cmap="gray_r", vmin=0, vmax=1)
plt.title(f"{100 * VOLFRAC:.0f} % materiálu, poddajnost {c_opt:.1f}")
plt.axis("off")
plt.show()
"""),
    md("""
## Na co se zeptat kódu

1. `VOLFRAC = 0.15` a `0.60` — změní se jen tloušťka žeber, nebo i jejich počet?
2. `RMIN = 1.0` — objeví se šachovnice s *lepší* poddajností. Proč je to artefakt
   sítě?
3. `PENAL = 1.0` — bez penalizace je úloha v hustotách konvexní. Jde výsledek
   vyrobit?
"""),
])


# ------------------------------------------------------------- kontrola ---

if "--zkontroluj" in sys.argv:
    # Kod z bunek se slepi do skriptu a spusti; instalacni bunka se v Colabu
    # postara o cvxpy, lokalne staci import. Obrazky se nezobrazuji.
    hlavicka = ("import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "plt.show = lambda *a, **k: None\n")
    spadlo = False
    with tempfile.TemporaryDirectory() as docasny:
        for cesta in sorted(CILOVY_ADRESAR.glob("*.ipynb")):
            bunky = json.loads(cesta.read_text())["cells"]
            kusy = ["import cvxpy as cp" if "%pip" in (s := "".join(b["source"])) else s
                    for b in bunky if b["cell_type"] == "code"]
            skript = Path(docasny) / f"{cesta.stem}.py"
            skript.write_text(hlavicka + "\n\n".join(kusy) + "\n")
            beh = subprocess.run([sys.executable, str(skript)], capture_output=True, text=True)
            print(f"{'OK   ' if beh.returncode == 0 else 'CHYBA'} {cesta.name}")
            if beh.returncode:
                spadlo = True
                print(beh.stdout[-2000:], beh.stderr[-3000:], sep="\n")
    sys.exit(1 if spadlo else 0)
