"""Tabule tretiho cviceni: dva leky, nejdriv bez omezeni, pak s nerovnosti (KKT).

    minimize (x1 - 4)^2 + (x2 - 2)^2      (odchylka od idealnich davek 4 mg a 2 mg)
    1. bez omezeni
    2. subject to x1 + x2 <= b            (KKT, mu; b = 4 aktivni, b = 7 neaktivni,
                                           b = 5 stinova cena)

Vypise vsechna cisla, ktera se pisou na tabuli, a nakresli obrazky reportu
cviceni/C3/reporty/tabule.html. Zavazna cisla jsou v cisla_leky.py.

    uv run python cviceni/C3/kod/tabule_leky.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cisla_leky import CIL, vyres

OBRAZKY = Path(__file__).resolve().parent.parent / "obrazky"


def f(x1, x2):
    return (x1 - 4) ** 2 + (x2 - 2) ** 2


def grad_f(x):
    return 2 * (np.asarray(x) - CIL)


def optimum(b):
    """Optimum pri x1 + x2 <= b: mu = max(0, 6 - b), x = cil - mu/2 * (1, 1)."""
    mu = max(0.0, CIL.sum() - b)
    x = CIL - mu / 2
    return x, f(*x), mu


# vzorec (pouziva ho i hriste v reportu) musi sedet se zavaznym resicem z cisla_leky.py
for b in (4.0, 5.0, 7.0):
    x_ref, f_ref, mu_ref, _ = vyres(b)
    x, fb, mu = optimum(b)
    assert np.allclose(x, x_ref, atol=1e-5) and abs(fb - f_ref) < 1e-5 and abs(mu - mu_ref[0]) < 1e-5

# ------------------------------------------------------------------ cisla na tabuli
print("1) bez omezeni")
H = np.array([[2.0, 0.0], [0.0, 2.0]])
print(f"   grad f = (2(x1-4), 2(x2-2)) = 0  ->  x = {CIL},  f = {f(*CIL):g}")
print(f"   Hessian = 2I: det = {np.linalg.det(H):g}, stopa = {np.trace(H):g}  ->  minimum")
S = np.array([[2.0, 0.0], [0.0, -2.0]])
print(f"   kontrast x1^2 - x2^2: Hessian diag(2, -2), det = {np.linalg.det(S):g}  ->  sedlo")

print("2) nerovnost x1 + x2 <= b:  g = x1 + x2 - b <= 0")
for b in (4, 7):
    x, fb, mu = optimum(b)
    g = x.sum() - b
    print(f"   b = {b}: x = ({x[0]:g}; {x[1]:g}), f = {fb:g}, mu = {mu:g}, g = {g:g}, "
          f"mu*g = {abs(mu * g):g}, grad f + mu*grad g = {grad_f(x) + mu}")
x, fb, mu = optimum(5)
print(f"   stinova cena: b = 5 -> x = ({x[0]:g}; {x[1]:g}), f = {fb:g} (z 2), mu = {mu:g}")
print("   spatny tip b = 4 jako neaktivni: (4; 2) dava g = 2 > 0, neni pripustne")
print("   spatny tip b = 7 jako aktivni: mu = 6 - 7 = -1 < 0, porusi mu >= 0")
print("   predikce (jen vyucujici): pridat x2 >= 1,5 -> v (3; 1) je x2 = 1 < 1,5, "
      "aktivni budou obe (vysledek v samostatne uloze)")


# ------------------------------------------------------------------ obrazky
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "mpc-omm-c3-tabule"})
BARVA_F, BARVA_G, BARVA_OPT, SEDA = "#c73e1d", "#1f4e79", "#1a7f37", "#9ca3af"
X1, X2 = np.meshgrid(np.linspace(-0.5, 7.5, 300), np.linspace(-1.5, 5.5, 300))
UROVNE = [0.5, 2, 4.5, 8, 12.5]


def zaklad(ax, titul):
    ax.contour(X1, X2, f(X1, X2), levels=UROVNE, colors=SEDA, linewidths=0.9)
    ax.plot(*CIL, "o", color=SEDA, ms=6)
    ax.set_xlim(0, 7)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("lék A  x₁ [mg]")
    ax.set_ylabel("lék B  x₂ [mg]")
    ax.set_title(titul, fontsize=12)
    ax.grid(alpha=0.25)


def sipka(ax, bod, vektor, barva, text, posun=(0.08, 0.08), lw=2.2):
    ax.annotate("", xy=bod + vektor, xytext=bod,
                arrowprops=dict(arrowstyle="-|>", color=barva, lw=lw, mutation_scale=16))
    ax.text(*(bod + vektor + np.array(posun)), text, color=barva, fontsize=11)


def kresli_limit(ax, b, popisky=True):
    """Vrstevnice, pripustna polorovina x1 + x2 <= b, optimum a sipky pro dane b."""
    x, fb, mu = optimum(b)
    zaklad(ax, f"limit b = {b:.1f} mg".replace(".", ","))
    ax.fill([-1, b + 2, -1], [-2, -2, b + 1], color="#dbeafe", zorder=0)
    ax.plot([b - 6, b + 1], [6, -1], color=BARVA_G, lw=2)
    ax.plot(*x, "o", color=BARVA_OPT, ms=8, zorder=5)
    if mu > 1e-9:
        # grad f a mu grad g proti sobe, soucet = 0
        sipka(ax, x, grad_f(x) / 2, BARVA_F, "∇f", posun=(-0.55, -0.1))
        sipka(ax, x, mu * np.array([1, 1]) / 2, BARVA_G, "μ∇g", posun=(0.05, 0.05))
        stav = f"aktivní,  μ = {mu:.1f}".replace(".", ",")
    else:
        stav = "neaktivní,  μ = 0"
    if popisky:
        ax.text(0.2, 4.5, stav, fontsize=12, fontweight="bold",
                color=BARVA_G if mu > 1e-9 else BARVA_OPT,
                bbox=dict(fc="white", ec="none", alpha=0.85))
        ax.text(0.2, -0.75, f"x* = ({x[0]:.2f}; {x[1]:.2f}),  f* = {fb:.2f}".replace(".", ","),
                fontsize=10.5, bbox=dict(fc="white", ec="none", alpha=0.85))
    return x, fb, mu


def uloz(fig, jmeno):
    fig.tight_layout()
    fig.savefig(OBRAZKY / jmeno, metadata={"Date": None})
    plt.close(fig)
    print("ulozeno", OBRAZKY / jmeno)


OBRAZKY.mkdir(exist_ok=True)

# 1) bez omezeni: miska dvou leku a kontrastni sedlo
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
zaklad(a, "f = (x₁−4)² + (x₂−2)²: minimum")
for p in [(1.5, 0.2), (6.3, 0.5), (2.0, 4.3), (6.2, 3.8)]:
    p = np.array(p)
    sipka(a, p, -grad_f(p) / 6, BARVA_F, "", lw=1.5)
a.plot(*CIL, "o", color=BARVA_OPT, ms=8)
a.text(4.2, 2.2, "(4; 2)\n∇f = 0", color=BARVA_OPT, bbox=dict(fc="white", ec="none", alpha=0.85))
a.text(0.2, 4.5, "H = 2I:  det 4 > 0, stopa 4 > 0", fontsize=10.5,
       bbox=dict(fc="white", ec="none", alpha=0.85))
S1, S2 = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
cs = b.contour(S1, S2, S1 ** 2 - S2 ** 2, levels=[-3, -2, -1, -0.3, 0.3, 1, 2, 3],
               cmap="coolwarm", linewidths=1.1)
b.plot(0, 0, "o", color="black", ms=7)
b.text(0.1, 0.15, "(0; 0)\n∇f = 0", fontsize=10)
b.text(-1.9, 1.65, "H = diag(2, −2):  det −4 < 0", fontsize=10.5,
       bbox=dict(fc="white", ec="none", alpha=0.85))
b.set_aspect("equal")
b.set_title("x₁² − x₂²: sedlo (nahoru po x₁, dolů po x₂)", fontsize=12)
b.set_xlabel("x₁")
b.set_ylabel("x₂")
b.grid(alpha=0.25)
uloz(fig, "tabule-bez-omezeni.svg")

# 2) nerovnost: aktivni (b = 4) a neaktivni (b = 7)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
kresli_limit(a, 4.0)
kresli_limit(b, 7.0)
b.text(2.6, 2.35, "∇f = 0", color=BARVA_OPT, bbox=dict(fc="white", ec="none", alpha=0.85))
uloz(fig, "tabule-kkt.svg")
