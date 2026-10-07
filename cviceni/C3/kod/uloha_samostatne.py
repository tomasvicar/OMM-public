"""Samostatna uloha tretiho cviceni: dva leky a DRUHE omezeni.

    uv run python cviceni/C3/kod/uloha_samostatne.py

Uloha (hlavni priklad z cisla_leky.py + minimalni davka leku B):
    minimize   (x1 - 4)^2 + (x2 - 2)^2
    subject to g1 = x1 + x2 - 4 <= 0       (zatez ledvin)
               g2 = 1.5 - x2    <= 0       (lek B ucinkuje az od 1,5 mg)
    -> x* = (2,5; 1,5), f* = 2,5, mu = (3; 2), obe omezeni aktivni.

Resic a konstanty bere z cisla_leky.py; skript z nich pocita vypisy pro
notebook a report a kresli obrazek samostatne-kkt.svg.
"""

import matplotlib.pyplot as plt
import numpy as np

import cisla_leky
from cisla_leky import CIL, LIMIT, MIN_B


def vyres(limit=LIMIT, min_b=MIN_B):
    """Resic z cisla_leky.py s vychozim druhym omezenim x2 >= 1,5."""
    x, f, mu, dcp = cisla_leky.vyres(limit, min_b)
    return x, f, [np.round(np.atleast_1d(m), 6) for m in mu], dcp


def kkt(x, mu1, mu2):
    """Rezidua KKT pro hlavni ulohu; g_i ve tvaru <= 0."""
    g = np.array([x[0] + x[1] - LIMIT, MIN_B - x[1]])
    grad_f = 2 * (x - CIL)
    stac = grad_f + mu1 * np.array([1.0, 1.0]) + mu2 * np.array([0.0, -1.0])
    return g, grad_f, stac, np.array([mu1, mu2]) * g


if __name__ == "__main__":
    x, f, mu, dcp = vyres()
    print(f"hlavni: x* = {np.round(x, 6)}, f* = {f:.6f}, mu = {mu}, is_dcp = {dcp}")
    g, grad_f, stac, kompl = kkt(x, mu[0][0], mu[1][0])
    print(f"  g = {np.round(g, 6)}, grad f = {np.round(grad_f, 6)}")
    print(f"  stacionarita = {np.round(stac, 6)}, komplementarita = {np.round(kompl, 6)}")
    print("  CHYBA se znamenkem (g2 = x2 - 1,5, grad (0; 1)) -> stacionarita "
          f"{np.round(grad_f + mu[0][0] * np.array([1, 1]) + mu[1][0] * np.array([0, 1]), 3)}")

    print("\nstinove ceny prepoctem (zmirneni o 0,1):")
    _, f_b, _, _ = vyres(min_b=MIN_B - 0.1)
    _, f_l, _, _ = vyres(limit=LIMIT + 0.1)
    print(f"  x2 >= 1,4:      f = {f_b:.4f}, uspora {f - f_b:.4f} (odhad mu2*0,1 = 0,2)")
    print(f"  x1 + x2 <= 4,1: f = {f_l:.4f}, uspora {f - f_l:.4f} (odhad mu1*0,1 = 0,3)")

    print("\nposuvniky: jak se meni aktivni mnozina")
    for limit, min_b in [(4, 1.5), (4, 1.0), (4, 0.5), (5, 1.5), (6, 1.5), (7, 1.5), (7, 2.5), (3, 2.5)]:
        xx, ff, mm, _ = vyres(limit, min_b)
        print(f"  limit {limit}, min B {min_b}: x = {np.round(xx, 3)}, f = {ff:.3f}, "
              f"mu = ({mm[0][0]:.3f}; {mm[1][0]:.3f})")

    # ------------------------------------------------------------ obrazek ----
    plt.rcParams["svg.hashsalt"] = "mpc-omm-c3-samostatne"
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    X1, X2 = np.meshgrid(np.linspace(0, 5.6, 300), np.linspace(0, 3.6, 300))
    F = (X1 - CIL[0]) ** 2 + (X2 - CIL[1]) ** 2
    t = np.linspace(0, 5.6, 2)
    pripustne = (X1 + X2 <= LIMIT) & (X2 >= MIN_B)
    ax.contourf(X1, X2, pripustne, levels=[0.5, 1.5], colors=["#dbeafe"])
    ax.contour(X1, X2, F, levels=[0.5, 1, 2, 2.5, 4, 6, 8], colors="#9ca3af",
               linewidths=0.8, linestyles=":")
    ax.plot(t, LIMIT - t, color="#b45309", lw=2, label="ledviny $x_1+x_2=4$")
    ax.axhline(MIN_B, color="#7c3aed", lw=2, label="min. dávka B $x_2=1{,}5$")
    ax.plot(*CIL, "+", ms=12, mew=2, color="#374151")
    ax.annotate("ideální dávky (4; 2)", CIL, xytext=(6, 6), textcoords="offset points",
                fontsize=9, color="#374151")
    grad_f = 2 * (x - CIL)
    kusy = [(mu[0][0] * np.array([1.0, 1.0]), "#b45309"),
            (mu[1][0] * np.array([0.0, -1.0]), "#7c3aed")]
    # stejna konvence jako na tabuli: grad f a soucet mu_i grad g_i proti sobe
    s = 0.3                                        # sipky zkracene na 30 %
    ax.annotate("", x + s * grad_f, x, arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                                       color="#1a7f37"))
    konec = x.copy()
    for v, barva in kusy:                          # mu_i grad g_i hlavou k pate
        ax.annotate("", konec + s * v, konec, arrowprops=dict(
            arrowstyle="-|>", lw=1.8, color=barva, ls="--"))
        konec = konec + s * v
    ax.plot(*x, "o", ms=9, color="#1a7f37", zorder=5)
    ax.annotate(f"optimum (2,5; 1,5)\nf = {f:.4g}".replace(".", ","), x,
                xytext=(-4, -54), textcoords="offset points", ha="right",
                fontsize=10, color="#1a7f37")
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 3.6)
    ax.set_aspect("equal")
    ax.set_xlabel("lék A  $x_1$ [mg]")
    ax.set_ylabel("lék B  $x_2$ [mg]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    mu_txt = f"{mu[0][0]:g}; {mu[1][0]:g}".replace(".", ",")
    ax.set_title(f"obě omezení aktivní\n$\\mu$ = ({mu_txt})", fontsize=11)
    fig.text(0.5, 0.01, "zelená: $\\nabla f$,  čárkované: $\\mu_i\\nabla g_i$ složené za sebou;\n"
             "jejich součet míří přesně proti $\\nabla f$ (stacionarita); šipky zkrácené na 30 %",
             ha="center", fontsize=9.5, color="#374151")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("cviceni/C3/obrazky/samostatne-kkt.svg", metadata={"Date": None}, bbox_inches="tight")
    print("\nobrazek: cviceni/C3/obrazky/samostatne-kkt.svg")
