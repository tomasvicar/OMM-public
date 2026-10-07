"""Cisla a obrazky k cervenemu praporku tretiho cviceni: zaporny podil oriznuty na nulu.

    uv run python cviceni/C3/kod/praporek_orezani.py

Slozeni vzorku krve ze 4 typu bunek (T-lymfocyty, NK bunky, monocyty,
neutrofily) z exprese 6 genu. Exprese smesi je vazeny prumer expresi cistych
typu, vahami jsou podily bunek:

    minimize   f(x) = ||A x - b||_2^2
    subject to x >= 0,  sum(x) = 1

Kolega (cerveny_praporek_orezani.py) spocital nejmensi ctverce bez omezeni,
zaporny podil nastavil na 0 a zbytek prenormoval. Bod je PRIPUSTNY, ale ne
optimalni. Dukaz bez solveru je KKT stacionarita

    grad f + lambda * 1 - mu = 0,   mu >= 0,   mu_i x_i = 0,

tedy na nenulovych slozkach musi mit grad f = 2 A^T (A x - b) STEJNOU hodnotu
(= -lambda), na nulovych muze byt jen vetsi (o mu_i).

Skript je jediny zdroj cisel pro report, notebook, tahak i data hriste.
Data hriste (A, b, bod kolegy) vlozi jako JSON do reportu reporty/praporek.html
mezi znacky <!-- praporek-data:zacatek --> a <!-- praporek-data:konec -->.
Vystupy:
    cviceni/C3/obrazky/praporek-orezani-podily.svg     LS / kolega / optimum
    cviceni/C3/obrazky/praporek-orezani-gradient.svg   gradient u kolegy a v optimu
"""

import json
import re
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

TYPY = ["T-lymfocyty", "NK buňky", "monocyty", "neutrofily"]
GENY = ["CD3E", "NKG7", "GNLY", "CD14", "FCGR3B", "LYZ"]
A = np.array([[100, 10, 1, 1],
              [40, 120, 2, 2],
              [20, 80, 1, 1],
              [2, 2, 150, 10],
              [1, 20, 5, 200],
              [5, 5, 300, 120]], dtype=float)
b = np.array([45, 10, 2, 30, 100, 105], dtype=float)

OUT = Path(__file__).resolve().parents[1] / "obrazky"
REPORT = Path(__file__).resolve().parents[1] / "reporty" / "praporek.html"
plt.rcParams["svg.hashsalt"] = "mpc-omm-c3-praporek-orezani"
MODRA, TYRKYS, CERVENA, SEDA = "#12355b", "#007f86", "#c73e1d", "#8a949b"


def f(x):
    return float(np.sum((A @ x - b) ** 2))


def grad(x):
    return 2 * A.T @ (A @ x - b)


def nejmensi_ctverce():
    return np.linalg.lstsq(A, b, rcond=None)[0]


def kolega():
    x = np.clip(nejmensi_ctverce(), 0, None)
    return x / x.sum()


def optimum():
    x = cp.Variable(4)
    omezeni = [x >= 0, cp.sum(x) == 1]
    uloha = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), omezeni)
    uloha.solve(solver=cp.CLARABEL)
    # CVXPY: Lagrangian f + lambda (sum x - 1) - mu^T x, tedy grad f + lambda 1 - mu = 0
    # (znaceni treti prednasky: lambda u rovnosti, mu u nerovnosti)
    return x.value, uloha.value, omezeni[0].dual_value, float(omezeni[1].dual_value)


def vloz_do_reportu(obsah: dict):
    """Prepise JSON mezi znackami v reportu (report jinak pise clovek)."""
    if not REPORT.exists():
        print(f"report {REPORT} neexistuje, JSON nevlozen")
        return
    html = REPORT.read_text(encoding="utf-8")
    blok = ('<!-- praporek-data:zacatek -->\n<script type="application/json" id="praporek-data">'
            + json.dumps(obsah, separators=(",", ":")) + "</script>\n<!-- praporek-data:konec -->")
    novy, pocet = re.subn(r"<!-- praporek-data:zacatek -->.*?<!-- praporek-data:konec -->",
                          lambda _: blok, html, flags=re.S)
    if pocet != 1:
        raise SystemExit("v reportu chybi znacky <!-- praporek-data:zacatek/konec -->")
    REPORT.write_text(novy, encoding="utf-8")
    print(f"JSON pro hriste vlozen do {REPORT}")


def cz(v, d=1):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def obrazek_podily(x_ls, x_kol, x_opt):
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    sirka, poz = 0.26, np.arange(4)
    rady = [(x_ls, "nejmenší čtverce bez omezení", SEDA),
            (x_kol, f"kolega: oříznuto a přenormováno, f = {cz(f(x_kol))}", CERVENA),
            (x_opt, f"optimum s omezeními, f = {cz(f(x_opt))}", TYRKYS)]
    for k, (x, popis, barva) in enumerate(rady):
        sl = ax.bar(poz + (k - 1) * sirka, 100 * x, sirka, color=barva, label=popis)
        for obd, v in zip(sl, 100 * x):
            ax.annotate(cz(v), (obd.get_x() + obd.get_width() / 2, max(v, 0)),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color=barva)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xticks(poz, TYPY)
    ax.set_ylabel("podíl buněk [%]")
    ax.set_ylim(-12, 78)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=2, fontsize=8.5, framealpha=0.95)
    ax.set_title("Oba nezáporné odhady splňují omezení, chyba f se liší skoro dvakrát",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "praporek-orezani-podily.svg", metadata={"Date": None})
    plt.close(fig)


def obrazek_gradient(x_kol, x_opt, lam, mu):
    fig, osy = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    poz = np.arange(4)
    for ax, x, nazev in [(osy[0], x_kol, "kolega"), (osy[1], x_opt, "optimum")]:
        g = grad(x)
        volne = x > 1e-6
        barvy = [MODRA if v else SEDA for v in volne]
        sl = ax.bar(poz, g, 0.6, color=barvy)
        for obd, v in zip(sl, g):
            ax.annotate(cz(v, 0), (obd.get_x() + obd.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="#374151", lw=0.8)
        ax.set_xticks(poz, TYPY, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        rozptyl = np.ptp(g[volne])
        ax.set_title(f"{nazev}: rozdíl na nenulových složkách {cz(rozptyl, 0)}", fontsize=10.5)
    osy[1].axhline(-lam, color=TYRKYS, ls="--", lw=1.4)
    osy[1].annotate(f"$-\\lambda$ = {cz(-lam, 0)}", (1, -lam), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top", fontsize=9.5,
                    color=TYRKYS)
    i0 = int(np.argmax(mu))
    osy[1].annotate("", (i0 + 0.4, grad(x_opt)[i0]), (i0 + 0.4, -lam),
                    arrowprops=dict(arrowstyle="<|-|>", color="#374151", lw=1.1,
                                    shrinkA=0, shrinkB=0))
    osy[1].annotate(f"$\\mu$ = {cz(mu[i0], 0)}\n(podíl je 0, gradient\nsmí být větší)",
                    (i0 + 0.48, 0.62 * grad(x_opt)[i0]), fontsize=8.5,
                    color="#374151", va="center")
    osy[0].set_ylabel("složka gradientu ∇f")
    osy[0].set_ylim(-7600, 3600)
    fig.text(0.5, 0.01, "modře nenulové podíly (tam musí být gradient stejný), šedě podíl 0",
             ha="center", fontsize=9, color="#374151")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "praporek-orezani-gradient.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    x_ls = nejmensi_ctverce()
    x_kol = kolega()
    x_opt, f_opt, mu, lam = optimum()
    print(f"LS bez omezeni:  x = {x_ls}, soucet {x_ls.sum():.4f}, f = {f(x_ls):.2f}")
    print(f"kolega:          x = {x_kol}, soucet {x_kol.sum():.4f}, f = {f(x_kol):.2f}")
    print(f"optimum (QP):    x = {x_opt}, soucet {x_opt.sum():.4f}, f = {f_opt:.2f}")
    print(f"  kolega horsi o {100 * (f(x_kol) / f_opt - 1):.1f} %"
          f" (f kolegy / f* = {f(x_kol) / f_opt:.3f})")
    print(f"  rozdil v procentnich bodech: {100 * (x_kol - x_opt)}")
    print(f"  duály: mu (x >= 0) = {mu}, lambda = {lam:.2f}")
    print(f"grad f kolega:   {grad(x_kol)}")
    print(f"grad f optimum:  {grad(x_opt)}")
    g = grad(x_opt)
    print(f"  stacionarita grad f + lambda - mu = {g + lam - mu}")
    print(f"  na nulove slozce: grad - (-lambda) = {g[1] + lam:.2f} = mu_2")
    gk = grad(x_kol)
    volne = x_kol > 0
    print(f"  kolega: rozdil max - min na volnych slozkach = {np.ptp(gk[volne]):.0f}")
    # smer zlepseni: z nejvetsiho gradientu mezi nenulovymi (T) do nejmensiho (monocyty)
    d = np.zeros(4); d[0], d[2] = -0.01, 0.01
    print(f"  presun 1 p.b. z T do monocytu: odhad zmeny f {gk @ d:.2f}, "
          f"skutecne {f(x_kol + d) - f(x_kol):.2f}")
    d = np.zeros(4); d[0], d[3] = -0.01, 0.01
    print(f"  presun 1 p.b. z T do neutrofilu: odhad {gk @ d:.2f}, "
          f"skutecne {f(x_kol + d) - f(x_kol):.2f}")
    # stinova cena: vnutit NK bunkam podil aspon 1 %
    x = cp.Variable(4)
    vnuceno = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                         [x >= 0, cp.sum(x) == 1, x[1] >= 0.01])
    vnuceno.solve(solver=cp.CLARABEL)
    print(f"  NK aspon 1 %: f = {vnuceno.value:.2f}, narust {vnuceno.value - f_opt:.2f}"
          f" (odhad mu_2 * 0,01 = {0.01 * mu[1]:.2f})")
    print(f"  kolega: soucet po oriznuti {np.clip(x_ls, 0, None).sum():.4f} -> deli se jim")
    vloz_do_reportu({"A": A.astype(int).tolist(), "b": b.astype(int).tolist(),
                     "xKolega": np.round(x_kol, 12).tolist()})
    obrazek_podily(x_ls, x_kol, x_opt)
    obrazek_gradient(x_kol, x_opt, lam, mu)
    print("obrazky: cviceni/C3/obrazky/praporek-orezani-podily.svg, "
          "praporek-orezani-gradient.svg")
