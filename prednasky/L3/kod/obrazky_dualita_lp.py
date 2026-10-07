"""Dualita v praxi: duální mezera jako zastavovací pravidlo a duál LP pekárny.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_dualita_lp.py

Dva obrázky:

* ``dualni-mezera.gif`` — bariérová metoda na dávkové úloze
  min (x - 3)^2 za x <= 2 (p* = 1, d(mu) = mu - mu^2/4, mu* = 2). Pro parametr
  t řeší solver  min (x - 3)^2 - (1/t) log(2 - x);  z jeho řešení x_t se dá
  zadarmo vyčíst cena mu_t = 1/(t (2 - x_t)) a duální mez d(mu_t). Mezera mezi
  primárem a duálem vyjde přesně 1/t, takže solver ví, kdy přestat.
* ``lp-dual-pekarna.svg`` — tabulka LP pekárny z druhé přednášky: primár čte
  řádky matice A, duál čte sloupce; obě úlohy dají 2600 Kč.
* ``lagrangian-sedlo.svg`` — vrstevnice lagrangiánu dávky
  L(x, mu) = (x - 3)^2 + mu (x - 2) jako hry: pekař volí x a minimalizuje,
  kontrolor volí cenu mu >= 0 a maximalizuje. Sedlo v (2; 2) s hodnotou 1 je
  zároveň optimum primáru i duálu.

Volitelně ``OMM_NAHLEDY=/cesta`` uloží PNG náhledy mimo repozitář.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import brentq

from spolecne import (
    A_PEKARNA,
    B_PEKARNA,
    CERVENA,
    MODRA,
    ORANZOVA,
    OUT,
    P_PEKARNA,
    SEDIVA,
    SVETLE_MODRA,
    SVG_METADATA,
    TYRKYSOVA,
    X_LP,
    uloz_gif,
)

plt.rcParams["figure.max_open_warning"] = 0   # snímky GIFu se zavírají až po uložení
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8}
SEDA = "#5c6672"
TOLERANCE = 1e-2
Y_LP = (10.0, 3.0)     # stínové ceny LP pekárny: mouka [Kč/kg], pec [Kč/min]


def _cislo(hodnota: float, mist: int = 3) -> str:
    return f"{hodnota:.{mist}f}".replace(".", ",")


def _nahled(fig: plt.Figure, jmeno: str) -> None:
    """Uloží PNG náhled, pokud je nastavená proměnná OMM_NAHLEDY."""
    nahledy = os.environ.get("OMM_NAHLEDY")
    if nahledy:
        Path(nahledy).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(nahledy) / f"{jmeno}.png", dpi=130)


# --- bariérová metoda na dávkové úloze ---
def dualni_davka(mu):
    """Duální funkce d(mu) = mu - mu^2/4 dávkové úlohy."""
    return mu - mu**2 / 4


def barierovy_krok(t: float) -> tuple[float, float, float, float]:
    """Řešení bariérové úlohy pro parametr t: vrací x_t, cenu mu_t, primár a duál."""
    x = brentq(lambda x: 2 * (x - 3) + 1 / (t * (2 - x)), -5, 2 - 1e-12, xtol=1e-14)
    mu = 1 / (t * (2 - x))
    return x, mu, (x - 3) ** 2, dualni_davka(mu)


def _kontrola_mezery() -> None:
    """Mezera primár minus duál je přesně 1/t; kontrolní hodnoty pro t = 1 a t = 10."""
    for t in np.geomspace(0.3, 100, 45):
        _, _, f, d = barierovy_krok(t)
        assert abs((f - d) - 1 / t) < 1e-9, (t, f - d)
    x, mu, f, d = barierovy_krok(1.0)
    assert abs(x - 1.634) < 1e-3 and abs(mu - 2.732) < 1e-3
    assert abs(f - 1.866) < 1e-3 and abs(d - 0.866) < 1e-3
    _, _, f, d = barierovy_krok(10.0)
    assert abs(f - 1.098) < 1e-3 and abs(d - 0.998) < 1e-3


def dualni_mezera_gif() -> None:
    """Vlevo primár a duál se svírají k p* = 1, vpravo mezera 1/t klesá pod toleranci."""
    hodnoty_t = np.geomspace(0.3, 100, 45)
    kroky = [barierovy_krok(t) for t in hodnoty_t]
    mu_osa = np.linspace(0, 4, 300)
    t_osa = np.geomspace(0.3, 300, 200)

    def snimek(i: int) -> plt.Figure:
        t = hodnoty_t[i]
        _, mu, f, d = kroky[i]
        mezera = f - d

        fig, (osa, osa_m) = plt.subplots(1, 2, figsize=(10.0, 4.5),
                                         gridspec_kw={"width_ratios": (1.15, 1.0)})
        fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.14, wspace=0.22)
        fig.suptitle(f"t = {_cislo(t, 1)}:   primár {_cislo(f)}   ·   duál {_cislo(d)}"
                     f"   ·   mezera {_cislo(mezera)}",
                     fontsize=15, color=MODRA, weight="bold", x=0.07, ha="left", y=0.97)

        # --- ceny: parabola d(mu), pod p* = 1, nad ní primár ---
        osa.axhline(1, color=ORANZOVA, linewidth=2.0, linestyle="--", zorder=2)
        osa.text(0.08, 1.07, "$p^{*}=1$", color="#9a6200", fontsize=14, weight="bold",
                 ha="left", va="bottom", zorder=6, bbox=STITEK)
        osa.plot(mu_osa, dualni_davka(mu_osa), color=TYRKYSOVA, linewidth=2.6, zorder=3)
        osa.text(0.1, 3.8, "$d(\\mu)=\\mu-\\mu^2/4$", color=TYRKYSOVA, fontsize=14,
                 ha="left", va="top", zorder=6)
        for _, mu_s, f_s, d_s in kroky[:i]:
            osa.scatter([mu_s], [d_s], s=28, color=TYRKYSOVA, alpha=0.25, zorder=4,
                        edgecolor="none")
            osa.scatter([mu_s], [f_s], s=28, color=CERVENA, alpha=0.25, zorder=4,
                        edgecolor="none")
        osa.plot([mu, mu], [d, f], color="#8a97a0", linewidth=2.2, zorder=5)
        osa.scatter([mu], [d], s=110, color=TYRKYSOVA, edgecolor="white", linewidth=1.3,
                    zorder=7)
        osa.scatter([mu], [f], s=110, color=CERVENA, edgecolor="white", linewidth=1.3,
                    zorder=7)
        if mezera > 0.6:   # body jsou daleko od sebe: popisky přímo u nich
            osa.text(mu - 0.1, f, "primár $f(x_t)$", color=CERVENA, fontsize=14,
                     weight="bold", ha="right", va="center", zorder=8, bbox=STITEK)
            osa.text(mu - 0.1, d - 0.12, "duál $d(\\mu_t)$", color=TYRKYSOVA, fontsize=14,
                     weight="bold", ha="right", va="top", zorder=8, bbox=STITEK)
            osa.text(mu + 0.07, (f + d) / 2, "mezera\n= 1/t", color=SEDA, fontsize=13,
                     ha="left", va="center", zorder=8, bbox=STITEK)
        else:              # body se slévají: popisky stranou se šipkou
            sipka = {"arrowstyle": "->", "linewidth": 1.4}
            osa.annotate("mezera = 1/t", xy=(mu, (f + d) / 2), xytext=(1.75, 2.45),
                         color=SEDA, fontsize=13, ha="right", va="center", zorder=8,
                         bbox=STITEK, arrowprops={**sipka, "color": SEDA})
            osa.annotate("primár $f(x_t)$", xy=(mu, f), xytext=(1.75, 1.8),
                         color=CERVENA, fontsize=14, weight="bold", ha="right",
                         va="center", zorder=8, bbox=STITEK,
                         arrowprops={**sipka, "color": CERVENA})
            osa.annotate("duál $d(\\mu_t)$", xy=(mu, d), xytext=(1.75, 0.3),
                         color=TYRKYSOVA, fontsize=14, weight="bold", ha="right",
                         va="center", zorder=8, bbox=STITEK,
                         arrowprops={**sipka, "color": TYRKYSOVA})
        osa.set_title("ceny: z $x_t$ zadarmo $\\mu_t = 1/(t\\,(2-x_t))$",
                      fontsize=14, color=MODRA, loc="left", pad=6)
        osa.set(xlim=(0, 4.6), ylim=(-0.3, 4.0))
        osa.set_xlabel("cena $\\mu$", fontsize=14)
        osa.tick_params(labelsize=12)
        osa.grid(alpha=0.2)

        # --- kdy přestat: mezera 1/t v log-log ---
        osa_m.axhspan(1e-3, TOLERANCE, color=SVETLE_MODRA, zorder=0)
        osa_m.axhline(TOLERANCE, color=ORANZOVA, linewidth=2.0, linestyle="--", zorder=2)
        osa_m.text(0.36, TOLERANCE * 1.15, "tolerance $10^{-2}$", color="#9a6200",
                   fontsize=13, weight="bold", ha="left", va="bottom", zorder=6)
        osa_m.plot(t_osa, 1 / t_osa, color="#8a97a0", linewidth=2.0, zorder=3)
        osa_m.text(1.1, 1.6, "mezera = 1/t", color=SEDA, fontsize=13, ha="left",
                   va="bottom", zorder=6)
        osa_m.scatter(hodnoty_t[:i], 1 / hodnoty_t[:i], s=24, color=CERVENA, alpha=0.25,
                      edgecolor="none", zorder=4)
        osa_m.scatter([t], [mezera], s=110, color=CERVENA, edgecolor="white",
                      linewidth=1.3, zorder=7)
        if mezera <= TOLERANCE * (1 + 1e-9):
            osa_m.text(0.36, 1.9e-3, "mezera ≤ tolerance\n→ solver končí", color=CERVENA,
                       fontsize=14, weight="bold", ha="left", va="bottom", zorder=8,
                       bbox=STITEK)
        osa_m.set_xscale("log")
        osa_m.set_yscale("log")
        osa_m.set_title("kdy přestat", fontsize=14, color=MODRA, loc="left", pad=6)
        osa_m.set(xlim=(0.3, 300), ylim=(1e-3, 5))
        osa_m.set_xlabel("parametr bariéry $t$", fontsize=14)
        osa_m.tick_params(labelsize=12)
        osa_m.grid(alpha=0.2, which="major")
        return fig

    posledni = len(hodnoty_t) - 1
    prvni = snimek(posledni)
    _nahled(prvni, "dualni-mezera")
    snimky, doby = [prvni], [2000]
    for i in range(len(hodnoty_t)):
        snimky.append(snimek(i))
        doby.append(160)
    doby[-1] = 3000
    uloz_gif(snimky, doby, "dualni-mezera", dpi=100)


# --- primár a duál LP pekárny ---
def _kontrola_lp() -> None:
    """A x* <= b s rovností v obou řádcích, A^T y* = c a c.x* = b.y* = 2600."""
    A, b, c = np.array(A_PEKARNA), np.array(B_PEKARNA), np.array(P_PEKARNA)
    x, y = np.array(X_LP), np.array(Y_LP)
    assert np.allclose(A @ x, b)
    assert np.allclose(A.T @ y, c)
    assert abs(c @ x - 2600) < 1e-9 and abs(b @ y - 2600) < 1e-9


def lp_dual_pekarna() -> None:
    """Tabulka A, b, c pekárny: primár čte řádky, duál sloupce."""
    fig = plt.figure(figsize=(7.0, 4.3))
    osa = fig.add_axes((0, 0, 1, 1))
    osa.set(xlim=(0, 700), ylim=(0, 430))
    osa.axis("off")

    sloupce = {"chléb": 380, "bageta": 490, "b": 615}   # středy sloupců
    radky = {"mouka": 300, "pec": 250, "c": 192}         # středy řádků
    sirka, vyska = 100, 42

    # buňky matice, sloupec b a řádek c
    for r in ("mouka", "pec"):
        for s in ("chléb", "bageta", "b"):
            osa.add_patch(Rectangle((sloupce[s] - sirka / 2, radky[r] - vyska / 2),
                                    sirka, vyska, facecolor="white" if s != "b" else SEDIVA,
                                    edgecolor="#b5c0c8", linewidth=1.0, zorder=1))
    for s in ("chléb", "bageta"):
        osa.add_patch(Rectangle((sloupce[s] - sirka / 2, radky["c"] - vyska / 2),
                                sirka, vyska, facecolor=SEDIVA, edgecolor="#b5c0c8",
                                linewidth=1.0, zorder=1))

    hodnoty = {("mouka", "chléb"): "0,5", ("mouka", "bageta"): "0,2", ("mouka", "b"): "80",
               ("pec", "chléb"): "3", ("pec", "bageta"): "2", ("pec", "b"): "600",
               ("c", "chléb"): "14", ("c", "bageta"): "8"}
    for (r, s), text in hodnoty.items():
        osa.text(sloupce[s], radky[r], text, fontsize=16, color="#1d2730",
                 ha="center", va="center", zorder=4)

    # záhlaví sloupců a nad nimi optimální plán x*
    for s in ("chléb", "bageta"):
        osa.text(sloupce[s], 350, s, fontsize=15, weight="bold", color="#1d2730",
                 ha="center", va="center", zorder=4)
    osa.text(sloupce["b"], 350, "zásoba $\\mathbf{b}$", fontsize=14, color=SEDA, ha="center",
             va="center", zorder=4)
    osa.text(sloupce["chléb"], 378, "$x_1^{\\star}=100$", fontsize=15, color=MODRA,
             weight="bold", ha="center", va="center", zorder=4)
    osa.text(sloupce["bageta"], 378, "$x_2^{\\star}=150$", fontsize=15, color=MODRA,
             weight="bold", ha="center", va="center", zorder=4)
    osa.text(312, 378, "plán [ks/den]", fontsize=13, color=MODRA, ha="right",
             va="center", zorder=4)

    # názvy řádků a před nimi stínové ceny y*
    osa.text(312, radky["mouka"], "mouka [kg]", fontsize=14, weight="bold",
             color="#1d2730", ha="right", va="center", zorder=4)
    osa.text(312, radky["pec"], "pec [min]", fontsize=14, weight="bold",
             color="#1d2730", ha="right", va="center", zorder=4)
    osa.text(312, radky["c"], "zisk $\\mathbf{c}$ [Kč/ks]", fontsize=13, color=SEDA,
             ha="right", va="center", zorder=4)
    osa.text(18, radky["mouka"], "$y_1^{\\star}=10$ Kč/kg", fontsize=14, color=TYRKYSOVA,
             ha="left", va="center", zorder=4)
    osa.text(18, radky["pec"], "$y_2^{\\star}=3$ Kč/min", fontsize=14, color=TYRKYSOVA,
             ha="left", va="center", zorder=4)

    # zvýraznění: řádek mouky (primár) a sloupec chleba (duál)
    osa.add_patch(Rectangle((sloupce["chléb"] - sirka / 2 - 6, radky["mouka"] - vyska / 2 - 5),
                            sloupce["b"] - sloupce["chléb"] + sirka + 12, vyska + 10,
                            facecolor="none", edgecolor=MODRA, linewidth=3.0, zorder=6))
    osa.add_patch(Rectangle((sloupce["chléb"] - sirka / 2 - 12, radky["c"] - vyska / 2 - 9),
                            sirka + 24, radky["mouka"] - radky["c"] + vyska + 18,
                            facecolor="none", edgecolor=TYRKYSOVA, linewidth=3.0,
                            linestyle=(0, (5, 2.5)), zorder=7))

    osa.text(18, 128, "primár čte řádky:   0,5 x₁ + 0,2 x₂ ≤ 80", fontsize=15,
             color=MODRA, weight="bold", ha="left", va="center", zorder=4)
    osa.text(18, 92, "duál čte sloupce:   0,5 y₁ + 3 y₂ ≥ 14", fontsize=15,
             color=TYRKYSOVA, weight="bold", ha="left", va="center", zorder=4)
    osa.text(18, 40, "14·100 + 8·150 = 2600 Kč = 80·10 + 600·3", fontsize=14,
             color=SEDA, ha="left", va="center", zorder=4)

    fig.savefig(OUT / "lp-dual-pekarna.svg", metadata=SVG_METADATA)
    _nahled(fig, "lp-dual-pekarna")
    plt.close(fig)


# --- lagrangián jako hra: sedlový bod ---
def lagr_davka(x, mu):
    """Lagrangián dávkové úlohy L(x, mu) = (x - 3)^2 + mu (x - 2)."""
    return (x - 3.0) ** 2 + mu * (x - 2.0)


def _kontrola_sedla() -> None:
    """(2; 2) je sedlo: v x minimum, v mu (lineárně) maximum na x = 2, hodnota 1."""
    xs = np.linspace(0, 4, 4001)
    mus = np.linspace(0, 4, 4001)
    assert abs(xs[np.argmin(lagr_davka(xs, 2.0))] - 2.0) < 1e-3
    assert np.allclose(lagr_davka(2.0, mus), 1.0)          # na limitu se neplatí nic
    assert abs(lagr_davka(2.0, 2.0) - 1.0) < 1e-12
    # max-min = min-max = 1: d(mu) = mu - mu^2/4 má maximum 1 v mu = 2
    assert abs(np.max(dualni_davka(mus)) - 1.0) < 1e-6


def lagrangian_sedlo() -> None:
    """Vrstevnice L(x, mu): pekař minimalizuje přes x, kontrolor maximalizuje přes mu."""
    x = np.linspace(0.0, 4.0, 401)
    mu = np.linspace(0.0, 4.0, 401)
    X, M = np.meshgrid(x, mu)
    L = lagr_davka(X, M)

    fig, osa = plt.subplots(figsize=(6.4, 5.0))
    osa.axvspan(2.0, 4.0, color="#fbe3dc", alpha=0.7, linewidth=0, zorder=0)
    urovne = [-1.0, 0.0, 0.5, 1.5, 2.0, 3.0, 4.0, 6.0]   # L = 1 kreslí modrá
    cary = osa.contour(X, M, L, levels=urovne, colors=SEDA, linewidths=1.0, zorder=1)
    osa.clabel(cary, fmt=lambda v: _cislo(v, 1).rstrip("0").rstrip(","), fontsize=10)
    osa.contour(X, M, L, levels=[1.0], colors=MODRA, linewidths=2.4, zorder=2)

    # nejlepší odpověď pekaře na cenu mu: x(mu) = 3 - mu/2
    osa.plot(3 - mu / 2, mu, color=TYRKYSOVA, linewidth=3.0, linestyle="--", zorder=3)
    osa.text(2.62, 3.55, "lékař: nejlepší $x$\npři ceně $\\mu$ → $d(\\mu)$",
             color=TYRKYSOVA, fontsize=12.5, weight="bold", ha="left", va="center",
             bbox=STITEK, zorder=5)
    # kontrolor: za porušený limit (x > 2) zvedne cenu nade všechny meze
    osa.annotate("", xy=(3.35, 3.0), xytext=(3.35, 1.0),
                 arrowprops={"arrowstyle": "-|>", "color": CERVENA, "lw": 2.4,
                             "mutation_scale": 18}, zorder=4)
    osa.text(3.45, 1.25, "x > 2: kontrolor\nzvedá cenu,\nL → ∞", color=CERVENA,
             fontsize=12, weight="bold", ha="left", va="bottom", bbox=STITEK, zorder=5)
    osa.text(0.15, 0.25, "x < 2: kontrolorovi se\nvyplatí jen μ = 0",
             color=SEDA, fontsize=12, ha="left", va="bottom", bbox=STITEK, zorder=5)

    osa.plot([2.0], [2.0], marker="*", markersize=24, color=CERVENA,
             markeredgecolor="white", zorder=6)
    osa.text(0.35, 3.42, "$\\mathcal{L}=1$", color=MODRA, fontsize=12.5, weight="bold",
             ha="left", va="center", bbox=STITEK, zorder=5)
    osa.annotate("", xy=(2.0, 2.0), xytext=(1.15, 2.55),
                 arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.6,
                             "shrinkB": 12}, zorder=6)
    osa.text(0.12, 2.72, "sedlo: $x^\\star=2$, $\\mu^\\star=2$\n$\\mathcal{L}=1=p^\\star=d^\\star$",
             color=CERVENA, fontsize=12.5, weight="bold", ha="left", va="center",
             bbox=STITEK, zorder=6)

    osa.set(xlim=(0, 4), ylim=(0, 4))
    osa.set_xlabel("dávka $x$ [mg] — volí lékař (min)", fontsize=12.5)
    osa.set_ylabel("cena $\\mu$ — volí kontrolor (max)", fontsize=12.5)
    osa.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "lagrangian-sedlo.svg", metadata=SVG_METADATA)
    _nahled(fig, "lagrangian-sedlo")
    plt.close(fig)


if __name__ == "__main__":
    _kontrola_mezery()
    dualni_mezera_gif()
    _kontrola_lp()
    lp_dual_pekarna()
    _kontrola_sedla()
    lagrangian_sedlo()
