"""Obrázky pro blok 2 třetí přednášky: katalog kvadratických úloh.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_blok2.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from spolecne import (
    OUT,
    SVG_METADATA,
    CERVENA,
    MODRA,
    ORANZOVA,
    SVETLE_MODRA,
    TYRKYSOVA,
)

STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0}

# Ridge regrese se na slidu kreslí interaktivně (widget přímo v _fragment-1.qmd,
# resp. index.qmd); počítá s A = [[2,2],[1,0],[0,1]], b = (5,4,3) jako
# qp_priklady_cvxpy.py, kde se čísla ověřují v CVXPY.

# --- SVM: dvě třídy ve dvou příznacích --------------------------------------
Z_NEMOCNI = np.array([[3.0, 3.0], [5.0, 1.0], [4.0, 4.0], [2.5, 4.5], [4.0, 2.5], [6.0, 0.5]])
Z_ZDRAVI = np.array([[1.0, 1.0], [2.0, 0.0], [0.5, 1.0], [0.0, 1.0], [1.5, 0.0], [0.0, 0.0]])


def pekarna_cena() -> None:
    """Klesající marže na kus a rozdíl mezi lineárním a kvadratickým ziskem."""
    fig, osy = plt.subplots(1, 2, figsize=(10.8, 3.75))

    # --- vlevo: marže na kus klesá s množstvím ------------------------------
    cena = osy[0]
    mnozstvi = np.linspace(0, 200, 300)
    cena.plot(mnozstvi, 14 - 0.025 * mnozstvi, color=TYRKYSOVA, linewidth=3.4)
    cena.plot(mnozstvi, 8 - 0.024 * mnozstvi, color=ORANZOVA, linewidth=3.4)
    cena.text(196, 9.75, "chléb $p_1$", color=TYRKYSOVA, fontsize=13.5, weight="bold",
              ha="right", va="bottom")
    cena.text(196, 3.95, "bageta $p_2$", color="#9a6200", fontsize=13.5, weight="bold",
              ha="right", va="bottom")
    for kusy, kc, barva in ((120.0, 11.0, TYRKYSOVA), (100.0, 5.6, ORANZOVA)):
        cena.plot([kusy, kusy, 0], [0, kc, kc], color=barva, linewidth=1.4, linestyle=":",
                  zorder=1)
        cena.scatter([kusy], [kc], s=190, marker="*", color=CERVENA, zorder=5,
                     edgecolor="white", linewidth=0.8)
    cena.annotate("120 ks $\\rightarrow$ 11 Kč", xy=(120, 11), xytext=(96, 14.9),
                  arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4},
                  color=CERVENA, fontsize=13, weight="bold", va="center", bbox=STITEK)
    cena.annotate("100 ks $\\rightarrow$ 5,60 Kč", xy=(100, 5.6), xytext=(22, 1.7),
                  arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4},
                  color=CERVENA, fontsize=13, weight="bold", va="center", bbox=STITEK)
    cena.annotate("v nule přesně\nčísla z minula", xy=(0, 14), xytext=(18, 15.6),
                  arrowprops={"arrowstyle": "->", "color": MODRA, "lw": 1.2},
                  color=MODRA, fontsize=12, va="center", style="italic",
                  linespacing=1.05)
    cena.set(xlabel="vyrobené množství [ks/den]", ylabel="marže [Kč/ks]",
             xlim=(0, 200), ylim=(0, 17.4), yticks=[0, 4, 8, 12, 16])
    cena.set_title("Čím víc napečeme, tím víc slevíme", fontsize=14, color=MODRA,
                   weight="bold")

    # --- vpravo: zisk z chleba ----------------------------------------------
    zisk = osy[1]
    chleby = np.linspace(0, 200, 300)
    zisk.plot(chleby, 14 * chleby, color=MODRA, linewidth=2.2, linestyle="--", alpha=0.8)
    zisk.plot(chleby, 14 * chleby - 0.025 * chleby ** 2, color=TYRKYSOVA, linewidth=3.4)
    zisk.text(150, 2790, "pevná marže: přímka", color=MODRA, fontsize=12.5,
              ha="right", va="top")
    zisk.text(196, 1560, "se slevou:\nparabola", color=TYRKYSOVA, fontsize=12.5,
              weight="bold", ha="right", va="top", linespacing=1.05)
    zisk.fill_between(chleby, 14 * chleby - 0.025 * chleby ** 2, 14 * chleby,
                      color=CERVENA, alpha=0.10, linewidth=0)
    zisk.plot([120, 120], [1320, 1680], color=CERVENA, linewidth=2.6, zorder=4)
    zisk.scatter([120, 120], [1320, 1680], s=56, color=CERVENA, zorder=5)
    zisk.annotate("sleva ubere\n$0{,}025\\cdot120^2=360$ Kč", xy=(118, 1500),
                  xytext=(8, 2280),
                  arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4},
                  color=CERVENA, fontsize=13, weight="bold", va="center", bbox=STITEK,
                  linespacing=1.1)
    zisk.set(xlabel="chleby $x_1$ [ks/den]", ylabel="zisk z chleba [Kč/den]",
             xlim=(0, 200), ylim=(0, 2950))
    zisk.set_title("Zisk už není přímka, ale parabola", fontsize=14, color=MODRA,
                   weight="bold")

    for osa in osy:
        osa.grid(alpha=0.22)
        osa.tick_params(labelsize=11.5)
        osa.xaxis.label.set_size(12.5)
        osa.yaxis.label.set_size(12.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "pekarna-cena.svg", metadata=SVG_METADATA)
    _nahled(fig, "pekarna-cena")
    plt.close(fig)


def svm_odstup() -> None:
    """Dělicí přímka s největším odstupem a support vektory."""
    w = np.array([0.5, 0.5])
    b0 = -2.0
    odstup = 2.0 / np.linalg.norm(w)

    fig, osa = plt.subplots(figsize=(6.3, 5.0))
    t = np.linspace(-1.2, 7.4, 50)
    osa.fill_between(t, (-b0 + 1 - w[0] * t) / w[1], (-b0 - 1 - w[0] * t) / w[1],
                     color=SVETLE_MODRA, alpha=0.85, linewidth=0, zorder=0)
    osa.plot(t, (-b0 - w[0] * t) / w[1], color=MODRA, linewidth=3.2, zorder=3)
    for znak in (-1.0, 1.0):
        osa.plot(t, (-b0 + znak - w[0] * t) / w[1], color=MODRA, linewidth=1.8,
                 linestyle="--", alpha=0.8, zorder=3)
    osa.text(0.95, 3.35, "dělicí přímka", color=MODRA, fontsize=13, weight="bold",
             rotation=-45, rotation_mode="anchor", ha="center", va="center",
             bbox=STITEK, zorder=4)

    osa.scatter(Z_ZDRAVI[:, 0], Z_ZDRAVI[:, 1], s=120, color=TYRKYSOVA, zorder=5,
                edgecolor="white", linewidth=1.0)
    osa.scatter(Z_NEMOCNI[:, 0], Z_NEMOCNI[:, 1], s=120, marker="s", color=ORANZOVA,
                zorder=5, edgecolor="white", linewidth=1.0)
    osa.text(-0.85, -0.68, "zdraví, $y_i=-1$", color=TYRKYSOVA, fontsize=13.5,
             weight="bold", ha="left", va="center", zorder=6)
    osa.text(7.05, 5.05, "nemocní, $y_i=+1$", color="#8a5a00", fontsize=13.5,
             weight="bold", ha="right", va="center", zorder=6)

    for bod in ((1.0, 1.0), (2.0, 0.0), (3.0, 3.0), (5.0, 1.0)):
        osa.scatter(*bod, s=420, facecolor="none", edgecolor=CERVENA, linewidth=2.4,
                    zorder=6)
    osa.text(7.05, 3.55, "support vektory\n= aktivní omezení", color=CERVENA,
             fontsize=13, weight="bold", ha="right", va="center", bbox=STITEK, zorder=8,
             linespacing=1.1)
    for cil in ((5.0, 1.0), (3.0, 3.0)):
        osa.annotate("", xy=cil, xytext=(5.6, 3.1),
                     arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4,
                                 "shrinkB": 9}, zorder=8)

    osa.add_patch(FancyArrowPatch((1.0, 1.0), (3.0, 3.0), arrowstyle="<|-|>",
                                  mutation_scale=16, linewidth=2.4, color=CERVENA, zorder=7))
    osa.text(2.55, 1.55, f"odstup $2/\\|\\mathbf{{w}}\\|={odstup:.2f}$".replace(".", "{,}"),
             color=CERVENA, fontsize=13, weight="bold", ha="left", va="center",
             bbox=STITEK, zorder=8)

    osa.set(xlim=(-1.0, 7.2), ylim=(-1.0, 5.4), aspect="equal")
    osa.set_xlabel("příznak $z_1$", fontsize=12.5)
    osa.set_ylabel("příznak $z_2$", fontsize=12.5)
    osa.grid(alpha=0.22)
    osa.tick_params(labelsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "svm-odstup.svg", metadata=SVG_METADATA)
    _nahled(fig, "svm-odstup")
    plt.close(fig)


def _nahled(fig, jmeno: str) -> None:
    """Uloží PNG náhled do adresáře v proměnné NAHLEDY (jen pro vizuální kontrolu)."""
    kam = os.environ.get("NAHLEDY")
    if not kam:
        return
    cesta = Path(kam)
    cesta.mkdir(parents=True, exist_ok=True)
    fig.savefig(cesta / f"{jmeno}.png", dpi=130)


if __name__ == "__main__":
    pekarna_cena()
    svm_odstup()
