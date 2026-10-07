"""Animace duální funkce pekárny: cena mouky jako horní mez zisku.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_blok6_dualita.py

Omezení na mouku zrušíme a místo něj necháme pekaře mouku libovolně
dokupovat (nebo přebytek prodat) za cenu mu Kč/kg; cena pece zůstává nulová.
Pekař pak řeší úlohu bez omezení

    max_x  zisk(x) - mu * (0,5 x1 + 0,2 x2 - 80),

jejíž hodnota je -d(mu), tedy horní mez skutečného zisku 1880 Kč (pekárna
zisk maximalizuje, takže se dolní mez z přednášky otočí na horní). Nejlepší
mez dá mu = 16 Kč/kg — přesně multiplikátor mouky — a rovná se 1880 Kč.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from spolecne import (
    OUT,
    SVG_METADATA,
    ZISK_QP,
    MU_QP,
    uloz_gif,
    MODRA,
    CERVENA,
    ORANZOVA,
    SVETLE_MODRA,
    TYRKYSOVA,
    zisk,
)

plt.rcParams["figure.max_open_warning"] = 0   # snímky GIFu se zavírají až po uložení
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8}
ZELENA = "#3a8a52"
MU_MAX = 26.0          # pro větší ceny by relaxované x1 vyšlo záporné


def relaxovany_plan(mu: float) -> np.ndarray:
    """Plán, který maximalizuje zisk minus platbu za mouku (bez omezení)."""
    return np.array([(14 - 0.5 * mu) / 0.05, (8 - 0.2 * mu) / 0.048])


def mez(mu: float) -> float:
    """Horní mez zisku -d(mu) pro cenu mouky mu [Kč/kg]."""
    x = relaxovany_plan(mu)
    return float(zisk(*x) - mu * (0.5 * x[0] + 0.2 * x[1] - 80))


def _kontrola() -> None:
    """Silná dualita na pekárně: nejlepší mez leží v mu = 16 a rovná se 1880 Kč."""
    mu_osa = np.linspace(0, MU_MAX, 2601)
    meze = np.array([mez(m) for m in mu_osa])
    assert np.all(meze >= ZISK_QP - 1e-9)
    assert abs(mu_osa[meze.argmin()] - MU_QP[0]) < 0.02
    assert abs(meze.min() - ZISK_QP) < 1e-6


def _cislo(hodnota: float, mist: int = 0) -> str:
    return f"{hodnota:.{mist}f}".replace(".", ",")


def dualita_cena_mouky_gif() -> None:
    """Posouvání ceny mouky: relaxovaný plán putuje a mez zisku se dotkne 1880 Kč."""
    mu_osa = np.linspace(0, MU_MAX, 300)
    meze = np.array([mez(m) for m in mu_osa])
    cesta = np.array([relaxovany_plan(m) for m in mu_osa])
    vrcholy = [[0, 0], [160, 0], [100, 150], [0, 300]]
    t = np.linspace(0, 320, 40)

    def snimek(mu: float) -> plt.Figure:
        x = relaxovany_plan(mu)
        spotreba = 0.5 * x[0] + 0.2 * x[1]
        hodnota = mez(mu)

        fig, (osa, osa_d) = plt.subplots(1, 2, figsize=(8.6, 4.0),
                                         gridspec_kw={"width_ratios": (1.0, 1.15)})
        fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.14, wspace=0.24)

        # --- rovina výroby ---
        osa.add_patch(Polygon(vrcholy, closed=True, facecolor=SVETLE_MODRA,
                              edgecolor="none", zorder=1))
        osa.plot(t, (80 - 0.5 * t) / 0.2, color=CERVENA, linewidth=2.4, zorder=3)
        osa.plot(t, (600 - 3 * t) / 2, color=ZELENA, linewidth=1.6, alpha=0.6, zorder=3)
        osa.plot(cesta[:, 0], cesta[:, 1], color="#8a97a0", linewidth=1.3,
                 linestyle=(0, (2, 2)), zorder=4)
        osa.scatter([120], [100], s=160, marker="*", color=CERVENA, edgecolor="white",
                    linewidth=1.0, zorder=5, alpha=0.55)
        osa.text(276, 172, "$\\mu=0$", color="#5c6672", fontsize=10, ha="right", va="bottom",
                 zorder=6)
        osa.text(24, 50, "$\\mu=26$", color="#5c6672", fontsize=10, ha="left", va="top",
                 zorder=6)
        osa.text(206, 34, "pec", color=ZELENA, fontsize=10.5, ha="left", va="center",
                 zorder=6, bbox=STITEK)
        osa.scatter(*x, s=110, color=TYRKYSOVA, edgecolor="white", linewidth=1.3, zorder=7)
        osa.text(166, 6, "mouka 80 kg", color=CERVENA, fontsize=10.5, weight="bold",
                 ha="left", va="bottom", zorder=8, bbox=STITEK)
        if spotreba > 80.05:
            obchod = f"dokoupí {_cislo(spotreba - 80)} kg"
        elif spotreba < 79.95:
            obchod = f"prodá {_cislo(80 - spotreba)} kg"
        else:
            obchod = "přesně zásoba"
        osa.set_title(f"pekař chce {_cislo(spotreba)} kg mouky → {obchod}",
                      fontsize=10.5, color=TYRKYSOVA, weight="bold", loc="left", pad=5)
        osa.set(xlim=(0, 300), ylim=(0, 220), xlabel="chleby $x_1$", ylabel="bagety $x_2$")
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.2)

        # --- mez zisku jako funkce ceny ---
        osa_d.fill_between(mu_osa, ZISK_QP, meze, color=SVETLE_MODRA, zorder=1)
        osa_d.plot(mu_osa, meze, color=TYRKYSOVA, linewidth=2.6, zorder=3)
        osa_d.axhline(ZISK_QP, color=CERVENA, linewidth=1.8, linestyle="--", zorder=2)
        osa_d.text(0.4, ZISK_QP - 18, "skutečný nejvyšší zisk 1880 Kč", color=CERVENA,
                   fontsize=10.5, weight="bold", va="top", zorder=6)
        osa_d.plot([mu, mu], [ZISK_QP, hodnota], color=ORANZOVA, linewidth=2.2, zorder=4)
        osa_d.scatter([mu], [hodnota], s=80, color=TYRKYSOVA, edgecolor="white",
                      linewidth=1.3, zorder=6)
        osa_d.annotate("nejlepší mez:\n16 Kč/kg = stínová cena", xy=(16, ZISK_QP),
                       xytext=(16, 2250), color=MODRA, fontsize=10.5, weight="bold",
                       ha="center", va="bottom", zorder=7,
                       arrowprops={"arrowstyle": "->", "color": MODRA})
        osa_d.set_title(f"cena $\\mu$ = {_cislo(mu, 1)} Kč/kg  →  mez {_cislo(hodnota)} Kč",
                        fontsize=10.5, color=TYRKYSOVA, weight="bold", loc="left", pad=5)
        osa_d.set(xlim=(0, MU_MAX), ylim=(1780, 2680),
                  xlabel="cena mouky $\\mu$ [Kč/kg]", ylabel="horní mez zisku [Kč]")
        osa_d.tick_params(labelsize=10)
        osa_d.grid(alpha=0.2)
        return fig

    cesta_mu = (list(np.linspace(16, 0, 17)) + list(np.linspace(0, MU_MAX, 27))[1:]
                + list(np.linspace(MU_MAX, 16, 11))[1:])
    snimky, doby = [], []
    for i, mu in enumerate(cesta_mu):
        snimky.append(snimek(mu))
        doby.append(3000 if i == 0 else (900 if mu in (0.0, MU_MAX) else 110))
    doby[-1] = 2600
    uloz_gif(snimky, doby, "dualita-cena-mouky", dpi=100)


# --- nejmenší příklad: duál k  min (x - 3)^2  za  x <= 2 ---
def lagr_davka(x, mu):
    """Lagrangián dávkové úlohy: odchylka plus platba mu za každý mg nad limit."""
    return (x - 3) ** 2 + mu * (x - 2)


def dualni_davka(mu):
    """Duální funkce d(mu) = min_x L(x, mu) = mu - mu^2/4, minimum leží v x = 3 - mu/2."""
    return mu - mu**2 / 4


def _kontrola_davky() -> None:
    mu = np.linspace(0, 6, 601)
    x = np.linspace(-5, 10, 15001)
    for m in mu[::50]:
        assert abs(lagr_davka(x, m).min() - dualni_davka(m)) < 1e-6
    assert np.all(dualni_davka(mu) <= 1 + 1e-12)
    assert abs(mu[dualni_davka(mu).argmax()] - 2) < 1e-9 and abs(dualni_davka(2) - 1) < 1e-12


def dualita_davka() -> None:
    """Vlevo lagrangiány pro různé ceny mu, vpravo jejich minima jako funkce mu."""
    ceny = (0.0, 1.0, 2.0, 3.0, 4.0)
    barvy = ("#8fbfcc", "#3f8fa6", CERVENA, "#2d6f8f", MODRA)
    x = np.linspace(-0.2, 5.0, 300)
    mu_osa = np.linspace(0, 5, 300)

    fig, (osa, osa_d) = plt.subplots(1, 2, figsize=(8.6, 3.5),
                                     gridspec_kw={"width_ratios": (1.15, 1.0)})
    osa.axvspan(-0.2, 2, color=SVETLE_MODRA, zorder=0)
    osa.axvline(2, color=CERVENA, linewidth=1.2, alpha=0.6, zorder=1)
    for mu, barva in zip(ceny, barvy):
        silna = mu == 2.0
        osa.plot(x, lagr_davka(x, mu), color=barva, linewidth=2.8 if silna else 1.7, zorder=3)
        xm = 3 - mu / 2
        osa.scatter([xm], [dualni_davka(mu)], s=70, color=barva, edgecolor="white",
                    linewidth=1.2, zorder=5)
        if not silna:   # mu = 2 popisuje šipka, jeho minimum leží pod hvězdou
            osa.text(xm, dualni_davka(mu) - 0.32, f"$\\mu={mu:.0f}$", color=barva,
                     fontsize=9.5, weight="bold", ha="center", va="top", zorder=6,
                     bbox=STITEK)
    osa.text(2.12, 1.9, "$\\mu=2$", color=CERVENA, fontsize=10, weight="bold",
             ha="left", va="bottom", zorder=6, bbox=STITEK)
    osa.scatter([2], [1], s=220, marker="*", color=ORANZOVA, edgecolor="white",
                linewidth=1.0, zorder=7)
    osa.annotate("všechny procházejí\nbodem $x=2$: na limitu\nse neplatí nic",
                 xy=(2, 1), xytext=(0.05, 4.1), fontsize=9.5, color="#9a6200",
                 weight="bold", ha="left", va="center", zorder=8, bbox=STITEK,
                 arrowprops={"arrowstyle": "->", "color": "#9a6200"})
    osa.text(0.05, -1.3, "smí se", color="#0d6a70", fontsize=10, va="bottom", zorder=6)
    osa.set_title("$\\mathcal{L}(x,\\mu)=(x-3)^2+\\mu\\,(x-2)$ pro pět cen",
                  fontsize=10.5, color=MODRA, loc="left", pad=5)
    osa.set(xlim=(-0.2, 5.0), ylim=(-1.4, 5.0), xlabel="dávka $x$ [mg]")
    osa.tick_params(labelsize=9.5)
    osa.grid(alpha=0.2)

    osa_d.axhline(1, color=ORANZOVA, linewidth=1.8, linestyle="--", zorder=2)
    osa_d.text(4.95, 1.08, "$p^{*}=1$ (skutečné optimum)", color="#9a6200",
               fontsize=10, weight="bold", ha="right", va="bottom", zorder=6)
    osa_d.plot(mu_osa, dualni_davka(mu_osa), color=TYRKYSOVA, linewidth=2.6, zorder=3)
    for mu, barva in zip(ceny, barvy):
        osa_d.scatter([mu], [dualni_davka(mu)], s=70, color=barva, edgecolor="white",
                      linewidth=1.2, zorder=5)
    osa_d.annotate("nejlepší mez $d^{*}=1$\npři $\\mu=2$", xy=(2, 1), xytext=(1.2, -0.9),
                   fontsize=10, color=CERVENA, weight="bold", ha="left", va="center",
                   zorder=7, bbox=STITEK,
                   arrowprops={"arrowstyle": "->", "color": CERVENA})
    osa_d.text(4.95, 0.6, "$d(\\mu)=\\mu-\\mu^2/4$", color=TYRKYSOVA, fontsize=11,
               weight="bold", ha="right", va="bottom", zorder=6)
    osa_d.set_title("minimum každé křivky = jedna hodnota $d(\\mu)$", fontsize=10.5,
                    color=MODRA, loc="left", pad=5)
    osa_d.set(xlim=(-0.15, 5), ylim=(-1.4, 1.6), xlabel="cena $\\mu$")
    osa_d.tick_params(labelsize=9.5)
    osa_d.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "dualita-davka.svg", metadata=SVG_METADATA, bbox_inches="tight")
    nahledy = os.environ.get("OMM_NAHLEDY")
    if nahledy:
        Path(nahledy).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(nahledy) / "dualita-davka.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _kontrola()
    dualita_cena_mouky_gif()
    _kontrola_davky()
    dualita_davka()
