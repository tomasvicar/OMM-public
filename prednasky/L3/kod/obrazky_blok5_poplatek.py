"""Animace lineárního poplatku za mouku: lagrangián pekárny s rostoucí cenou μ.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_blok5_poplatek.py

Omezení na mouku nahradíme lineárním poplatkem: pekař platí μ Kč za každé
kilo nad 80 kg a stejně tolik dostane zpět za každé ušetřené. Řeší tedy úlohu
bez omezení

    max_x  zisk(x) - μ · (0,5 x1 + 0,2 x2 - 80),

tj. minimum lagrangiánu  L(x, μ) = f(x) + μ g1(x)  s f = -zisk. Plán je

    x(μ) = (280 - 10 μ ;  166,67 - 25 μ / 6),
    mouka(μ) = 173,33 - 35 μ / 6  [kg],

a přesně při μ = 16 Kč/kg vyjde (120; 100) na hranici mouky (80 kg).

Výstup: ``prednasky/L3/obrazky/poplatek-mouka.gif``. Levý panel: přípustná
oblast pekárny, hranice mouky a pece, cesta plánů x(μ) a vrstevnice
zisku minus poplatku (jejich střed je plán). Pravý panel: spotřeba mouky
jako funkce ceny, protíná zásobu 80 kg v μ = 16. **První snímek je výsledný
stav μ = 16** (záloha pro PDF a tisk), poslední snímek má dlouhou výdrž.

S proměnnou prostředí ``OMM_NAHLEDY`` se do toho adresáře uloží i PNG
náhled prvního snímku.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from spolecne import (
    CERVENA,
    MODRA,
    MU_QP,
    ORANZOVA,
    SVETLE_MODRA,
    TYRKYSOVA,
    X_NEOMEZENE,
    X_QP,
    uloz_gif,
    zisk,
)

plt.rcParams["figure.max_open_warning"] = 0   # snímky GIFu se zavírají až po uložení
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8}
ZELENA = "#3a8a52"
SEDA_TEXT = "#52616b"
VRCHOLY = [(0.0, 0.0), (160.0, 0.0), (100.0, 150.0), (0.0, 300.0)]
MU_MAX = 22.0


def plan(mu: float) -> np.ndarray:
    """Plán, který maximalizuje zisk minus poplatek za mouku (bez omezení)."""
    return np.array([(14 - 0.5 * mu) / 0.05, (8 - 0.2 * mu) / 0.048])


def mouka(x: np.ndarray) -> float:
    """Spotřeba mouky [kg]."""
    return float(0.5 * x[0] + 0.2 * x[1])


def zisk_minus_poplatek(x1, x2, mu: float):
    """Pekařův cíl při ceně mu: zisk - mu * g1(x), tj. -L(x, mu)."""
    return zisk(x1, x2) - mu * (0.5 * x1 + 0.2 * x2 - 80)


def _kontrola() -> None:
    """Čísla ze slidu: tabulka x(μ) a dotyk hranice mouky přesně v μ = 16."""
    assert np.allclose(plan(0), X_NEOMEZENE)
    assert np.allclose(plan(MU_QP[0]), X_QP)
    assert abs(mouka(plan(MU_QP[0])) - 80) < 1e-9
    ocekavane = {0: (280, 500 / 3, 520 / 3), 10: (180, 125, 115),
                 16: (120, 100, 80), 20: (80, 250 / 3, 170 / 3)}
    for mu, (x1, x2, kg) in ocekavane.items():
        x = plan(mu)
        assert np.allclose(x, (x1, x2)) and abs(mouka(x) - kg) < 1e-9, mu
    # x(μ) je opravdu maximum zisku minus poplatku: gradient je nulový.
    for mu in (0.0, 7.5, 16.0, 21.0):
        x = plan(mu)
        grad = np.array([14 - 0.05 * x[0] - 0.5 * mu, 8 - 0.048 * x[1] - 0.2 * mu])
        assert np.allclose(grad, 0)


def _cislo(hodnota: float, mist: int = 0) -> str:
    return f"{hodnota:.{mist}f}".replace(".", ",")


def poplatek_mouka_gif() -> None:
    """Cena mouky roste od 0 do 22 Kč/kg, plán sjíždí k hranici a při 16 ji trefí."""
    mu_osa = np.linspace(0, MU_MAX, 300)
    cesta = np.array([plan(m) for m in mu_osa])
    spotreba = np.array([mouka(x) for x in cesta])
    t = np.linspace(0, 320, 40)
    g1, g2 = np.meshgrid(np.linspace(0, 300, 241), np.linspace(0, 320, 241))

    def snimek(mu: float) -> plt.Figure:
        x = plan(mu)
        kg = mouka(x)
        fig, (osa, osa_m) = plt.subplots(1, 2, figsize=(8.8, 4.1),
                                         gridspec_kw={"width_ratios": (1.12, 1.0)})
        fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.135, wspace=0.26)

        # --- rovina výroby ---
        osa.add_patch(Polygon(VRCHOLY, closed=True, facecolor=SVETLE_MODRA,
                              edgecolor="none", zorder=1))
        osa.plot(t, (80 - 0.5 * t) / 0.2, color=CERVENA, linewidth=2.4, zorder=3)
        osa.plot(t, (600 - 3 * t) / 2, color=ZELENA, linewidth=1.6, alpha=0.6, zorder=3)
        hodnoty = zisk_minus_poplatek(g1, g2, mu)
        vrchol = float(zisk_minus_poplatek(*x, mu))
        osa.contour(g1, g2, hodnoty, levels=vrchol - np.array([1600, 900, 400, 100]),
                    colors=[MODRA], linewidths=0.9, alpha=0.45, zorder=2)
        osa.plot(cesta[:, 0], cesta[:, 1], color="#8a97a0", linewidth=1.3,
                 linestyle=(0, (2, 2)), zorder=4)
        osa.scatter(*X_QP, s=170, marker="*", color=CERVENA, edgecolor="white",
                    linewidth=1.0, zorder=5, alpha=0.55)
        osa.text(276, 176, "$\\mu=0$", color=SEDA_TEXT, fontsize=10, ha="right",
                 va="bottom", zorder=6, bbox=STITEK)
        konec = plan(MU_MAX)
        osa.text(konec[0] - 6, konec[1] - 6, "$\\mu=22$", color=SEDA_TEXT, fontsize=10,
                 ha="right", va="top", zorder=6, bbox=STITEK)
        osa.text(206, 30, "pec", color=ZELENA, fontsize=10.5, ha="left", va="center",
                 zorder=6, bbox=STITEK)
        osa.text(166, 6, "mouka 80 kg", color=CERVENA, fontsize=10.5, weight="bold",
                 ha="left", va="bottom", zorder=8, bbox=STITEK)
        osa.text(30, 140, "přípustná\noblast", color="#006a70", fontsize=10.5,
                 ha="center", zorder=6)
        osa.scatter(*x, s=120, color=TYRKYSOVA, edgecolor="white", linewidth=1.4, zorder=7)
        osa.set_title(f"plán $x(\\mu)$ = ({_cislo(x[0])}; {_cislo(x[1], 1)})",
                      fontsize=10.5, color=TYRKYSOVA, weight="bold", loc="left", pad=5)
        osa.set(xlim=(0, 300), ylim=(0, 320))
        osa.set_xlabel("chleby $x_1$ [ks/den]", fontsize=10.5)
        osa.set_ylabel("bagety $x_2$ [ks/den]", fontsize=10.5)
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.2)

        # --- spotřeba mouky jako funkce ceny ---
        osa_m.plot(mu_osa, spotreba, color=TYRKYSOVA, linewidth=2.6, zorder=3)
        osa_m.axhline(80, color=CERVENA, linewidth=1.8, linestyle="--", zorder=2)
        osa_m.text(0.4, 83, "zásoba 80 kg", color=CERVENA, fontsize=10.5,
                   weight="bold", ha="left", va="bottom", zorder=6)
        osa_m.annotate("správná cena:\n16 Kč/kg", xy=(16, 80), xytext=(8.0, 50),
                       color=MODRA, fontsize=10.5, weight="bold", ha="center",
                       va="center", zorder=7,
                       arrowprops={"arrowstyle": "->", "color": MODRA})
        osa_m.plot([mu, mu], [80, kg], color=ORANZOVA, linewidth=2.2, zorder=4)
        osa_m.scatter([mu], [kg], s=80, color=TYRKYSOVA, edgecolor="white",
                      linewidth=1.3, zorder=6)
        if kg > 80.05:
            stav = f"chybí {_cislo(kg - 80, 1)} kg"
        elif kg < 79.95:
            stav = f"zbývá {_cislo(80 - kg, 1)} kg"
        else:
            stav = "přesně zásoba"
        osa_m.text(MU_MAX - 0.4, 178, f"mouka {_cislo(kg, 1)} kg\n{stav}", color=TYRKYSOVA,
                   fontsize=10.5, weight="bold", ha="right", va="top", zorder=6, bbox=STITEK)
        osa_m.set_title(f"cena $\\mu$ = {_cislo(mu, 1)} Kč/kg",
                        fontsize=10.5, color=TYRKYSOVA, weight="bold", loc="left", pad=5)
        osa_m.set(xlim=(0, MU_MAX), ylim=(30, 185))
        osa_m.set_xlabel("cena mouky $\\mu$ [Kč/kg]", fontsize=10.5)
        osa_m.set_ylabel("spotřeba mouky [kg]", fontsize=10.5)
        osa_m.tick_params(labelsize=10)
        osa_m.grid(alpha=0.2)
        return fig

    mu16 = MU_QP[0]
    cesta_mu = ([mu16] + list(np.linspace(0, mu16, 25))
                + list(np.linspace(mu16, MU_MAX, 10))[1:]
                + list(np.linspace(MU_MAX, mu16, 9))[1:])
    snimky, doby = [], []
    for i, mu in enumerate(cesta_mu):
        snimky.append(snimek(mu))
        if i == 0:
            doby.append(3000)
        elif mu == 0.0 or mu == MU_MAX:
            doby.append(900)
        elif abs(mu - mu16) < 1e-9:
            doby.append(1600)
        else:
            doby.append(110)
    doby[-1] = 3000

    nahledy = os.environ.get("OMM_NAHLEDY")
    if nahledy:
        Path(nahledy).mkdir(parents=True, exist_ok=True)
        snimky[0].savefig(Path(nahledy) / "poplatek-mouka.png", dpi=100)
        snimek(10.0).savefig(Path(nahledy) / "poplatek-mouka-mu10.png", dpi=100)
    uloz_gif(snimky, doby, "poplatek-mouka", dpi=100)


if __name__ == "__main__":
    _kontrola()
    poplatek_mouka_gif()
