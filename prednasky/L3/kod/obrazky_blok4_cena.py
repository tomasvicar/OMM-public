"""Obrázek „Obě cesty vedou k jedné ceně“ (blok Omezení jako pokuta a zeď).

Pokuta i bariéra v řešení „účtují“ za omezení cenu — sklon svého členu:

    pokuta    ρ·max(0, g)²       sklon 2ρ·překročení
    bariéra   −(1/t)·log(−g)     sklon 1/(t·rezerva)

S rostoucím parametrem (ρ = t) se odhad ceny mouky z obou stran blíží
k 16 Kč/kg, stínové ceně mouky v QP pekárně. Pec má rezervu, její cena jde
k nule. Řešení obou úloh se počítají stejnými funkcemi jako animace
``penalizace.gif`` a ``bariera-cesta.gif`` (``obrazky_blok7.py``), čísla se
tedy nikde neopisují ručně.

Výstup: ``prednasky/L3/obrazky/obe-cesty-cena.svg``

    cd prednasky/L3/kod && uv run python obrazky_blok4_cena.py

S proměnnou prostředí ``OMM_NAHLEDY`` (adresář) se uloží i PNG náhled.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from obrazky_blok7 import A, A4, B, B4, MU1, centralni_cesta, penalizovane_reseni
from spolecne import CERVENA, MODRA, ORANZOVA, OUT, SVG_METADATA, TYRKYSOVA

PARAMETR = np.geomspace(0.3, 1e3, 201)   # ρ u pokuty a t u bariéry
SEDA = "#52616b"
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}


def ceny_pokuty() -> np.ndarray:
    """Odhad ceny 2ρ·max(0, gᵢ) pro mouku a pec; tvar (len(PARAMETR), 2)."""
    ceny = []
    for rho in PARAMETR:
        x = penalizovane_reseni(rho)
        ceny.append(2 * rho * np.maximum(0.0, A @ x - B))
    return np.array(ceny)


def ceny_bariery() -> np.ndarray:
    """Odhad ceny 1/(t·rezervaᵢ) pro mouku a pec; tvar (len(PARAMETR), 2)."""
    # Centrální cesta se počítá s teplým startem už od malých t, aby šla spojitě.
    nabeh = np.geomspace(2e-5, PARAMETR[0], 12, endpoint=False)
    cesta = centralni_cesta(np.concatenate([nabeh, PARAMETR]))[len(nabeh):]
    rezervy = B4[:2] - cesta @ A4[:2].T
    return 1.0 / (PARAMETR[:, None] * rezervy)


def obrazek() -> None:
    pokuta = ceny_pokuty()
    bariera = ceny_bariery()

    fig, osa = plt.subplots(figsize=(8.6, 3.9))

    # Mouka: obě křivky k 16 Kč/kg. Pec má rezervu a její cena jde k nule —
    # to říká text slidu, obrázek drží jen jedno sdělení.
    osa.axhline(MU1, color=CERVENA, linewidth=1.8, linestyle="--", zorder=2)
    osa.text(0.33, MU1 + 0.6, "stínová cena mouky 16 Kč/kg", color=CERVENA,
             fontsize=12.5, weight="bold", zorder=5, bbox=STITEK)
    osa.semilogx(PARAMETR, pokuta[:, 0], color=ORANZOVA, linewidth=3.0, zorder=4)
    osa.semilogx(PARAMETR, bariera[:, 0], color=TYRKYSOVA, linewidth=3.0, zorder=4)
    for parametr in (1.0, 10.0, 100.0):
        i = int(np.argmin(np.abs(PARAMETR - parametr)))
        osa.scatter([PARAMETR[i]] * 2, [pokuta[i, 0], bariera[i, 0]], s=36,
                    color=[ORANZOVA, TYRKYSOVA], edgecolor="white", zorder=6)
    osa.text(1.2, 9.0, "pokuta zvenku: $2\\rho\\cdot$překročení", color="#8a6100",
             fontsize=12.5, ha="left", zorder=5, bbox=STITEK)
    osa.text(30.0, 12.0, "bariéra zevnitř: $1/(t\\cdot$rezerva$)$", color="#006a70",
             fontsize=12.5, ha="left", zorder=5, bbox=STITEK)
    osa.set(xlim=(PARAMETR[0], PARAMETR[-1]), ylim=(0, 19))
    osa.set_ylabel("cena za kilo mouky [Kč/kg]", fontsize=12)
    osa.set_xlabel("strmost pokuty $\\rho$ = tenkost zdi $t$  (logaritmicky)", fontsize=12)
    osa.tick_params(labelsize=11)
    osa.grid(alpha=0.2, which="major")
    osa.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT / "obe-cesty-cena.svg", metadata=SVG_METADATA, bbox_inches="tight")
    nahledy = os.environ.get("OMM_NAHLEDY")
    if nahledy:
        adresar = Path(nahledy)
        adresar.mkdir(parents=True, exist_ok=True)
        fig.savefig(adresar / "obe-cesty-cena.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for rho in (0.1, 1.0, 10.0, 100.0):
        i = int(np.argmin(np.abs(PARAMETR - rho)))
        print(f"rho = t = {rho:g}: pokuta {pokuta[i].round(3)}, bariéra {bariera[i].round(3)}")


if __name__ == "__main__":
    obrazek()
