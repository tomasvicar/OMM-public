"""Dvě krátké ukázky k přednášce Úvod do optimalizace.

Spuštění:

    uv run python prednasky/L1/kod/demo_optimalizace.py

První okno animuje gradientní sestup ze dvou počátečních bodů. Po jeho zavření
skript vyřeší malý koncepční model plánování radioterapie pomocí CVXPY.
Model je didaktická ilustrace, nikoli klinický výpočet.
"""

from __future__ import annotations

import argparse

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def funkce(x: np.ndarray | float) -> np.ndarray | float:
    """Nekonvexní funkce se dvěma minimy."""
    return 0.16 * (x**2 - 4) ** 2 + 0.32 * x


def gradient(x: float) -> float:
    return 0.64 * x * (x**2 - 4) + 0.32


def iterace(start: float, krok: float = 0.07, pocet: int = 55) -> np.ndarray:
    body = [start]
    for _ in range(pocet):
        body.append(body[-1] - krok * gradient(body[-1]))
    return np.asarray(body)


def animace_gradientniho_sestupu() -> FuncAnimation:
    """Ukáže, že různé počáteční body mohou skončit v různých minimech."""
    x = np.linspace(-3.2, 3.2, 900)
    cesty = (iterace(-3.0), iterace(3.0))

    fig, osa = plt.subplots(figsize=(10, 5.5))
    osa.plot(x, funkce(x), color="#007f86", linewidth=3)
    osa.set(
        title="Gradientní sestup: stejná metoda, jiný počáteční bod",
        xlabel="$x$",
        ylabel="$f(x)$",
        xlim=(-3.25, 3.25),
        ylim=(-1.5, 4.4),
    )
    osa.grid(alpha=0.25)
    body = [
        osa.plot([], [], "o", color="#c73e1d", markersize=10, label="start vlevo")[0],
        osa.plot([], [], "o", color="#e9a23b", markersize=10, label="start vpravo")[0],
    ]
    stopy = [
        osa.plot([], [], color="#c73e1d", alpha=0.4)[0],
        osa.plot([], [], color="#e9a23b", alpha=0.4)[0],
    ]
    osa.legend(loc="upper center")

    def aktualizuj(snimek: int):
        for cesta, bod, stopa in zip(cesty, body, stopy):
            index = min(snimek, len(cesta) - 1)
            bod.set_data([cesta[index]], [funkce(cesta[index])])
            stopa.set_data(cesta[: index + 1], funkce(cesta[: index + 1]))
        return (*body, *stopy)

    animace = FuncAnimation(
        fig,
        aktualizuj,
        frames=max(map(len, cesty)),
        interval=110,
        blit=True,
        repeat=False,
    )
    plt.tight_layout()
    plt.show()
    return animace


def planovani_radioterapie() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vyřeší zjednodušené rozdělení intenzit čtyř svazků."""
    # Řádky představují oblasti, sloupce čtyři nastavitelné svazky.
    davka_nador = np.array(
        [
            [0.90, 0.55, 0.20, 0.35],
            [0.70, 0.75, 0.25, 0.25],
            [0.50, 0.60, 0.65, 0.15],
        ]
    )
    davka_organ = np.array(
        [
            [0.10, 0.35, 0.45, 0.60],
            [0.20, 0.15, 0.50, 0.55],
        ]
    )
    davka_zdrava_tkan = np.array(
        [
            [0.35, 0.15, 0.25, 0.40],
            [0.20, 0.30, 0.15, 0.35],
            [0.15, 0.25, 0.35, 0.20],
        ]
    )

    intenzity = cp.Variable(4, nonneg=True, name="intenzity_svazku")
    cilova_davka = np.ones(3)
    horni_mez_organu = np.full(2, 0.50)

    cil = cp.Minimize(
        cp.sum_squares(davka_nador @ intenzity - cilova_davka)
        + 0.20 * cp.sum_squares(davka_zdrava_tkan @ intenzity)
        + 0.03 * cp.sum_squares(intenzity)
    )
    omezeni = [
        davka_organ @ intenzity <= horni_mez_organu,
        intenzity <= 2.0,
    ]
    problem = cp.Problem(cil, omezeni)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Solver nenašel optimum: {problem.status}")

    x = np.asarray(intenzity.value)
    nador = davka_nador @ x
    organ = davka_organ @ x
    print("\nZjednodušený plán radioterapie")
    print("--------------------------------")
    print(f"Intenzity svazků: {np.round(x, 3)}")
    print(f"Dávka v nádoru:   {np.round(nador, 3)} (cíl 1.0)")
    print(f"Dávka v orgánech: {np.round(organ, 3)} (mez 0.5)")
    print(f"Hodnota cíle:     {problem.value:.4f}")
    return x, nador, organ


def argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        choices=("both", "gradient", "radiotherapy"),
        default="both",
        help="Kterou ukázku spustit.",
    )
    return parser.parse_args()


def main() -> None:
    volby = argumenty()
    if volby.demo in {"both", "gradient"}:
        animace_gradientniho_sestupu()
    if volby.demo in {"both", "radiotherapy"}:
        planovani_radioterapie()


if __name__ == "__main__":
    main()
