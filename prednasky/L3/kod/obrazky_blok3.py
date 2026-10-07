"""Obrázky pro blok „Bez omezení: jak poznám minimum“ (třetí přednáška)."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from spolecne import (
    OUT,
    SVG_METADATA,
    MODRA,
    TYRKYSOVA,
    CERVENA,
    ORANZOVA,
    SVETLE_MODRA,
    SEDIVA,
    uloz_gif,
)

# Volitelný adresář pro kontrolní PNG náhledy (nepatří do repozitáře).
NAHLEDY = os.environ.get("OMM_NAHLEDY")

STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8}


def uloz(fig, jmeno: str, **kwargs) -> None:
    """Uloží obrázek jako SVG do obrazky/ a případně jako PNG náhled."""
    fig.savefig(OUT / f"{jmeno}.svg", metadata=SVG_METADATA, **kwargs)
    if NAHLEDY:
        adresar = Path(NAHLEDY)
        adresar.mkdir(parents=True, exist_ok=True)
        fig.savefig(adresar / f"{jmeno}.png", dpi=130, **kwargs)
    plt.close(fig)


def stacionarni_bod() -> None:
    """Animace: kulička sjíždí po krajině proti gradientu, šipka ∇f se zkracuje.

    Na pozadí jsou vrstevnice a slabé šipky gradientu (kolmé na vrstevnice),
    v popředí bod, který jede po nejstrmějším spádu ke dnu. U dna se šipka
    ∇f zkrátí na nulu. **První snímek je výsledný stav** (bod na dně, žádná
    šipka), aby obrázek dával smysl i v PDF.
    """
    stred = np.array([2.0, 1.5])
    Q = np.array([[1.0, 0.55], [0.55, 1.6]])

    def hodnota(x, y):
        dx, dy = x - stred[0], y - stred[1]
        return 0.5 * (Q[0, 0] * dx * dx + 2 * Q[0, 1] * dx * dy + Q[1, 1] * dy * dy)

    def gradient(bod):
        return Q @ (np.asarray(bod) - stred)

    # Dráha nejstrmějšího spádu (spojitý tok x' = -∇f) z rohu obrázku.
    bod = np.array([-1.0, 3.9])
    draha = [bod.copy()]
    for _ in range(4000):
        bod = bod - 0.004 * gradient(bod)
        draha.append(bod.copy())
    draha = np.array(draha)
    oblouk = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(draha, axis=0), axis=1))))
    # snímky rovnoměrně v čase „kamery“, s dojezdem (ease-out) u dna
    t = np.linspace(0.0, 1.0, 30)
    cile = oblouk[-1] * (1.0 - (1.0 - t) ** 1.6)
    indexy = np.searchsorted(oblouk, cile).clip(0, len(draha) - 1)

    mrizka_x = np.linspace(-1.6, 5.6, 220)
    mrizka_y = np.linspace(-1.9, 4.9, 220)
    X, Y = np.meshgrid(mrizka_x, mrizka_y)
    Z = hodnota(X, Y)
    urovne = np.array([0.4, 1.2, 2.5, 4.4, 7.0, 10.4, 14.6])

    def snimek(i_bodu: int, konec: bool):
        fig, osa = plt.subplots(figsize=(6.0, 5.6))
        osa.contourf(X, Y, Z, levels=np.concatenate(([0.0], urovne, [40.0])),
                     cmap="Blues", alpha=0.5, zorder=0)
        cary = osa.contour(X, Y, Z, levels=urovne, colors=MODRA, linewidths=1.0, zorder=1)
        osa.clabel(cary, cary.levels[:3], fmt=lambda v: f"$f={v:g}$".replace(".", "{,}"),
                   fontsize=9.5, colors=MODRA)
        # slabé pole šipek: gradient je všude kolmý na vrstevnici
        for uhel_deg, uroven in ((20, 4.4), (95, 2.5), (150, 2.5), (215, 4.4),
                                 (310, 2.5), (335, 1.2), (60, 7.0), (240, 7.0)):
            smer = np.array([np.cos(np.deg2rad(uhel_deg)), np.sin(np.deg2rad(uhel_deg))])
            r = np.sqrt(2.0 * uroven / (smer @ Q @ smer))
            b = stred + r * smer
            g = gradient(b)
            delka = float(np.clip(0.2 * np.linalg.norm(g), 0.3, 0.8))
            k = b + delka * g / np.linalg.norm(g)
            if not all(-1.35 <= p[0] <= 5.35 and -1.4 <= p[1] <= 4.6 for p in (b, k)):
                continue
            osa.add_patch(FancyArrowPatch(b, k, arrowstyle="-|>", mutation_scale=11,
                                          linewidth=1.5, color=CERVENA, alpha=0.35, zorder=2))

        hotovo = draha[: i_bodu + 1]
        osa.plot(hotovo[:, 0], hotovo[:, 1], color=ORANZOVA, linewidth=2.6,
                 linestyle="--", zorder=4)
        poloha = draha[i_bodu]
        g = gradient(poloha)
        velikost = float(np.linalg.norm(g))
        if not konec:
            k = poloha + 0.55 * g
            osa.add_patch(FancyArrowPatch(poloha, k, arrowstyle="-|>", mutation_scale=18,
                                          linewidth=3.0, color=CERVENA, zorder=6))
            osa.text(k[0] + 0.12, k[1] + 0.05, "$\\nabla f$", color=CERVENA, fontsize=15,
                     weight="bold", ha="left", va="center", bbox=STITEK, zorder=8)
            osa.scatter(*poloha, s=150, color=MODRA, edgecolor="white", linewidth=1.5,
                        zorder=7)
        osa.scatter(*stred, s=300 if konec else 120, marker="*", color=ORANZOVA,
                    edgecolor=MODRA, linewidth=1.1, zorder=7)

        if konec:
            popis = "dno: $\\nabla f=\\mathbf{0}$\nšipka zmizela, není kam klesat"
            barva = "#9a6200"
        else:
            popis = ("$\\|\\nabla f\\|=%.2f$" % velikost).replace(".", "{,}")
            popis += "\nbod jede proti šipce, z kopce"
            barva = CERVENA
        osa.text(5.45, -1.75, popis, fontsize=12.5, weight="bold", color=barva,
                 ha="right", va="bottom", zorder=9,
                 bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                       "edgecolor": barva, "linewidth": 1.4})
        osa.set_title("$\\nabla f$ je kolmý na vrstevnici a míří do kopce",
                      fontsize=12.5, color=CERVENA, weight="bold")
        osa.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-1.6, 5.6), ylim=(-1.9, 4.9),
                aspect="equal")
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.15)
        fig.tight_layout()
        return fig

    snimky = [snimek(len(draha) - 1, konec=True)]
    doby = [1800]
    for i in indexy[:-1]:
        snimky.append(snimek(int(i), konec=False))
        doby.append(110)
    snimky.append(snimek(len(draha) - 1, konec=True))
    doby.append(3500)
    uloz_gif(snimky, doby, "stacionarni-bod-sestup")


def stacionarni_typy() -> None:
    """Minimum, maximum a sedlo jako plochy — a co k nim říká Hessián."""
    # Matice mají schválně nenulový mimodiagonální prvek: na diagonále samé
    # nejde typ přečíst a test přes determinant a stopu teprve dostane smysl.
    panely = (
        ("minimum", "[ 2   1 ]\n[ 1   2 ]",
         (2.0, 2.0, 1.0), "$\\det Q=3>0$,   $\\mathrm{tr}\\,Q=4>0$",
         "pozitivně definitní:  $\\mathbf{d}^\\mathsf{T}Q\\mathbf{d}>0$ v každém směru", TYRKYSOVA),
        ("maximum", "[ -2   1 ]\n[  1  -2 ]",
         (-2.0, -2.0, 1.0), "$\\det Q=3>0$,   $\\mathrm{tr}\\,Q=-4<0$",
         "negativně definitní:  $\\mathbf{d}^\\mathsf{T}Q\\mathbf{d}<0$ v každém směru", ORANZOVA),
        ("sedlo", "[ 1   2 ]\n[ 2   1 ]",
         (1.0, 1.0, 2.0), "$\\det Q=-3<0$",
         "indefinitní: jedním směrem nahoru, jiným dolů", CERVENA),
    )

    fig = plt.figure(figsize=(12.6, 4.05))
    mrizka = np.linspace(-1.0, 1.0, 41)
    X, Y = np.meshgrid(mrizka, mrizka)

    for i, (nazev, matice, (q11, q22, q12), test, popis, barva) in enumerate(panely):
        osa = fig.add_subplot(1, 3, i + 1, projection="3d")
        osa.computed_zorder = False
        Z = 0.5 * (q11 * X**2 + q22 * Y**2) + q12 * X * Y
        osa.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap="Blues",
                         edgecolor="#2a4f70", linewidth=0.15, alpha=0.92,
                         antialiased=True, shade=True, zorder=1)
        osa.scatter([0], [0], [0], s=110, color=CERVENA, edgecolor="white",
                    linewidth=1.2, depthshade=False, zorder=5)
        osa.set(xlabel="$x_1$", ylabel="$x_2$", zlim=(-1.6, 1.6),
                xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[])
        osa.tick_params(labelsize=9, pad=-2)
        osa.xaxis.labelpad = -6
        osa.yaxis.labelpad = -6
        osa.view_init(elev=24, azim=-58)

        stred = 1.0 / 6.0 + i / 3.0
        fig.text(stred - 0.055, 0.945, nazev, ha="right", va="top", fontsize=13.5,
                 color=barva, weight="bold")
        fig.text(stred - 0.035, 0.952, "$Q=$", ha="left", va="top", fontsize=12.5,
                 color=MODRA)
        fig.text(stred + 0.012, 0.955, matice, ha="left", va="top", fontsize=11,
                 color=MODRA, family="monospace", linespacing=1.15)
        fig.text(stred, 0.115, test, ha="center", fontsize=12, color=MODRA, weight="bold")
        fig.text(stred, 0.035, popis, ha="center", fontsize=10.5, color="#52616b")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.17, top=0.88, wspace=0.02)
    uloz(fig, "stacionarni-typy")


def normalni_rovnice() -> None:
    """Proložení přímky s rezidui a paraboloid chyby s jediným minimem."""
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([2.4, 2.6, 4.6, 4.2, 6.3, 6.5])
    A = np.column_stack([t, np.ones_like(t)])
    reseni = np.linalg.solve(A.T @ A, A.T @ y)
    smernice, posun = reseni

    def chyba(a, b):
        return sum((a * ti + b - yi) ** 2 for ti, yi in zip(t, y))

    fig = plt.figure(figsize=(7.6, 4.0))
    mrizka = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], left=0.075, right=0.995,
                              bottom=0.15, top=0.88, wspace=0.02)
    leva = fig.add_subplot(mrizka[0, 0])
    prava = fig.add_subplot(mrizka[0, 1], projection="3d")

    # --- levý panel: data, přímka a svislá rezidua ---------------------------
    osa_t = np.linspace(0.4, 6.6, 50)
    leva.plot(osa_t, smernice * osa_t + posun, color=TYRKYSOVA, linewidth=2.8, zorder=3)
    leva.text(0.55, 8.35, ("$y=%.2f\\,x+%.2f$" % (smernice, posun)).replace(".", "{,}"),
              color=TYRKYSOVA, fontsize=13, weight="bold", ha="left", va="top",
              zorder=6, bbox=STITEK)
    for ti, yi in zip(t, y):
        leva.plot([ti, ti], [yi, smernice * ti + posun], color=CERVENA, linewidth=2.0,
                  linestyle="--", zorder=4)
    leva.plot([], [], color=CERVENA, linewidth=2.0, linestyle="--",
              label="reziduum $r_i$")
    leva.scatter(t, y, s=80, color=MODRA, zorder=5, label="měření")
    leva.text(0.5, 9.6, "minimalizujeme $\\sum_i r_i^2$",
              fontsize=13, color=MODRA, va="top", zorder=6, bbox=STITEK)
    leva.set(xlabel="koncentrace [mmol/l]", ylabel="signál [mV]",
             xlim=(0.3, 6.8), ylim=(1.0, 9.9))
    leva.tick_params(labelsize=11)
    leva.xaxis.label.set_size(12)
    leva.yaxis.label.set_size(12)
    leva.grid(alpha=0.25)
    leva.legend(loc="lower right", fontsize=11.5, framealpha=0.94, handlelength=1.6,
                borderpad=0.4)
    leva.set_title("hledáme dvě čísla: směrnici a posun", fontsize=13, color=MODRA)

    # --- pravý panel: paraboloid chyby --------------------------------------
    a_osa = np.linspace(smernice - 0.62, smernice + 0.62, 45)
    b_osa = np.linspace(posun - 2.1, posun + 2.1, 45)
    AA, BB = np.meshgrid(a_osa, b_osa)
    ZZ = np.zeros_like(AA)
    for ti, yi in zip(t, y):
        ZZ += (AA * ti + BB - yi) ** 2

    prava.computed_zorder = False
    prava.plot_surface(AA, BB, ZZ, rcount=44, ccount=44, cmap="Blues",
                       edgecolor="#2a4f70", linewidth=0.14, alpha=0.9, antialiased=True,
                       zorder=1)
    z_min = chyba(smernice, posun)
    prava.scatter([smernice], [posun], [z_min], s=130, color=CERVENA,
                  edgecolor="white", linewidth=1.2, depthshade=False, zorder=5)
    popis_optima = ("jediné dno:\n$\\mathbf{x}^\\star=(%.2f;\\ %.2f)$"
                    % (smernice, posun)).replace(".", "{,}")
    prava.text2D(0.01, 0.93, popis_optima, transform=prava.transAxes, color=CERVENA,
                 fontsize=13, weight="bold", ha="left", va="top", zorder=6)
    prava.set(xlabel="směrnice $x_1$", ylabel="posun $x_2$")
    prava.set_zlim(0.0, float(ZZ.max()))
    prava.set_box_aspect((1.0, 1.0, 0.72))
    prava.xaxis.set_major_locator(plt.MaxNLocator(4))
    prava.yaxis.set_major_locator(plt.MaxNLocator(4))
    prava.tick_params(labelsize=10, pad=-2)
    prava.xaxis.labelpad = -3
    prava.yaxis.labelpad = -3
    prava.set_zticks([])
    carka = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    prava.xaxis.set_major_formatter(carka)
    prava.yaxis.set_major_formatter(carka)
    prava.view_init(elev=24, azim=-56)
    fig.text(0.99, 0.965, "chyba = konvexní paraboloid",
             ha="right", va="top", fontsize=13, color=MODRA)
    uloz(fig, "normalni-rovnice")



if __name__ == "__main__":
    stacionarni_bod()
    stacionarni_typy()
    normalni_rovnice()
