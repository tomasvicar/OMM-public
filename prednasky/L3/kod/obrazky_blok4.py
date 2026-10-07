"""Obrázky k bloku „Lagrangián: pokuta se správnou cenou“ třetí přednášky: rovnost a multiplikátor."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d import proj3d

from spolecne import (
    OUT,
    SVG_METADATA,
    MODRA,
    TYRKYSOVA,
    CERVENA,
    ORANZOVA,
    SVETLE_MODRA,
    uloz_gif,
)

# Volitelný adresář pro kontrolní PNG náhledy (nepatří do repozitáře).
NAHLEDY = os.environ.get("OMM_NAHLEDY")

SEDA_TEXT = "#52616b"
SEDA_CARA = "#a9bcc7"
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8}


def uloz(fig, jmeno: str, **kwargs) -> None:
    """Uloží obrázek jako SVG do obrazky/ a případně jako PNG náhled."""
    fig.savefig(OUT / f"{jmeno}.svg", metadata=SVG_METADATA, **kwargs)
    if NAHLEDY:
        adresar = Path(NAHLEDY)
        adresar.mkdir(parents=True, exist_ok=True)
        fig.savefig(adresar / f"{jmeno}.png", dpi=130, **kwargs)
    plt.close(fig)


# --- společná úloha pro oba klíčové obrázky --------------------------------
# min f(x) za h(x) = 0: eliptické vrstevnice se středem mimo kružnici omezení.
STRED = np.array([2.3, 1.5])
VAHY = np.array([0.9, 0.45])
POLOMER = 1.6


def _f(x1, x2):
    return VAHY[0] * (x1 - STRED[0]) ** 2 + VAHY[1] * (x2 - STRED[1]) ** 2


def _grad_f(bod):
    return 2.0 * VAHY * (bod - STRED)


def _bod_na_kruznici(uhel):
    return POLOMER * np.array([np.cos(uhel), np.sin(uhel)])


def _optimalni_uhel() -> float:
    uhly = np.linspace(0.0, 2.0 * np.pi, 200_001)
    body = POLOMER * np.column_stack([np.cos(uhly), np.sin(uhly)])
    return float(uhly[np.argmin(_f(body[:, 0], body[:, 1]))])


def _sipka(osa, bod, smer, barva, delka=0.72, sirka=2.6, zorder=6):
    """Šipka pevné délky ve směru smer; vrací koncový bod."""
    jednotkovy = smer / np.linalg.norm(smer)
    konec = bod + delka * jednotkovy
    osa.add_patch(FancyArrowPatch(bod, konec, arrowstyle="-|>", mutation_scale=17,
                                  linewidth=sirka, color=barva, zorder=zorder))
    return konec


def lagrange_3d() -> None:
    """Krajina z=f(x1,x2) s cestou (křivkou omezení): nejnižší bod cesty není dno."""
    uhel_opt = _optimalni_uhel()
    bod = _bod_na_kruznici(uhel_opt)
    kruh = np.linspace(0.0, 2.0 * np.pi, 400)
    krivka = POLOMER * np.column_stack([np.cos(kruh), np.sin(kruh)])
    vysky = _f(krivka[:, 0], krivka[:, 1])

    fig = plt.figure(figsize=(6.8, 5.6))
    prostor = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)

    mrizka = np.linspace(-2.4, 4.8, 73)
    xx, yy = np.meshgrid(mrizka, mrizka)
    zz = _f(xx, yy)
    zz[zz > 15.5] = np.nan
    zaklad = 0.0
    prostor.plot_surface(xx, yy, zz, color=SVETLE_MODRA, alpha=0.75, edgecolor="#9cc4dc",
                         linewidth=0.3, rstride=2, cstride=2, shade=False, zorder=1)
    prostor.scatter(*STRED, _f(*STRED), s=90, color=SEDA_TEXT, edgecolor="white",
                    linewidth=1.0, depthshade=False, zorder=4)
    for i in range(0, len(krivka), 6):
        prostor.plot([krivka[i, 0]] * 2, [krivka[i, 1]] * 2, [zaklad, vysky[i]],
                     color=TYRKYSOVA, linewidth=0.8, alpha=0.25, zorder=3)
    prostor.plot(krivka[:, 0], krivka[:, 1], zaklad, color=TYRKYSOVA, linewidth=1.6,
                 linestyle="--", zorder=2)
    prostor.plot(krivka[:, 0], krivka[:, 1], vysky, color=TYRKYSOVA, linewidth=3.6, zorder=5)
    prostor.scatter(*bod, _f(*bod), s=150, color=CERVENA, edgecolor="white",
                    linewidth=1.0, depthshade=False, zorder=7)
    prostor.set(zlim=(zaklad, 16), xlabel="", ylabel="")
    prostor.set_zticks([])
    prostor.set_xticks([])
    prostor.set_yticks([])
    prostor.view_init(elev=26, azim=-12)
    prostor.set_box_aspect((1.0, 1.0, 0.8), zoom=1.15)

    fig.canvas.draw()
    matice = prostor.get_proj()
    px, py, _ = proj3d.proj_transform(bod[0], bod[1], _f(*bod), matice)
    sx, sy, _ = proj3d.proj_transform(STRED[0], STRED[1], _f(*STRED), matice)
    prostor.annotate("nejnižší bod cesty $=\\mathbf{x}^\\star$\n$\\nabla f\\neq\\mathbf{0}$",
                     xy=(px, py), xycoords="data", xytext=(0.02, 0.22),
                     textcoords="axes fraction", color=CERVENA, fontsize=12, weight="bold",
                     ha="left", va="center", bbox=STITEK,
                     arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4}, zorder=10)
    prostor.annotate("dno krajiny: $\\nabla f=\\mathbf{0}$,\nale na cestě neleží",
                     xy=(sx, sy), xycoords="data", xytext=(0.66, 0.12),
                     textcoords="axes fraction", color=SEDA_TEXT, fontsize=11.5,
                     ha="left", va="center", bbox=STITEK,
                     arrowprops={"arrowstyle": "->", "color": SEDA_TEXT, "lw": 1.2}, zorder=10)
    prostor.text2D(0.02, 0.90, "cesta = omezení $h(\\mathbf{x})=0$\nzvednuté na krajinu",
                   transform=prostor.transAxes, color=TYRKYSOVA, fontsize=12,
                   weight="bold", linespacing=1.25)
    prostor.text2D(0.70, 0.90, "krajina $z=f(\\mathbf{x})$", transform=prostor.transAxes,
                   color=MODRA, fontsize=12)

    fig.tight_layout()
    uloz(fig, "lagrange-3d", bbox_inches="tight")


def _kruznice_a_primka(osa, b_primky, r_kruhu, dotyk: bool, stare=()):
    """Společná kresba školního příkladu: vrstevnice f=x1²+x2², přímka x1+x2=b."""
    kruh = np.linspace(0.0, 2.0 * np.pi, 400)
    for polomer in stare:
        osa.plot(polomer * np.cos(kruh), polomer * np.sin(kruh), color=SEDA_CARA,
                 linewidth=1.0, zorder=1)
    osa.plot(r_kruhu * np.cos(kruh), r_kruhu * np.sin(kruh), color=MODRA,
             linewidth=2.6, zorder=3)
    t = np.linspace(-3.0, 5.0, 50)
    osa.plot(t, b_primky - t, color=TYRKYSOVA, linewidth=3.0, zorder=4)
    osa.scatter([0.0], [0.0], s=40, color=SEDA_TEXT, zorder=5)
    if dotyk:
        bod = np.array([b_primky / 2.0, b_primky / 2.0])
        osa.scatter(*bod, s=260, marker="*", color=CERVENA, edgecolor="white",
                    linewidth=1.0, zorder=7)
    osa.axhline(0.0, color=SEDA_TEXT, linewidth=0.9, alpha=0.6)
    osa.axvline(0.0, color=SEDA_TEXT, linewidth=0.9, alpha=0.6)
    osa.set(xlim=(-1.6, 2.7), ylim=(-1.6, 2.7), aspect="equal",
            xlabel="$x_1$", ylabel="$x_2$", xticks=[-1, 0, 1, 2], yticks=[-1, 0, 1, 2])
    osa.tick_params(labelsize=11.5)
    osa.xaxis.label.set_size(13)
    osa.yaxis.label.set_size(13)
    osa.grid(alpha=0.15)


def lagrange_skolni() -> None:
    """Animace: nafukujeme kružnice f=konst., dokud se přímky x1+x2=2 nedotknou.

    První snímek je výsledný stav (dotyk v (1,1) s rovnoběžnými gradienty),
    potom kružnice roste od malého poloměru a na dotyku se zastaví.
    """
    r_opt = np.sqrt(2.0)

    def snimek(r, stare):
        fig, osa = plt.subplots(figsize=(5.2, 5.0))
        dotyk = abs(r - r_opt) < 1e-9
        _kruznice_a_primka(osa, 2.0, r, dotyk, stare)
        osa.text(2.55, -0.62, "$x_1+x_2=2$", color=TYRKYSOVA, fontsize=12,
                 weight="bold", ha="right", va="center", rotation=-45, bbox=STITEK,
                 zorder=8)
        if dotyk:
            bod = np.array([1.0, 1.0])
            _sipka(osa, bod, np.array([1.0, 1.0]), CERVENA, delka=0.78, sirka=2.6)
            _sipka(osa, bod + np.array([0.16, -0.16]), np.array([1.0, 1.0]), TYRKYSOVA,
                   delka=0.42, sirka=2.6)
            osa.text(1.0, 2.05, "$\\nabla f=(2,2)$", color=CERVENA, fontsize=12,
                     weight="bold", ha="left", va="center", bbox=STITEK, zorder=8)
            osa.text(1.62, 0.95, "$\\nabla h=(1,1)$", color=TYRKYSOVA, fontsize=12,
                     weight="bold", ha="left", va="center", bbox=STITEK, zorder=8)
            osa.annotate("$\\mathbf{x}^\\star=(1,1)$", xy=(0.97, 0.97), xytext=(-1.45, 2.2),
                         color=CERVENA, fontsize=12.5, weight="bold",
                         arrowprops={"arrowstyle": "->", "color": CERVENA}, zorder=8)
            stav, barva = "první dotyk: $f=2$ je optimum", CERVENA
        else:
            stav = ("$f=%.2f$: přímky se ještě nedotkne" % (r * r)).replace(".", "{,}")
            barva = MODRA
        osa.set_title(stav, fontsize=12.5, color=barva, weight="bold")
        fig.tight_layout()
        return fig

    polomery = np.linspace(0.25, r_opt, 22)
    snimky = [snimek(r_opt, (0.7, 1.05))]
    doby = [1800]
    for i, r in enumerate(polomery[:-1]):
        stare = [p for p in (0.7, 1.05) if p < r]
        snimky.append(snimek(float(r), stare))
        doby.append(140)
    snimky.append(snimek(r_opt, (0.7, 1.05)))
    doby.append(3800)
    uloz_gif(snimky, doby, "lagrange-skolni")


def lagrange_citlivost() -> None:
    """Animace: posouváme omezení x1+x2=b a sledujeme optimální hodnotu p*(b)=b²/2.

    Vlevo se přímka posouvá a kružnice dotyku roste, vpravo po parabole
    p*(b) jede bod. První i poslední snímek je dnešní úloha b=2 s tečnou
    o směrnici 2 = -λ.
    """

    def snimek(b, tecna: bool):
        fig, (leva, prava) = plt.subplots(1, 2, figsize=(8.2, 4.3),
                                          gridspec_kw={"width_ratios": [1.0, 1.15]})
        _kruznice_a_primka(leva, b, b / np.sqrt(2.0), True, (0.7, 1.05, 1.75))
        leva.plot(np.linspace(-3.0, 5.0, 50), 2.0 - np.linspace(-3.0, 5.0, 50),
                  color=TYRKYSOVA, linewidth=1.2, linestyle=":", zorder=2)
        leva.set_title(("omezení $x_1+x_2=b$,  $b=%.2f$" % b).replace(".", "{,}"),
                       fontsize=14.5, color=TYRKYSOVA, weight="bold")

        osa_b = np.linspace(0.0, 4.0, 200)
        prava.plot(osa_b, osa_b ** 2 / 2.0, color=MODRA, linewidth=3.0, zorder=3,
                   label="$p^\\star(b)=b^2/2$")
        prava.scatter([b], [b * b / 2.0], s=120, color=CERVENA, zorder=6,
                      edgecolor="white", linewidth=1.0)
        if tecna:
            t = np.linspace(0.75, 3.45, 50)
            prava.plot(t, 2.0 + 2.0 * (t - 2.0), color=CERVENA, linewidth=2.2,
                       linestyle="--", zorder=4, label="tečna, směrnice $2$")
            prava.plot([2.0, 3.0], [2.0, 2.0], color=SEDA_TEXT, linewidth=1.4, zorder=5)
            prava.plot([3.0, 3.0], [2.0, 4.0], color=SEDA_TEXT, linewidth=1.4, zorder=5)
            prava.text(2.5, 1.72, "$\\Delta b=1$", color=SEDA_TEXT, fontsize=12.5,
                       ha="center", va="top", bbox=STITEK, zorder=7)
            prava.text(3.08, 3.0, "$\\Delta p^\\star\\approx2$", color=SEDA_TEXT,
                       fontsize=12.5, ha="left", va="center", bbox=STITEK, zorder=7)
            prava.text(0.15, 6.2, "směrnice $2=-\\lambda$,  $\\lambda=-2$",
                       color=CERVENA, fontsize=14, weight="bold", ha="left",
                       va="center", zorder=7,
                       bbox={"boxstyle": "round,pad=0.34", "facecolor": "#fbeeea",
                             "edgecolor": CERVENA, "linewidth": 1.3})
        prava.set(xlabel="pravá strana $b$", ylabel="$p^\\star(b)$",
                  xlim=(0, 4.0), ylim=(0, 7.0))
        prava.set_yticks([0, 2, 4, 6])
        prava.set_xticks([0, 1, 2, 3, 4])
        prava.tick_params(labelsize=12)
        prava.xaxis.label.set_size(13)
        prava.yaxis.label.set_size(13)
        prava.grid(alpha=0.22)
        prava.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=12, framealpha=0.95)
        prava.set_title(("$p^\\star=%.2f$" % (b * b / 2.0)).replace(".", "{,}"),
                        fontsize=14.5, color=MODRA, weight="bold")
        fig.tight_layout()
        return fig

    cesta = np.concatenate((np.linspace(2.0, 1.0, 9), np.linspace(1.0, 3.2, 20)[1:],
                            np.linspace(3.2, 2.0, 11)[1:-1]))
    snimky = [snimek(2.0, True)]
    doby = [2000]
    for b in cesta:
        snimky.append(snimek(float(b), False))
        doby.append(150)
    snimky.append(snimek(2.0, True))
    doby.append(4000)
    uloz_gif(snimky, doby, "lagrange-citlivost", dpi=110)


def objem_povrch() -> None:
    """Krychle a placatý kvádr se stejným povrchem 24, ale různým objemem."""

    def promitni(x, y, z):
        """Jednoduchá axonometrie: prostor do roviny obrázku."""
        return np.array([(x - y) * 0.87, (x + y) * 0.5 + z])

    def kvadr(osa, posun, rozmery, barva, popisky, barva_textu):
        """Nakreslí kvádr (viditelné jsou stěny nahoře, vpravo a vlevo) i s kótami."""
        a, b, c = rozmery
        rohy = {(i, j, k): promitni(i * a, j * b, k * c) + posun
                for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        steny = (
            ([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 0.55),   # horní
            ([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], 0.32),   # vpravo dopředu
            ([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], 0.16),   # vlevo dopředu
        )
        for vrcholy, sytost in steny:
            osa.add_patch(Polygon([rohy[v] for v in vrcholy], closed=True,
                                  facecolor=barva, alpha=sytost, edgecolor=barva,
                                  linewidth=2.2, zorder=3))
        koty = (
            (popisky[0], 0.5 * (rohy[(0, 0, 0)] + rohy[(1, 0, 0)]) + [0.42, -0.30], "left"),
            (popisky[1], 0.5 * (rohy[(0, 0, 0)] + rohy[(0, 1, 0)]) + [-0.42, -0.30], "right"),
            (popisky[2], 0.5 * (rohy[(0, 0, 0)] + rohy[(0, 0, 1)]) + [-0.18, 0.0], "right"),
        )
        for text, misto, zarovnani in koty:
            osa.text(*misto, text, color=barva_textu, fontsize=12.5, weight="bold",
                     ha=zarovnani, va="center", bbox=STITEK, zorder=6)

    hneda = "#9a6200"
    fig, osa = plt.subplots(figsize=(6.6, 3.0))
    osa.set(xlim=(-4.2, 11.0), ylim=(-2.3, 5.0), aspect="equal")
    osa.axis("off")

    kvadr(osa, np.array([0.0, 0.0]), (2.0, 2.0, 2.0), TYRKYSOVA, ("2", "2", "2"), TYRKYSOVA)
    osa.text(-0.4, -1.75, "krychle $2\\times2\\times2$\npovrch 24  ·  objem $8$",
             color=TYRKYSOVA, fontsize=13, weight="bold", ha="center", va="center")

    kvadr(osa, np.array([6.3, 0.6]), (3.0, 3.0, 0.5), ORANZOVA, ("3", "3", "0,5"), hneda)
    osa.text(7.2, -1.75, "kvádr $3\\times3\\times0{,}5$\npovrch 24  ·  objem jen $4{,}5$",
             color=hneda, fontsize=13, weight="bold", ha="center", va="center")

    osa.text(2.8, 4.35, "stejný povrch,\njiný tvar", color=MODRA, fontsize=12.5,
             ha="center", va="center", style="italic")
    osa.add_patch(FancyArrowPatch((2.05, 2.6), (3.55, 2.6), arrowstyle="<|-|>",
                                  mutation_scale=16, linewidth=2.2, color=MODRA))

    fig.tight_layout()
    uloz(fig, "objem-povrch", bbox_inches="tight")


if __name__ == "__main__":
    lagrange_3d()
    lagrange_skolni()
    lagrange_citlivost()
    objem_povrch()
