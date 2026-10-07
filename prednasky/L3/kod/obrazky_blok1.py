"""Obrázky k bloku 1 třetí přednášky: od přímek k zakřivenému světu.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_blok1.py

Vytvoří ``kde-linearita-praska.svg``, animaci ``vrstevnice-posun.gif``,
``tri-plochy.svg`` a ``mapa-trid.svg``. Písmo je záměrně velké: obrázky se na
slidu zobrazují zhruba na šířku plátna a musí být čitelné i ze zadních řad.
"""

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from spolecne import (
    OUT,
    SVG_METADATA,
    MODRA,
    TYRKYSOVA,
    CERVENA,
    ORANZOVA,
    SVETLE_MODRA,
    SEDIVA,
    D_PEKARNA,
    P_PEKARNA,
    X_QP,
    X_NEOMEZENE,
    X_LP,
    uloz_gif,
)

# Vrcholy přípustné oblasti pekárny — stejný polyedr jako ve druhé přednášce.
POLYEDR = [[0.0, 0.0], [160.0, 0.0], [100.0, 150.0], [0.0, 300.0]]
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8}
TMAVE_ORANZOVA = "#9a6200"
SEDY_TEXT = "#5b6b75"

# Třetí panel vrstevnic: tatáž pekárna, jen s 3,5× strmější slevou. Střed
# vrstevnic (optimum bez omezení) je pak P / (2 k D) a leží uvnitř oblasti.
K_STRMA = 3.5
STRED_STRMA = (P_PEKARNA[0] / (2 * K_STRMA * D_PEKARNA[0]),
               P_PEKARNA[1] / (2 * K_STRMA * D_PEKARNA[1]))      # (80; 47,6)

MAPA_PLOCH = LinearSegmentedColormap.from_list(
    "omm-plochy", ["#eaf6fb", "#7fc4d8", "#2f6f92", MODRA]
)


def uloz(fig, jmeno: str) -> None:
    """Uloží obrázek jako SVG; s proměnnou OMM_NAHLEDY navíc PNG na kontrolu."""
    fig.savefig(OUT / f"{jmeno}.svg", metadata=SVG_METADATA, bbox_inches="tight")
    nahledy = os.environ.get("OMM_NAHLEDY")
    if nahledy:
        adresar = Path(nahledy)
        adresar.mkdir(parents=True, exist_ok=True)
        fig.savefig(adresar / f"{jmeno}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def kde_linearita_praska() -> None:
    """Tři situace, kde lineární model končí: čtverce, zakřivené omezení, exponenciála."""
    fig, osy = plt.subplots(1, 3, figsize=(11.0, 3.9))

    # --- (a) chyba se měří kvadraticky ---------------------------------------
    ctverce = osy[0]
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.array([3.5, 3.9, 7.3, 8.0, 9.6, 13.4, 14.9])
    smernice, posun = np.polyfit(x, y, 1)
    mrizka = np.linspace(0.4, 7.6, 50)
    ctverce.plot(mrizka, smernice * mrizka + posun, color=MODRA, linewidth=2.6, zorder=3)
    for xi, yi in zip(x, y):
        ctverce.plot([xi, xi], [yi, smernice * xi + posun], color=CERVENA,
                     linewidth=2.2, zorder=2)
    ctverce.scatter(x, y, s=60, color=TYRKYSOVA, zorder=4)
    ctverce.annotate("odchylka $r_i$", xy=(6.05, 12.9), xytext=(0.7, 14.3),
                     arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4},
                     color=CERVENA, fontsize=13.5, weight="bold", bbox=STITEK)
    ctverce.text(7.6, 1.2, "účel $\\sum_i r_i^2$", color=MODRA, ha="right",
                 fontsize=14, weight="bold", bbox=STITEK)
    ctverce.set(xlabel="koncentrace [mmol/l]", ylabel="signál [mV]",
                xlim=(0.2, 7.8), ylim=(0, 16.5), yticks=[0, 5, 10, 15])
    ctverce.set_title("chyba se měří kvadraticky", fontsize=14.5, color=MODRA,
                      weight="bold")

    # --- (b) zakřivené omezení -----------------------------------------------
    kruh = osy[1]
    kruh.add_patch(Circle((0, 0), 1.0, facecolor=SVETLE_MODRA, edgecolor=TYRKYSOVA,
                          linewidth=3.0, zorder=2))
    kruh.add_patch(Polygon([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                           closed=True, facecolor="none", edgecolor="#9aa7ae",
                           linewidth=1.5, linestyle="--", zorder=3))
    for vrchol in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        kruh.scatter(*vrchol, s=34, color="#9aa7ae", zorder=3)
    kruh.text(0, 0.16, "$x_1^2+x_2^2\\leq E$", color=TYRKYSOVA, fontsize=15,
              weight="bold", ha="center", zorder=5, bbox=STITEK)
    kruh.text(0, -0.40, "žádné vrcholy", color=MODRA, fontsize=13.5,
              ha="center", zorder=5, bbox=STITEK)
    kruh.text(1.08, 1.12, "polyedr\nby měl vrcholy", color=SEDY_TEXT, fontsize=11.5,
              ha="right", va="bottom", zorder=5, linespacing=1.1)
    kruh.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-1.4, 1.4), ylim=(-1.3, 1.62),
             aspect="equal", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    kruh.set_title("zakřivené omezení", fontsize=14.5, color=MODRA, weight="bold")

    # --- (c) nelineární model -------------------------------------------------
    vymyti = osy[2]
    cas = np.linspace(0, 12, 300)
    vymyti.plot(cas, 100 - 8.0 * cas, color=MODRA, linewidth=1.8, linestyle="--",
                alpha=0.6, zorder=2)
    vymyti.plot(cas, 100 * np.exp(-0.28 * cas), color=CERVENA, linewidth=3.2, zorder=3)
    mereni_t = np.array([0.5, 1.5, 3.0, 4.5, 6.0, 8.0, 10.5])
    mereni_c = 100 * np.exp(-0.28 * mereni_t) + np.array([4, -6, 5, -3, 4, -2, 3])
    vymyti.scatter(mereni_t, mereni_c, s=56, color=TYRKYSOVA, zorder=4)
    vymyti.text(4.3, 78, "$c(t)=c_0\\,\\mathrm{e}^{-kt}$", color=CERVENA, fontsize=15,
                weight="bold", bbox=STITEK)
    vymyti.text(4.3, 52, "hledané $k$\nje v exponentu", color=CERVENA, fontsize=13,
                bbox=STITEK, linespacing=1.15)
    vymyti.text(8.9, 33, "přímka", color=MODRA, fontsize=12, rotation=-33,
                rotation_mode="anchor", alpha=0.8)
    vymyti.set(xlabel="čas [min]", ylabel="koncentrace [%]", xlim=(0, 12), ylim=(0, 112))
    vymyti.set_title("nelineární model", fontsize=14.5, color=MODRA, weight="bold")

    for osa in osy:
        osa.grid(alpha=0.22)
        osa.tick_params(labelsize=11.5)
        osa.xaxis.label.set_size(12.5)
        osa.yaxis.label.set_size(12.5)

    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.6)
    popisky = ("kalibrace, rekonstrukce CT",
               "limit dávky nebo energie",
               "vymývání kontrastní látky")
    for osa, text, barva in zip(osy, popisky, (MODRA, TYRKYSOVA, CERVENA)):
        ramec = osa.get_position()
        fig.text(0.5 * (ramec.x0 + ramec.x1), 0.045, text, ha="center", va="center",
                 fontsize=13.5, weight="bold", color="white",
                 bbox={"facecolor": barva, "edgecolor": "none",
                       "boxstyle": "round,pad=0.35"})
    uloz(fig, "kde-linearita-praska")


def vrstevnice_posun_gif() -> None:
    """Animace: vrstevnice zisku se posouvá, dokud se oblasti ještě dotýká.

    Tři panely nad stejnou přípustnou oblastí pekárny: bez slevy (lineární
    zisk, poslední dotyk ve vrcholu), dnešní sleva (dotyk uprostřed hrany
    mouky) a 3,5× strmější sleva (vrstevnice se stáhne do bodu uvnitř).
    První snímek je výsledný stav, pak se pohyb přehraje od začátku.
    """
    xmin, xmax, ymin, ymax = -18.0, 368.0, -28.0, 348.0
    t = np.linspace(xmin, xmax, 200)
    uhel = np.linspace(0, 2 * np.pi, 400)
    q1, q2 = D_PEKARNA        # tvar elips: 0,025 (x1 - s1)^2 + 0,024 (x2 - s2)^2 = u

    def elipsa(stred, polomer):
        """Body vrstevnice 0,025 dx1^2 + 0,024 dx2^2 = polomer^2."""
        return (stred[0] + polomer / np.sqrt(q1) * np.cos(uhel),
                stred[1] + polomer / np.sqrt(q2) * np.sin(uhel))

    def polomer_dotyku(stred):
        return np.sqrt(q1 * (X_QP[0] - stred[0]) ** 2 + q2 * (X_QP[1] - stred[1]) ** 2)

    fig, osy = plt.subplots(1, 3, figsize=(11.6, 4.45))
    for osa in osy:
        osa.add_patch(Polygon(POLYEDR, closed=True, facecolor="#cdecee",
                              edgecolor=TYRKYSOVA, linewidth=2.4, zorder=2))
        for vrchol in POLYEDR:
            osa.scatter(*vrchol, s=34, color=TYRKYSOVA, zorder=5)
        osa.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal",
                xticks=[0, 100, 200, 300], yticks=[0, 100, 200, 300])
        osa.set_xlabel("chleby $x_1$ [ks/den]", fontsize=12.5)
        osa.tick_params(labelsize=11)
        osa.grid(alpha=0.18)
    osy[0].set_ylabel("bagety $x_2$ [ks/den]", fontsize=12.5)

    # --- (a) bez slevy: lineární zisk ---------------------------------------
    prima = osy[0]
    for z in (900, 1700):
        prima.plot(t, (z - 14 * t) / 8, color=MODRA, linewidth=1.2, linestyle=":",
                   zorder=3)
    cara_lp, = prima.plot([], [], color=CERVENA, linewidth=3.0, zorder=4)
    prima.add_patch(FancyArrowPatch((30, 40), (96, 78), arrowstyle="-|>",
                                    mutation_scale=18, linewidth=2.6, color=ORANZOVA,
                                    zorder=5))
    prima.text(24, 6, "víc zisku", color=TMAVE_ORANZOVA, fontsize=12,
               weight="bold", zorder=8, bbox=STITEK)
    hvezda_lp = prima.scatter(*X_LP, s=300, marker="*", color=CERVENA, edgecolor="white",
                              linewidth=1.1, zorder=7)
    popis_lp = prima.annotate("$(100,150)$\nvrchol", xy=X_LP, xytext=(150, 222),
                              fontsize=13, weight="bold", color=CERVENA, zorder=8,
                              bbox=STITEK, linespacing=1.1,
                              arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4})
    prima.set_title("bez slevy (minule)", fontsize=14, color=MODRA, weight="bold")

    # --- (b) dnešní sleva: dotyk uprostřed hrany ------------------------------
    hrana = osy[1]
    for u in (0.45, 1.45):
        hrana.plot(*elipsa(X_NEOMEZENE, u * polomer_dotyku(X_NEOMEZENE)), color=MODRA,
                   linewidth=1.2, linestyle=":", zorder=3)
    hrana.plot([160, 100], [0, 150], color=ORANZOVA, linewidth=5.5,
               solid_capstyle="round", alpha=0.95, zorder=3)
    hrana.text(8, 30, "mouka\naktivní", color=TMAVE_ORANZOVA, fontsize=12,
               weight="bold", zorder=8, bbox=STITEK, linespacing=1.1)
    elipsa_b, = hrana.plot([], [], color=CERVENA, linewidth=3.0, zorder=4)
    hrana.scatter(*X_NEOMEZENE, s=80, facecolor="white", edgecolor=MODRA,
                  linewidth=2.2, zorder=6)
    hrana.text(X_NEOMEZENE[0], X_NEOMEZENE[1] + 22, "střed\n(nepřípustný)", color=MODRA,
               fontsize=11.5, ha="center", va="bottom", zorder=8, bbox=STITEK,
               linespacing=1.05)
    hvezda_b = hrana.scatter(*X_QP, s=300, marker="*", color=CERVENA, edgecolor="white",
                             linewidth=1.1, zorder=7)
    popis_b = hrana.annotate("$(120,100)$\nuvnitř hrany", xy=X_QP, xytext=(8, 236),
                             fontsize=13, weight="bold", color=CERVENA, zorder=8,
                             bbox=STITEK, linespacing=1.1,
                             arrowprops={"arrowstyle": "->", "color": CERVENA, "lw": 1.4})
    hrana.set_title("dnešní sleva", fontsize=14, color=MODRA, weight="bold")

    # --- (c) strmá sleva: střed uvnitř oblasti --------------------------------
    vnitrek = osy[2]
    for u in (0.35, 0.75):
        vnitrek.plot(*elipsa(STRED_STRMA, u * polomer_dotyku(X_NEOMEZENE)),
                     color=MODRA, linewidth=1.2, linestyle=":", zorder=3)
    elipsa_c, = vnitrek.plot([], [], color=CERVENA, linewidth=3.0, zorder=4)
    hvezda_c = vnitrek.scatter(*STRED_STRMA, s=300, marker="*", color=CERVENA,
                               edgecolor="white", linewidth=1.1, zorder=7)
    popis_c = vnitrek.annotate("$(80;\\,47{,}6)$\nuvnitř oblasti", xy=STRED_STRMA,
                               xytext=(150, 212), fontsize=13, weight="bold",
                               color=CERVENA, zorder=8, bbox=STITEK, linespacing=1.1,
                               arrowprops={"arrowstyle": "->", "color": CERVENA,
                                           "lw": 1.4})
    vnitrek.text(360, 318, "nic není aktivní", color=TYRKYSOVA, fontsize=12.5,
                 ha="right", weight="bold", zorder=8, bbox=STITEK)
    vnitrek.set_title("3,5× strmější sleva", fontsize=14, color=MODRA, weight="bold")

    fig.tight_layout(w_pad=0.8)

    # Průběh: 0 = vrstevnice daleko (malý zisk), 1 = poslední dotyk.
    polomer_b_start = 1.9 * polomer_dotyku(X_NEOMEZENE)
    polomer_c_start = 1.25 * polomer_dotyku(X_NEOMEZENE)

    def nastav(s: float) -> None:
        hladke = 0.5 - 0.5 * np.cos(np.pi * s)
        z = 700 + (2600 - 700) * hladke
        cara_lp.set_data(t, (z - 14 * t) / 8)
        elipsa_b.set_data(*elipsa(X_NEOMEZENE, polomer_b_start
                                  + (polomer_dotyku(X_NEOMEZENE) - polomer_b_start) * hladke))
        elipsa_c.set_data(*elipsa(STRED_STRMA, max(polomer_c_start * (1 - hladke), 0.0)))
        hotovo = s >= 1.0
        for prvek in (hvezda_lp, popis_lp, hvezda_b, popis_b, hvezda_c, popis_c):
            prvek.set_visible(hotovo)

    def snimek(s: float):
        nastav(s)
        fig.canvas.draw()
        return copy.deepcopy(fig)

    kroky = np.linspace(0.0, 1.0, 22)
    snimky = [snimek(1.0)] + [snimek(s) for s in kroky]
    doby = [2600] + [110] * (len(kroky) - 1) + [4200]
    plt.close(fig)
    uloz_gif(snimky, doby, "vrstevnice-posun", dpi=100)


def tri_plochy() -> None:
    """Miska, sedlo a žlab: co dělá matice Q se zakřivením."""
    fig = plt.figure(figsize=(11.0, 3.55))
    osa1d = np.linspace(-1.0, 1.0, 27)
    xx, yy = np.meshgrid(osa1d, osa1d)
    cara = np.linspace(-1.0, 1.0, 60)

    plochy = (
        ((2.0, 0.0, 2.0), (0.0, 2.1), "miska",
         "$Q$ pozitivně definitní\nnahoru v každém směru"),
        ((2.0, 0.0, -2.0), (-1.05, 1.05), "sedlo",
         "$Q$ indefinitní\nnahoru i dolů — není minimum"),
        ((2.0, 0.0, 0.0), (0.0, 1.05), "žlab",
         "$Q$ pozitivně semidefinitní\nv jednom směru rovně"),
    )

    for i, ((q11, q12, q22), meze_z, nadpis, popis) in enumerate(plochy):
        osa = fig.add_subplot(1, 3, i + 1, projection="3d")
        zz = 0.5 * (q11 * xx**2 + 2 * q12 * xx * yy + q22 * yy**2)
        osa.plot_surface(xx, yy, zz, cmap=MAPA_PLOCH, linewidth=0, antialiased=True,
                         alpha=0.92, rstride=1, cstride=1, zorder=1)
        osa.plot(cara, np.zeros_like(cara), 0.5 * q11 * cara**2, color=CERVENA,
                 linewidth=3.6, zorder=6)
        osa.plot(np.zeros_like(cara), cara, 0.5 * q22 * cara**2, color=ORANZOVA,
                 linewidth=3.6, zorder=6)
        osa.set(xlim=(-1, 1), ylim=(-1, 1), zlim=meze_z,
                xticks=[], yticks=[], zticks=[])
        osa.set_xlabel("$x_1$", fontsize=13, labelpad=-10)
        osa.set_ylabel("$x_2$", fontsize=13, labelpad=-10)
        osa.view_init(elev=24, azim=-58)
        osa.set_box_aspect((1, 1, 0.78), zoom=1.2)
        osa.set_title(nadpis, fontsize=16, color=MODRA, weight="bold", pad=-6)
        fig.text((2 * i + 1) / 6, 0.19, popis, ha="center", va="top",
                 fontsize=13, color="#3d4a55", linespacing=1.3)

    fig.subplots_adjust(left=0.005, right=0.995, top=1.0, bottom=0.19, wspace=0.0)
    uloz(fig, "tri-plochy")


def mapa_trid() -> None:
    """Vnořené třídy úloh LP ⊂ QP ⊂ konvexní NLP ⊂ obecné NLP."""
    fig, osa = plt.subplots(figsize=(11.4, 4.3))
    osa.set(xlim=(0, 12.2), ylim=(0.3, 4.85))
    osa.axis("off")

    vrstvy = (
        (6.6, 4.30, "obecné NLP", "$f$, $g_i$ libovolné", SEDIVA, "#6f7f8a", 1.6,
         "jen lokální optimum,\nzáleží na startu"),
        (5.6, 3.60, "konvexní NLP", "konvexní $f$ i oblast", "#e3f2f6", TYRKYSOVA, 1.9,
         "lokální = globální,\nspolehlivě"),
        (4.6, 2.90, "QP", "$\\frac{1}{2}\\mathbf{x}^\\mathsf{T}Q\\mathbf{x}"
         "+\\mathbf{c}^\\mathsf{T}\\mathbf{x}$", "#c9e7ec", "#00666c", 2.1,
         "při $Q\\succeq0$ rychle\na s certifikátem"),
        (3.6, 2.20, "LP", "$\\mathbf{c}^\\mathsf{T}\\mathbf{x}$", "#a7d8e6", MODRA, 2.4,
         "simplex, vnitřní bod;\nmiliony proměnných"),
    )

    stred_x, dolni_y = 3.45, 1.22
    for sirka, horni_y, nazev, vzorec, vypln, obrys, sila, poznamka in vrstvy:
        x0 = stred_x - sirka / 2
        osa.add_patch(FancyBboxPatch(
            (x0, dolni_y), sirka, horni_y - dolni_y,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor=vypln, edgecolor=obrys, linewidth=sila, zorder=2,
        ))
        osa.text(x0 + 0.2, horni_y - 0.25, nazev, ha="left", va="center",
                 fontsize=15, weight="bold", color=obrys, zorder=4)
        osa.text(x0 + sirka - 0.2, horni_y - 0.25, vzorec, ha="right", va="center",
                 fontsize=13, color=obrys, zorder=4)
        osa.plot([x0 + sirka, 7.05], [horni_y - 0.25, horni_y - 0.25],
                 color=obrys, linewidth=1.1, linestyle=":", zorder=1)
        osa.text(7.15, horni_y - 0.25, poznamka, ha="left", va="center",
                 fontsize=13.5, color=obrys, linespacing=1.15, zorder=4)

    # Hranice, na které opravdu záleží: konvexní × nekonvexní.
    sirka_konv, horni_konv = vrstvy[1][0], vrstvy[1][1]
    osa.add_patch(FancyBboxPatch(
        (stred_x - sirka_konv / 2 - 0.1, dolni_y - 0.1), sirka_konv + 0.2,
        horni_konv - dolni_y + 0.2, boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor="none", edgecolor=CERVENA, linewidth=2.4, linestyle="--", zorder=5))
    osa.text(12.1, 4.62, "- - -  hranice, na které záleží: konvexní × nekonvexní",
             ha="right", va="center", fontsize=13, color=CERVENA, weight="bold", zorder=6)
    osa.text(0.15, 4.62, "čím dál od středu, tím míň záruk", ha="left", va="center",
             fontsize=13, color=SEDY_TEXT, style="italic")

    osa.text(6.1, 0.55,
             "LP: pekárna z minula   ·   QP: pekárna se slevou, ridge, SVM   ·   "
             "NLP: proložení exponenciálou",
             ha="center", va="center", fontsize=13.5, color=MODRA, zorder=4)
    uloz(fig, "mapa-trid")


if __name__ == "__main__":
    kde_linearita_praska()
    vrstevnice_posun_gif()
    tri_plochy()
    mapa_trid()
