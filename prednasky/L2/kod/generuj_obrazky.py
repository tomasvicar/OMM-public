"""Vytvoří ilustrace a grafy pro druhou přednášku.

Většina obrázků je vektorová (SVG); dva slidy, kde sdělení nese pohyb, jsou
animované smyčky (GIF) — viz `uloz_gif`.
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from PIL import Image


OUT = Path(__file__).resolve().parents[1] / "obrazky"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "mpc-omm-l2"
SVG_METADATA = {"Date": None}

MODRA = "#12355b"
TYRKYSOVA = "#007f86"
CERVENA = "#c73e1d"
ORANZOVA = "#e9a23b"
SVETLE_MODRA = "#d9eef7"
SEDIVA = "#eef2f4"


def uloz_gif(snimky, doby, jmeno: str, dpi: int = 120) -> None:
    """Uloží posloupnost figur jako smyčkovaný GIF.

    `snimky` jsou hotové figury v pořadí přehrávání, `doby` jejich doby
    zobrazení v milisekundách. **První snímek je zároveň statickou zálohou**
    (v PDF nebo v tisku se ukáže jen on), takže do něj patří výsledný stav.
    """
    obrazky = []
    for fig in snimky:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        obrazky.append(Image.open(buffer).convert("RGB")
                       .quantize(colors=96, method=Image.MEDIANCUT))
    obrazky[0].save(OUT / f"{jmeno}.gif", save_all=True, append_images=obrazky[1:],
                    duration=list(doby), loop=0, optimize=True, disposal=2)


def linearita() -> None:
    """Tři závislosti vedle sebe: přímka, množstevní sleva a saturace."""
    fig, osy = plt.subplots(1, 3, figsize=(12.8, 3.6))
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}
    verdikty = []

    prima = osy[0]
    chleby = np.linspace(0, 120, 100)
    prima.plot(chleby, 0.5 * chleby, color=TYRKYSOVA, linewidth=3)
    for kusy in (40, 80):
        prima.plot([kusy, kusy, 0], [0, 0.5 * kusy, 0.5 * kusy], color=MODRA,
                   linewidth=1.1, linestyle=":", zorder=1)
        prima.scatter([kusy], [0.5 * kusy], s=55, color=CERVENA, zorder=4)
    prima.text(12, 44, "2× chleby\n= 2× mouka", color=CERVENA, fontsize=11,
               weight="bold", bbox=stitek)
    prima.set(xlabel="chleby $x_1$ [ks]", ylabel="mouka [kg]", xlim=(0, 120), ylim=(0, 62))
    prima.set_title("spotřeba mouky", fontsize=12.5, color=MODRA)
    verdikty.append("✓ lineární — patří do LP")

    sleva = osy[1]
    mnozstvi = np.linspace(0, 100, 400)
    cena = np.where(mnozstvi <= 40, 10 * mnozstvi, 400 + 7 * (mnozstvi - 40))
    sleva.plot(mnozstvi, 10 * mnozstvi, color=MODRA, linewidth=1.6, linestyle="--",
               alpha=0.65)
    sleva.plot(mnozstvi, cena, color=CERVENA, linewidth=3)
    sleva.scatter([40], [400], s=60, color=CERVENA, zorder=4)
    sleva.text(44, 150, "zlom: od 40 kg\nlevnější sazba", color=CERVENA, fontsize=11,
               weight="bold", bbox=stitek)
    sleva.text(64, 780, "bez slevy", color=MODRA, fontsize=10.5, rotation=30,
               rotation_mode="anchor", bbox=stitek)
    sleva.set(xlabel="nakoupená mouka [kg]", ylabel="cena [Kč]", xlim=(0, 100), ylim=(0, 1000))
    sleva.set_title("množstevní sleva", fontsize=12.5, color=MODRA)
    verdikty.append("✗ lomená čára")

    saturace = osy[2]
    davka = np.linspace(0, 10, 300)
    odezva = 100 * davka / (2 + davka)
    saturace.plot(davka, 50 * davka, color=MODRA, linewidth=1.6, linestyle="--", alpha=0.65)
    saturace.plot(davka, odezva, color=CERVENA, linewidth=3)
    saturace.axhline(100, color="#52616b", linewidth=1.1, linestyle=":")
    saturace.text(3.0, 118, "všechny receptory obsazené", color="#52616b", fontsize=10.5,
                  va="center", bbox=stitek)
    saturace.text(1.5, 40, "úměrná odezva\njen na začátku", color=MODRA, fontsize=10.5,
                  bbox=stitek)
    saturace.set(xlabel="dávka [mg]", ylabel="odezva [%]", xlim=(0, 10), ylim=(0, 132))
    saturace.set_title("saturace receptoru", fontsize=12.5, color=MODRA)
    verdikty.append("✗ zakřivená")

    for osa in osy:
        osa.grid(alpha=0.25)
        osa.tick_params(labelsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for osa, text, barva in zip(osy, verdikty, (TYRKYSOVA, CERVENA, CERVENA)):
        ramec = osa.get_position()
        fig.text(0.5 * (ramec.x0 + ramec.x1), 0.06, text, ha="center", va="center",
                 fontsize=11.5, weight="bold", color="white",
                 bbox={"facecolor": barva, "edgecolor": "none",
                       "boxstyle": "round,pad=0.35"})
    fig.savefig(OUT / "linearita.svg", metadata=SVG_METADATA)
    plt.close(fig)


def matice_tabulka() -> None:
    """Tabulka receptur se mění v matici A: řádek = surovina, sloupec = výrobek."""

    def zavorky(osa, x0, x1, y0, y1, delka=0.16, barva=MODRA, sirka=2.0):
        osa.plot([x0, x0], [y0, y1], color=barva, lw=sirka, solid_capstyle="round")
        osa.plot([x1, x1], [y0, y1], color=barva, lw=sirka, solid_capstyle="round")
        for y in (y0, y1):
            osa.plot([x0, x0 + delka], [y, y], color=barva, lw=sirka, solid_capstyle="round")
            osa.plot([x1 - delka, x1], [y, y], color=barva, lw=sirka, solid_capstyle="round")

    fig, ax = plt.subplots(figsize=(12.0, 3.85))
    ax.set(xlim=(0, 12), ylim=(0.33, 4.05))
    ax.axis("off")

    hodnoty = (("0,5", "0,2"), ("3", "2"))
    barvy_radku = (SVETLE_MODRA, SEDIVA)
    barvy_sloupcu = (TYRKYSOVA, CERVENA)
    vyrobky = ("chléb $x_1$", "bageta $x_2$")
    suroviny = ("mouka\n[kg / ks]", "pec\n[min / ks]")

    # --- tabulka receptur ---------------------------------------------------
    tab_x = (0.20, 1.75, 3.00, 4.25)
    tab_y = (1.50, 2.20, 2.90, 3.50)
    ax.text(2.22, 3.76, "tabulka receptur", ha="center", va="center",
            fontsize=12.5, weight="bold", color=MODRA)

    for i in range(2):
        ax.add_patch(Rectangle(
            (tab_x[0], tab_y[1 - i]), tab_x[3] - tab_x[0], tab_y[2 - i] - tab_y[1 - i],
            facecolor=barvy_radku[i], edgecolor="none", zorder=0,
        ))
        ax.text((tab_x[0] + tab_x[1]) / 2, (tab_y[1 - i] + tab_y[2 - i]) / 2, suroviny[i],
                ha="center", va="center", fontsize=11.5, color=MODRA)
        for j in range(2):
            ax.text((tab_x[j + 1] + tab_x[j + 2]) / 2, (tab_y[1 - i] + tab_y[2 - i]) / 2,
                    hodnoty[i][j], ha="center", va="center", fontsize=14, color=MODRA)

    for j in range(2):
        ax.text((tab_x[j + 1] + tab_x[j + 2]) / 2, (tab_y[2] + tab_y[3]) / 2, vyrobky[j],
                ha="center", va="center", fontsize=12, weight="bold", color=barvy_sloupcu[j])

    for x in tab_x:
        ax.plot([x, x], [tab_y[0], tab_y[3]], color="#b9ccd4", lw=1.1, zorder=2)
    for y in tab_y:
        ax.plot([tab_x[0], tab_x[3]], [y, y], color="#b9ccd4", lw=1.1, zorder=2)

    # --- šipka --------------------------------------------------------------
    ax.add_patch(FancyArrowPatch((4.50, 2.30), (5.35, 2.30), arrowstyle="-|>",
                                 mutation_scale=20, lw=4, color=ORANZOVA))
    ax.text(4.92, 2.75, "necháme\njen čísla", ha="center", va="center",
            fontsize=11, color="#8a6100")

    # --- matice A -----------------------------------------------------------
    mat_x = (6.15, 7.25, 8.35)
    mat_y = (1.55, 2.25, 2.95)
    ax.text(6.00, 2.25, "$A=$", ha="right", va="center", fontsize=17, color=MODRA)

    for i in range(2):
        ax.add_patch(Rectangle(
            (mat_x[0], mat_y[1 - i]), mat_x[2] - mat_x[0], mat_y[2 - i] - mat_y[1 - i],
            facecolor=barvy_radku[i], edgecolor="none", zorder=0,
        ))
        for j in range(2):
            ax.text((mat_x[j] + mat_x[j + 1]) / 2, (mat_y[1 - i] + mat_y[2 - i]) / 2,
                    hodnoty[i][j], ha="center", va="center", fontsize=16, color=MODRA)
    zavorky(ax, mat_x[0] - 0.12, mat_x[2] + 0.12, mat_y[0] - 0.05, mat_y[2] + 0.05)

    for j in range(2):
        ax.text((mat_x[j] + mat_x[j + 1]) / 2, 3.18, vyrobky[j], ha="center", va="bottom",
                fontsize=12, weight="bold", color=barvy_sloupcu[j])
        ax.add_patch(FancyArrowPatch(
            ((mat_x[j] + mat_x[j + 1]) / 2, 3.14), ((mat_x[j] + mat_x[j + 1]) / 2, 3.00),
            arrowstyle="-|>", mutation_scale=13, lw=1.6, color=barvy_sloupcu[j]))
    ax.text(7.25, 3.85, "sloupec = výrobek (proměnná)", ha="center", va="center",
            fontsize=12.5, weight="bold", color=MODRA)

    popisky_radku = ("mouka [kg]", "pec [min]")
    for i in range(2):
        stred = (mat_y[1 - i] + mat_y[2 - i]) / 2
        ax.add_patch(FancyArrowPatch((8.60, stred), (9.05, stred), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.6, color=MODRA))
        ax.text(9.20, stred, popisky_radku[i], ha="left", va="center",
                fontsize=12, color=MODRA)
    ax.text(8.95, 3.35, "řádek = surovina\n(omezení)", ha="left", va="center",
            fontsize=12.5, weight="bold", color=MODRA, linespacing=1.25)

    ax.add_patch(FancyBboxPatch(
        (0.55, 0.45), 10.9, 0.62, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="#eef7f8", edgecolor="#cfe6e8", lw=1.4, zorder=1,
    ))
    ax.text(6.0, 0.76,
            "$A_{ij}$ = kolik suroviny $i$ spotřebuje jeden kus výrobku $j$;"
            "   zde $A_{12}=0{,}2$, tedy 0,2 kg mouky na jednu bagetu",
            ha="center", va="center", fontsize=12.5, color=MODRA, zorder=2)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "matice-tabulka.svg", metadata=SVG_METADATA)
    plt.close(fig)


def soucin_ax() -> None:
    """Součin Ax dvěma pohledy: po řádcích a po sloupcích."""

    def zavorky(osa, x0, x1, y0, y1, delka=0.16, barva=MODRA, sirka=2.0):
        osa.plot([x0, x0], [y0, y1], color=barva, lw=sirka, solid_capstyle="round")
        osa.plot([x1, x1], [y0, y1], color=barva, lw=sirka, solid_capstyle="round")
        for y in (y0, y1):
            osa.plot([x0, x0 + delka], [y, y], color=barva, lw=sirka, solid_capstyle="round")
            osa.plot([x1 - delka, x1], [y, y], color=barva, lw=sirka, solid_capstyle="round")

    fig, osy = plt.subplots(1, 2, figsize=(12.6, 4.0))
    for osa in osy:
        osa.set(xlim=(0, 12), ylim=(0.62, 4.88))
        osa.axis("off")

    barvy_radku = (SVETLE_MODRA, SEDIVA)
    y_horni, y_stred, y_dolni = 3.35, 2.65, 1.95
    stredy_radku = (3.00, 2.30)

    # ------------------------------------------------------------------ řádky
    radky = osy[0]
    radky.text(6.0, 4.55, "po řádcích:  $(A\\mathbf{x})_i = \\mathbf{a}_i^{\\mathsf{T}}\\mathbf{x}$", ha="center",
               va="center", fontsize=15, weight="bold", color=MODRA)
    radky.text(6.0, 4.05, "řádek matice krát vektor $\\mathbf{x}$ dá jedno číslo",
               ha="center", va="center", fontsize=12, color="#52616b", style="italic")

    for i in range(2):
        radky.add_patch(Rectangle((0.58, y_dolni + (1 - i) * 0.70), 8.85, 0.70,
                                  facecolor=barvy_radku[i], edgecolor="none", zorder=0))

    a_hodnoty = (("0,5", "0,2"), ("3", "2"))
    a_x = (0.60, 1.60, 2.60)
    for i in range(2):
        for j in range(2):
            radky.text((a_x[j] + a_x[j + 1]) / 2, stredy_radku[i], a_hodnoty[i][j],
                       ha="center", va="center", fontsize=15, color=MODRA)
    zavorky(radky, a_x[0] - 0.12, a_x[2] + 0.12, y_dolni + 0.05, y_horni - 0.05)

    for i, znak in enumerate(("$x_1$", "$x_2$")):
        radky.text(3.55, stredy_radku[i], znak, ha="center", va="center",
                   fontsize=15, color=MODRA)
    zavorky(radky, 3.15, 3.95, y_dolni + 0.05, y_horni - 0.05)

    radky.text(4.32, y_stred, "$=$", ha="center", va="center", fontsize=17, color=MODRA)

    vysledek = ("$0{,}5\\,x_1+0{,}2\\,x_2$", "$3\\,x_1+2\\,x_2$")
    for i in range(2):
        radky.text(7.07, stredy_radku[i], vysledek[i], ha="center", va="center",
                   fontsize=15, color=MODRA)
    zavorky(radky, 4.70, 9.45, y_dolni + 0.05, y_horni - 0.05)

    popisky = ("spotřeba mouky", "vytížení pece")
    for i in range(2):
        radky.add_patch(FancyArrowPatch((9.70, stredy_radku[i]), (10.15, stredy_radku[i]),
                                        arrowstyle="-|>", mutation_scale=13, lw=1.6,
                                        color=MODRA))
        radky.text(10.30, stredy_radku[i], popisky[i], ha="left", va="center",
                   fontsize=11.5, color=MODRA)

    radky.text(6.0, 1.05, "každý řádek = jedno omezení, tedy jedno číslo na výstupu",
               ha="center", va="center", fontsize=12.5, color=TYRKYSOVA, weight="bold")

    # ---------------------------------------------------------------- sloupce
    sloupce = osy[1]
    sloupce.text(6.0, 4.55, "po sloupcích:  $A\\mathbf{x} = \\sum_j x_j\\,A_{:,j}$", ha="center",
                 va="center", fontsize=15, weight="bold", color=MODRA)
    sloupce.text(6.0, 4.05, "vážený součet celých receptur",
                 ha="center", va="center", fontsize=12, color="#52616b", style="italic")

    sloupce.text(0.95, y_stred, "$A\\mathbf{x}=$", ha="center", va="center", fontsize=17, color=MODRA)

    sloupcove_hodnoty = (("0,5", "3"), ("0,2", "2"))
    barvy_sloupcu = (TYRKYSOVA, CERVENA)
    koeficienty = ("$x_1$", "$x_2$")
    popisky_sloupcu = ("recept\nna 1 chléb", "recept\nna 1 bagetu")
    zacatky = (2.45, 5.85)

    for j, x0 in enumerate(zacatky):
        sloupce.text(x0 - 0.60, y_stred, koeficienty[j], ha="center", va="center",
                     fontsize=16, weight="bold", color=barvy_sloupcu[j])
        sloupce.add_patch(Rectangle((x0, y_dolni), 1.30, 1.40,
                                    facecolor=barvy_sloupcu[j], alpha=0.14,
                                    edgecolor="none", zorder=0))
        for i in range(2):
            sloupce.text(x0 + 0.65, stredy_radku[i], sloupcove_hodnoty[j][i],
                         ha="center", va="center", fontsize=15, color=MODRA)
        zavorky(sloupce, x0 - 0.12, x0 + 1.42, y_dolni + 0.05, y_horni - 0.05,
                barva=barvy_sloupcu[j])
        sloupce.text(x0 + 0.65, 1.55, popisky_sloupcu[j], ha="center", va="center",
                     fontsize=11.5, color=barvy_sloupcu[j], linespacing=1.25)

    sloupce.text(4.60, y_stred, "$+$", ha="center", va="center", fontsize=17, color=MODRA)
    sloupce.text(7.90, y_stred, "$=$", ha="center", va="center", fontsize=17, color=MODRA)

    for i in range(2):
        sloupce.add_patch(Rectangle((8.50, y_dolni + (1 - i) * 0.70), 3.10, 0.70,
                                    facecolor=barvy_radku[i], edgecolor="none", zorder=0))
    for i, text in enumerate(("mouka [kg]", "pec [min]")):
        sloupce.text(10.05, stredy_radku[i], text, ha="center", va="center",
                     fontsize=13, color=MODRA)
    zavorky(sloupce, 8.40, 11.70, y_dolni + 0.05, y_horni - 0.05)

    sloupce.text(6.0, 1.05, "sloupec = celý recept jednoho výrobku",
                 ha="center", va="center", fontsize=12.5, color=TYRKYSOVA, weight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "soucin-ax.svg", metadata=SVG_METADATA)
    plt.close(fig)


def formulace_recept() -> None:
    """Pět kroků od slovního zadání k matici A, vektorům b a c."""
    fig, osa = plt.subplots(figsize=(12.8, 3.5))
    osa.set(xlim=(0, 13.4), ylim=(0, 4.15))
    osa.axis("off")

    kroky = (
        ("1 · PROMĚNNÉ", "Co můžu měnit?", "$x_1,\\dots,x_n$", "jednotka ke každé,\nobvykle $\\mathbf{x}\\geq0$"),
        ("2 · ÚČEL", "Co chci?", "$\\mathbf{c}^\\mathsf{T}\\mathbf{x}$", "co znamená\njeden kus navíc"),
        ("3 · OMEZENÍ", "Co musím dodržet?", "$\\mathbf{a}_i^\\mathsf{T}\\mathbf{x}\\leq\\beta_i$", "jeden zdroj\n= jeden řádek"),
        ("4 · TABULKA", "Kam s koeficienty?", "řádek = omezení\nsloupec = proměnná", "jednotky\nv každém řádku"),
        ("5 · MATICE", "Sedí to?", "$A,\\;\\mathbf{b},\\;\\mathbf{c}$", "kontrola rozměrů\n$(m\\times n)(n\\times 1)$"),
    )

    sirka, mezera, x0 = 2.24, 0.4, 0.3
    for i, (nadpis, otazka, jadro, poznamka) in enumerate(kroky):
        x = x0 + i * (sirka + mezera)
        osa.add_patch(FancyBboxPatch(
            (x, 0.35), sirka, 2.55,
            boxstyle="round,pad=0.1,rounding_size=0.18",
            facecolor="#f7fafb", edgecolor=TYRKYSOVA, linewidth=2, zorder=2,
        ))
        stred = x + sirka / 2
        osa.text(stred, 2.56, nadpis, ha="center", va="center", fontsize=11.5,
                 weight="bold", color=TYRKYSOVA, zorder=3)
        osa.text(stred, 2.10, otazka, ha="center", va="center", fontsize=11,
                 color=MODRA, style="italic", zorder=3)
        osa.text(stred, 1.42, jadro, ha="center", va="center", fontsize=12.5,
                 color=MODRA, zorder=3, linespacing=1.35)
        osa.text(stred, 0.68, poznamka, ha="center", va="center", fontsize=10,
                 color="#52616b", zorder=3, linespacing=1.3)
        if i < len(kroky) - 1:
            osa.add_patch(FancyArrowPatch(
                (x + sirka + 0.05, 1.62), (x + sirka + mezera - 0.05, 1.62),
                arrowstyle="-|>", mutation_scale=15, linewidth=2.6, color=ORANZOVA, zorder=4,
            ))

    osa.add_patch(FancyArrowPatch(
        (0.35, 3.45), (13.05, 3.45), arrowstyle="-|>", mutation_scale=16,
        linewidth=2, color=MODRA, alpha=0.45,
    ))
    osa.text(0.35, 3.78, "slovní zadání („co dnes péct?“)", ha="left", va="center",
             fontsize=11.5, color=MODRA, style="italic")
    osa.text(13.05, 3.78, "hotový model pro solver", ha="right", va="center",
             fontsize=11.5, color=MODRA, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "formulace-recept.svg", metadata=SVG_METADATA)
    plt.close(fig)


# --- radioterapie: táž geometrie jako v první přednášce a na cvičení -------
# Řez, nádor, mícha i profil svazku jsou stejné jako v `prednasky/L1` a v živé
# ukázce `cviceni/C1/kod/ukazka_radioterapie.py`; tady se ale ozařuje jen ze
# tří směrů, aby matice zůstala čitelná a proměnných bylo 36.

POLOH = 12            # příčných poloh svazku v jednom směru
SIGMA = 2.6           # šířka svazku (odchylka gaussovského profilu) [voxel]
ROZSAH = 18.0         # krajní polohy svazku, ±ROZSAH kolem osy [voxel]
UHLY = [0, 135, 270]
MRIZKA = 64


def radioterapie_matice() -> None:
    """Svazky nad řezem pacientem a matice dávek jako heatmapa."""
    yy, xx = np.mgrid[0:MRIZKA, 0:MRIZKA]
    stred = (MRIZKA - 1) / 2
    telo = np.hypot(xx - stred, yy - stred) < 30
    nador = np.hypot(xx - stred - 6, yy - stred + 4) < 7
    micha = np.hypot(xx - stred + 8, yy - stred - 6) < 4

    # Sloupec matice = mapa dávky od svazku s jednotkovou intenzitou.
    sloupce = []
    for uhel in np.deg2rad(UHLY):
        pricne = (xx - stred) * np.cos(uhel) + (yy - stred) * np.sin(uhel)
        hloubka = -(xx - stred) * np.sin(uhel) + (yy - stred) * np.cos(uhel)
        for posun in np.linspace(-ROZSAH, ROZSAH, POLOH):
            profil = np.exp(-0.5 * ((pricne - posun) / SIGMA) ** 2)   # šířka svazku
            utlum = np.exp(-0.02 * (hloubka + 32))                    # útlum s hloubkou
            sloupce.append((profil * utlum * telo).ravel())
    D = np.array(sloupce).T

    matice = np.vstack([D[nador.ravel()], D[micha.ravel()]])
    m_nador, m_micha = int(nador.sum()), int(micha.sum())
    m, n = matice.shape

    fig, osy = plt.subplots(1, 2, figsize=(12.6, 4.35),
                            gridspec_kw={"width_ratios": [1.0, 1.42]})

    # --- levý panel: schéma řezu se svazky ---
    schema = osy[0]
    schema.set(xlim=(-40, 40), ylim=(-42, 38), aspect="equal")
    schema.axis("off")
    telo_kruh = Circle((0, 0), 30, facecolor=SEDIVA, edgecolor=MODRA, linewidth=2, zorder=2)
    schema.add_patch(telo_kruh)
    for uhel in UHLY:
        smer = np.array([np.sin(np.deg2rad(uhel)), -np.cos(np.deg2rad(uhel))])
        kolmice = np.array([smer[1], -smer[0]])
        # kreslí se každý druhý svazek, jinak by se pásy slily do plné plochy
        for posun in np.linspace(-ROZSAH, ROZSAH, POLOH)[::2]:
            pas, = schema.plot(*np.column_stack([-32 * smer + posun * kolmice,
                                                 32 * smer + posun * kolmice]),
                               color="#4ea8de", alpha=0.4, linewidth=1.9 * SIGMA,
                               solid_capstyle="butt", zorder=3)
            pas.set_clip_path(telo_kruh)
        schema.add_patch(FancyArrowPatch(
            -38 * smer, -32 * smer, arrowstyle="-|>", mutation_scale=14,
            linewidth=2.2, color="#1676a5", zorder=6,
        ))
    schema.add_patch(Circle((6, 4), 7, facecolor="#e76f51", edgecolor=CERVENA,
                            linewidth=1.8, zorder=4))
    schema.add_patch(Circle((-8, -6), 4, facecolor="#ffd166", edgecolor="#c98200",
                            linewidth=1.8, zorder=4))
    schema.text(6, 4, "nádor", ha="center", va="center", fontsize=11,
                weight="bold", color="white", zorder=5)
    schema.text(-8, -11.5, "mícha", ha="center", va="top", fontsize=10.5,
                weight="bold", color="#a36700", zorder=5)
    schema.set_title(f"{len(UHLY)} směry × {POLOH} svazků = $n={n}$ intenzit $x_j$",
                     fontsize=11.5, color=MODRA)
    schema.text(0, -40, "řez rozdělíme na voxely — každý voxel je jeden řádek $A$",
                ha="center", fontsize=10.5, color="#52616b", style="italic")

    # --- pravý panel: heatmapa matice ---
    heat = osy[1]
    obraz = heat.imshow(matice, aspect="auto", cmap="YlGnBu", interpolation="nearest",
                        extent=(0, n, m, 0))
    fig.colorbar(obraz, ax=heat, fraction=0.045, label="$A_{ij}$ [Gy na jednotku $x_j$]")
    heat.add_patch(Rectangle((-2.6, 0), 2.0, m_nador, facecolor=CERVENA,
                             clip_on=False, zorder=5))
    heat.add_patch(Rectangle((-2.6, m_nador), 2.0, m_micha, facecolor=ORANZOVA,
                             clip_on=False, zorder=5))
    heat.text(-3.4, m_nador / 2, f"nádor\n{m_nador} řádků", ha="right", va="center",
              fontsize=10.5, color=CERVENA, weight="bold")
    heat.text(-3.4, m_nador + m_micha / 2, f"mícha\n{m_micha} řádků", ha="right", va="center",
              fontsize=10.5, color="#a36700", weight="bold")
    heat.axhline(m_nador, color=MODRA, linewidth=1.4)
    for hranice in range(POLOH, n, POLOH):
        heat.axvline(hranice, color="white", linewidth=1.2, alpha=0.8)
    heat.set_xlabel("svazky $j=1,\\dots,n$")
    heat.set_yticks([])
    heat.set_title(f"$A$ má {m} řádků a {n} sloupců — zápis $A\\mathbf{{x}}\\leq \\mathbf{{b}}$ se nezměnil",
                   fontsize=11.5, color=MODRA)
    fig.tight_layout()
    fig.savefig(OUT / "radioterapie-matice.svg", metadata=SVG_METADATA)
    plt.close(fig)


def l1_vs_mnc() -> None:
    """Animace: odlehlé měření putuje nahoru; nejmenší čtverce se za ním otáčejí, L1 stojí."""
    # Stejná data jako v kod/lp_triky_cvxpy.py; poslední vzorek je ten odlehlý.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_zaklad = np.array([3.2, 5.0, 6.9, 9.1, 10.8, 12.8, 15.0, 17.0])

    def proloz_l1(hodnoty):
        """Optimum L1 leží vždy na přímce dvěma body dat — stačí projít dvojice."""
        nejlepsi = None
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                smernice = (hodnoty[j] - hodnoty[i]) / (x[j] - x[i])
                posun = hodnoty[i] - smernice * x[i]
                ztrata = np.abs(smernice * x + posun - hodnoty).sum()
                if nejlepsi is None or ztrata < nejlepsi[0]:
                    nejlepsi = (ztrata, smernice, posun)
        return nejlepsi[1], nejlepsi[2]

    mrizka = np.linspace(0.4, 8.6, 50)

    def snimek(y8: float, finale: bool):
        y = y_zaklad.copy()
        y[-1] = y8
        a_l1, b_l1 = proloz_l1(y)
        a_mnc, b_mnc = np.polyfit(x, y, 1)

        fig, osa = plt.subplots(figsize=(7.0, 4.8))
        osa.plot(mrizka, a_mnc * mrizka + b_mnc, color=CERVENA, linewidth=2.4,
                 linestyle="--", label="nejmenší čtverce")
        osa.plot(mrizka, a_l1 * mrizka + b_l1, color=TYRKYSOVA, linewidth=2.8,
                 label="norma $L_1$ (LP)")
        osa.scatter(x[:-1], y[:-1], s=70, color=MODRA, zorder=4, label="měření")
        osa.scatter([8.0], [y8], s=150, marker="X", color=CERVENA, zorder=5,
                    edgecolor="white", linewidth=1.0)

        znamenko = "+" if b_mnc >= 0 else "−"
        osa.text(0.62, 25.9,
                 f"nejmenší čtverce:  $y={a_mnc:.2f}\\,x{znamenko}{abs(b_mnc):.2f}$".replace(".", "{,}"),
                 color=CERVENA, fontsize=11.5, weight="bold", va="top",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.6})
        osa.text(0.62, 23.6,
                 f"norma $L_1$:  $y={a_l1:.2f}\\,x+{b_l1:.2f}$".replace(".", "{,}"),
                 color=TYRKYSOVA, fontsize=11.5, weight="bold", va="top",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.6})

        if finale:
            osa.plot([8.0, 8.0], [a_l1 * 8.0 + b_l1, y8], color=CERVENA, linewidth=1.4,
                     linestyle=":", zorder=3)
            osa.text(8.18, 21.0, "zbytek $|r_8|=8$", color=CERVENA, fontsize=10,
                     rotation=90, va="center", ha="center",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
                           "pad": 1.6})
            osa.annotate("odlehlé měření\n(bublina v kyvetě)", xy=(7.85, y8), xytext=(6.9, 21.4),
                         arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA,
                         fontsize=10.5, weight="bold", ha="right", va="center")
        osa.set(xlabel="koncentrace $x$ [mmol/l]", ylabel="signál $y$ [mV]",
                xlim=(0.3, 9.6), ylim=(0, 27))
        osa.grid(alpha=0.25)
        osa.legend(loc="lower right", fontsize=10.5, framealpha=0.94)
        fig.tight_layout()
        return fig

    snimky = [snimek(25.0, True)]                       # statická záloha: výsledek
    doby = [1800]
    for y8 in np.linspace(17.0, 25.0, 9):               # bublina zvedá poslední měření
        snimky.append(snimek(y8, finale=False))
        doby.append(260)
    snimky.append(snimek(25.0, True))
    doby.append(3000)
    uloz_gif(snimky, doby, "l1-vs-mnc", dpi=118)


def celociselne_zaokrouhleni() -> None:
    """Zaokrouhlení řešení LP může být nepřípustné i výrazně horší než optimum."""
    # max 7x1+5x2, x1+2x2<=7 (sestry), 2x1+x2<=6 (sál), x>=0
    fig, osa = plt.subplots(figsize=(6.6, 5.0))
    oblast = Polygon([[0, 0], [3, 0], [5 / 3, 8 / 3], [0, 3.5]], closed=True,
                     facecolor="#cdecee", edgecolor=TYRKYSOVA, linewidth=2, alpha=0.85)
    osa.add_patch(oblast)
    mrizka = np.linspace(-0.2, 4.5, 40)
    osa.plot(mrizka, (7 - mrizka) / 2, color=CERVENA, linewidth=2.0,
             label="sestry: $x_1+2x_2\\leq7$")
    osa.plot(mrizka, 6 - 2 * mrizka, color="#3a8a52", linewidth=2.0,
             label="sál: $2x_1+x_2\\leq6$")

    for i in range(4):
        for j in range(4):
            uvnitr = i + 2 * j <= 7 and 2 * i + j <= 6
            osa.scatter(i, j, s=34 if uvnitr else 22, zorder=3,
                        color=MODRA if uvnitr else "#9aa7ae")

    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8}
    osa.scatter(5 / 3, 8 / 3, s=230, marker="*", color=ORANZOVA, edgecolor=MODRA,
                linewidth=1.0, zorder=6)
    osa.annotate("optimum LP\n$(1{,}67;\\,2{,}67)$, účel $25$", xy=(1.62, 2.72),
                 xytext=(0.1, 4.05), arrowprops={"arrowstyle": "->", "color": "#9a6200"},
                 color="#9a6200", fontsize=10.5, weight="bold", va="top", bbox=stitek)
    osa.scatter(2, 3, s=170, marker="X", color=CERVENA, zorder=6)
    osa.annotate("zaokrouhlení $(2,3)$\nleží mimo oblast", xy=(2.1, 3.0), xytext=(2.45, 3.25),
                 arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA,
                 fontsize=10.5, weight="bold", va="center", bbox=stitek)
    osa.scatter(2, 2, s=150, color=TYRKYSOVA, edgecolor="white", linewidth=1.2, zorder=6)
    osa.annotate("celočíselné optimum\n$(2,2)$, účel $24$", xy=(2.1, 2.0), xytext=(2.62, 1.55),
                 arrowprops={"arrowstyle": "->", "color": TYRKYSOVA}, color=TYRKYSOVA,
                 fontsize=10.5, weight="bold", va="center", bbox=stitek)
    osa.annotate("zaokrouhlení dolů $(1,2)$\núčel jen $17$", xy=(1, 2), xytext=(0.08, 1.36),
                 arrowprops={"arrowstyle": "->", "color": MODRA}, color=MODRA,
                 fontsize=10.5, va="bottom", bbox=stitek)

    osa.set(xlabel="lůžka pro plánované výkony $x_1$", ylabel="lůžka pro akutní příjmy $x_2$",
            xlim=(-0.3, 4.5), ylim=(-0.3, 4.5))
    osa.set_xticks(range(5))
    osa.set_yticks(range(5))
    osa.grid(alpha=0.2)
    osa.legend(loc="upper right", fontsize=10, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(OUT / "celociselne-zaokrouhleni.svg", metadata=SVG_METADATA)
    plt.close(fig)


def polyedr_poloprostory() -> None:
    """Přípustná oblast pekárny jako průnik čtyř polorovin."""
    zelena = "#3a8a52"
    xmin, xmax, ymin, ymax = -22.0, 232.0, -34.0, 360.0
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}

    def polorovina(osa, a, beta, barva, sirka=2.2, pruhlednost=0.3):
        """Vybarví poloprostor {x : aᵀx ≤ beta} a jeho hraniční nadrovinu."""
        t = np.linspace(xmin, xmax, 300)
        if abs(a[1]) > 1e-9:
            hranice = (beta - a[0] * t) / a[1]
            if a[1] > 0:
                osa.fill_between(t, ymin, np.minimum(hranice, ymax), color=barva,
                                 alpha=pruhlednost, linewidth=0)
            else:
                osa.fill_between(t, np.maximum(hranice, ymin), ymax, color=barva,
                                 alpha=pruhlednost, linewidth=0)
            osa.plot(t, hranice, color=barva, linewidth=sirka)
        else:
            hranice = beta / a[0]
            if a[0] > 0:
                osa.axvspan(xmin, hranice, color=barva, alpha=pruhlednost, linewidth=0)
            else:
                osa.axvspan(hranice, xmax, color=barva, alpha=pruhlednost, linewidth=0)
            osa.axvline(hranice, color=barva, linewidth=sirka)

    omezeni = (
        ((0.5, 0.2), 80.0, CERVENA, "$\\mathbf{a}_1^\\mathsf{T}\\mathbf{x}\\leq80$  ·  mouka"),
        ((3.0, 2.0), 600.0, zelena, "$\\mathbf{a}_2^\\mathsf{T}\\mathbf{x}\\leq600$  ·  pec"),
        ((-1.0, 0.0), 0.0, MODRA, "$-x_1\\leq0$  ·  nezápornost"),
        ((0.0, -1.0), 0.0, ORANZOVA, "$-x_2\\leq0$  ·  nezápornost"),
    )

    fig = plt.figure(figsize=(12.4, 5.3))
    mrizka = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 2.05], wspace=0.28, hspace=0.42)
    male = [fig.add_subplot(mrizka[i // 2, i % 2]) for i in range(4)]

    for osa, (a, beta, barva, popis) in zip(male, omezeni):
        polorovina(osa, a, beta, barva, sirka=1.8)
        osa.set_title(popis, fontsize=10.5, color=barva, pad=5)
        osa.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[0, 200], yticks=[0, 300])
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.15)

    velka = fig.add_subplot(mrizka[:, 2])
    velka.add_patch(Polygon([[0, 0], [160, 0], [100, 150], [0, 300]], closed=True,
                            facecolor="#cdecee", edgecolor="none", zorder=2))
    for a, beta, barva, popis in omezeni:
        polorovina(velka, a, beta, barva, sirka=2.8, pruhlednost=0.0)
    for vrchol, posun in (((0, 0), (12, -26)), ((160, 0), (-2, -26)),
                          ((100, 150), (12, 12)), ((0, 300), (10, 8))):
        velka.scatter(*vrchol, s=60, color=MODRA, zorder=5)
        velka.annotate(f"$({vrchol[0]},{vrchol[1]})$", xy=vrchol,
                       xytext=(vrchol[0] + posun[0], vrchol[1] + posun[1]),
                       color=MODRA, fontsize=11, zorder=6, bbox=stitek)
    velka.text(26, 92, "polyedr\n$P=\\{\\mathbf{x}: A\\mathbf{x}\\leq \\mathbf{b}\\}$", color=TYRKYSOVA, weight="bold",
               fontsize=12.5, zorder=6, bbox=stitek)
    velka.text(84, 246, "mouka", color=CERVENA, fontsize=11, weight="bold", zorder=6, bbox=stitek)
    velka.text(150, 96, "pec", color=zelena, fontsize=11, weight="bold", zorder=6, bbox=stitek)
    velka.set_title("průnik čtyř polorovin  =  přípustná oblast", fontsize=12, color=MODRA, pad=7)
    velka.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
              xlabel="chleby $x_1$ [ks/den]", ylabel="bagety $x_2$ [ks/den]")
    velka.tick_params(labelsize=10)
    velka.grid(alpha=0.18)

    fig.savefig(OUT / "polyedr-poloprostory.svg", metadata=SVG_METADATA, bbox_inches="tight")
    plt.close(fig)


def vrstevnice_posun() -> None:
    """Animace: vrstevnice se posouvá ve směru c až k poslednímu dotyku ve vrcholu."""
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.6}
    t = np.linspace(-10, 300, 200)

    def snimek(zisk: float, finale: bool, mimo: bool = False):
        fig, osa = plt.subplots(figsize=(5.6, 5.9))
        osa.add_patch(Polygon([[0, 0], [160, 0], [100, 150], [0, 300]], closed=True,
                              facecolor="#cdecee", edgecolor=TYRKYSOVA, linewidth=2.2,
                              zorder=2))
        osa.set(xlim=(-12, 300), ylim=(-18, 348), aspect="equal",
                xlabel="chleby $x_1$", ylabel="bagety $x_2$",
                xticks=[0, 100, 200], yticks=[0, 100, 200, 300])
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.18)

        osa.add_patch(FancyArrowPatch((30, 40), (108, 84), arrowstyle="-|>",
                                      mutation_scale=18, linewidth=2.6, color=ORANZOVA,
                                      zorder=5))
        osa.text(52, 12, "$\\mathbf{c}=(14,8)$", color="#9a6200", fontsize=12,
                 weight="bold", zorder=6, bbox=stitek)
        osa.text(296, 126, "vrstevnice $\\mathbf{c}^\\mathsf{T}\\mathbf{x}=$ konst.\n"
                           "jsou kolmé na $\\mathbf{c}$",
                 color=MODRA, fontsize=11, ha="right", va="top", zorder=6, bbox=stitek)

        if mimo:
            osa.plot(t, (3400 - 14 * t) / 8, color="#8b95a1", linewidth=2.0,
                     linestyle="--", zorder=3)
            osa.text(216, 268, "3400 Kč:\nuž mimo oblast", color="#5c6672", fontsize=11.5,
                     ha="center", zorder=6, bbox=stitek)

        barva = CERVENA if finale else MODRA
        osa.plot(t, (zisk - 14 * t) / 8, color=barva, linewidth=2.8, zorder=4)
        osa.text(296, 6, f"{zisk:.0f} Kč", color=barva, fontsize=15, weight="bold",
                 ha="right", va="bottom", zorder=8,
                 bbox={"facecolor": "white", "edgecolor": barva, "linewidth": 1.4,
                       "boxstyle": "round,pad=0.32"})

        if finale:
            # vybledlé dřívější polohy, aby i statická záloha ukázala posun
            for drivejsi in (900, 1400, 1900):
                osa.plot(t, (drivejsi - 14 * t) / 8, color=MODRA, linewidth=1.3,
                         linestyle=":", alpha=0.55, zorder=3)
            osa.scatter(100, 150, s=230, marker="*", color=CERVENA, edgecolor="white",
                        linewidth=1.1, zorder=7)
            osa.annotate("$\\mathbf{x}^\\star=(100,150)$", xy=(103, 148), xytext=(168, 214),
                         arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA,
                         weight="bold", fontsize=11.5, zorder=8, bbox=stitek)
            osa.text(-4, 340, "poslední dotyk je vrchol", color=CERVENA, fontsize=12.5,
                     weight="bold", ha="left", va="top", zorder=6, bbox=stitek)
        osa.set_title("Posouváme vrstevnici ve směru $\\mathbf{c}$", fontsize=12.5,
                      color=MODRA)
        fig.tight_layout()
        return fig

    snimky = [snimek(2600, True)]                       # statická záloha: výsledek
    doby = [1600]
    for zisk in range(600, 2601, 200):                  # samotný posun, kulaté částky
        snimky.append(snimek(zisk, finale=(zisk == 2600)))
        doby.append(260)
    snimky.append(snimek(2600, True, mimo=True))        # o kus dál už jsme venku
    doby.append(1400)
    snimky.append(snimek(2600, True))
    doby.append(2600)
    uloz_gif(snimky, doby, "vrstevnice-posun", dpi=150)


def ctyri_konce() -> None:
    """Čtyři možné výsledky lineárního programu."""
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.4}
    oblast = [[0, 0], [8, 0], [6.5, 5], [0, 7]]
    t = np.linspace(-0.6, 11.6, 200)

    fig, osy = plt.subplots(1, 4, figsize=(13.0, 3.7))

    for osa in osy[:2]:
        osa.add_patch(Polygon(oblast, closed=True, facecolor="#cdecee",
                              edgecolor=TYRKYSOVA, linewidth=2.0, zorder=2))

    for uroven in (5, 8, 11):
        osy[0].plot(t, (uroven - t) / 0.9, color=MODRA, linewidth=1.3, linestyle=":", zorder=3)
    osy[0].scatter(6.5, 5, s=210, marker="*", color=CERVENA, edgecolor="white",
                   linewidth=1.0, zorder=5)
    osy[0].add_patch(FancyArrowPatch((1.2, 1.0), (3.4, 2.9), arrowstyle="-|>", mutation_scale=15,
                                     linewidth=2.2, color=ORANZOVA, zorder=4))
    osy[0].text(7.0, 4.4, "$\\mathbf{x}^\\star$ je vrchol", color=CERVENA, fontsize=10.5, weight="bold",
                ha="left", zorder=6, bbox=stitek)
    osy[0].set_title("Jediné optimum", fontsize=12, color=MODRA)

    for uroven in (20, 30, 40):
        osy[1].plot(t, (uroven - 5 * t) / 1.5, color=MODRA, linewidth=1.3, linestyle=":", zorder=3)
    osy[1].plot([8, 6.5], [0, 5], color=CERVENA, linewidth=5.0, solid_capstyle="round", zorder=5)
    osy[1].add_patch(FancyArrowPatch((1.2, 1.0), (3.6, 1.7), arrowstyle="-|>", mutation_scale=15,
                                     linewidth=2.2, color=ORANZOVA, zorder=4))
    osy[1].text(3.6, 8.7, "všechny body hrany\njsou optimální", color=CERVENA, fontsize=10.5,
                ha="center", zorder=6, bbox=stitek)
    osy[1].set_title("Celá hrana optim", fontsize=12, color=MODRA)

    osy[2].add_patch(Polygon([[1, 1], [11.6, 1], [11.6, 11.6], [7.6, 11.6], [1, 5]], closed=True,
                             facecolor="#cdecee", edgecolor="none", zorder=2))
    osy[2].plot([1, 1], [1, 5], color=TYRKYSOVA, linewidth=2.0, zorder=3)
    osy[2].plot([1, 11.6], [1, 1], color=TYRKYSOVA, linewidth=2.0, zorder=3)
    osy[2].plot([1, 7.6], [5, 11.6], color=TYRKYSOVA, linewidth=2.0, zorder=3)
    for uroven in (4, 8, 12, 16):
        osy[2].plot(t, (uroven - t) / 0.9, color=MODRA, linewidth=1.3, linestyle=":", zorder=3)
    osy[2].add_patch(FancyArrowPatch((3.0, 2.2), (9.6, 8.1), arrowstyle="-|>", mutation_scale=17,
                                     linewidth=2.6, color=ORANZOVA, zorder=4))
    osy[2].text(4.6, 9.6, "oblast pokračuje\ndonekonečna:\n$\\mathbf{c}^\\mathsf{T}\\mathbf{x}\\to\\infty$",
                color="#9a6200", fontsize=10.5, weight="bold", ha="center", va="top",
                zorder=6, bbox=stitek)
    osy[2].set_title("Neomezená úloha", fontsize=12, color=MODRA)

    osy[3].fill_between(t, -0.6, np.minimum(4 - t, 11.6), color=TYRKYSOVA, alpha=0.28, linewidth=0)
    osy[3].plot(t, 4 - t, color=TYRKYSOVA, linewidth=2.0)
    osy[3].fill_between(t, np.maximum(9.5 - t, -0.6), 11.6, color=CERVENA, alpha=0.24, linewidth=0)
    osy[3].plot(t, 9.5 - t, color=CERVENA, linewidth=2.0)
    osy[3].text(0.6, 0.6, "$\\mathbf{a}_1^\\mathsf{T}\\mathbf{x}\\leq b_1$", color="#005a60", fontsize=11,
                zorder=6, bbox=stitek)
    osy[3].text(6.6, 9.6, "$\\mathbf{a}_2^\\mathsf{T}\\mathbf{x}\\geq b_2$", color=CERVENA, fontsize=11,
                zorder=6, bbox=stitek)
    osy[3].text(5.9, 4.6, "průnik je prázdný", color=MODRA, fontsize=11, ha="center",
                weight="bold", rotation=-45, zorder=6, bbox=stitek)
    osy[3].set_title("Prázdná přípustná oblast", fontsize=12, color=MODRA)

    for osa in osy:
        osa.set(xlim=(-0.6, 11.6), ylim=(-0.6, 11.6), aspect="equal",
                xlabel="$x_1$", xticks=[], yticks=[])
        osa.grid(alpha=0.14)
    osy[0].set_ylabel("$x_2$")

    fig.tight_layout()
    fig.savefig(OUT / "ctyri-konce.svg", metadata=SVG_METADATA, bbox_inches="tight")
    plt.close(fig)


def simplex_chuze() -> None:
    """Chůze simplexu po vrcholech pekárny se ziskem v každém vrcholu."""
    fig, osa = plt.subplots(figsize=(7.0, 5.2))
    x1 = np.linspace(-15, 225, 200)

    oblast = Polygon(
        [[0, 0], [160, 0], [100, 150], [0, 300]], closed=True,
        facecolor="#cdecee", edgecolor=TYRKYSOVA, linewidth=2, alpha=0.85, zorder=1,
    )
    osa.add_patch(oblast)
    osa.plot(x1, (80 - 0.5 * x1) / 0.2, color=CERVENA, linewidth=1.5, alpha=0.55, zorder=2)
    osa.plot(x1, (600 - 3 * x1) / 2, color="#3a8a52", linewidth=1.5, alpha=0.55, zorder=2)
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}
    osa.text(139, 50, "mouka", color=CERVENA, fontsize=10, ha="center", va="center",
             bbox=stitek, zorder=6)
    osa.text(193, 18, "pec", color="#3a8a52", fontsize=10, ha="center", va="center",
             bbox=stitek, zorder=6)

    # Cesta simplexu: (0,0) -> (0,300) -> (100,150).
    kroky = (((0, 12), (0, 288), "1", (-13, 150)), ((6, 291), (94, 156), "2", (36, 254)))
    for zacatek, konec, cislo, popisek in kroky:
        osa.add_patch(FancyArrowPatch(
            zacatek, konec, arrowstyle="-|>", mutation_scale=20,
            linewidth=3.2, color=ORANZOVA, zorder=4,
        ))
        osa.text(*popisek, f"krok {cislo}", color="#9a6200", fontsize=11, weight="bold",
                 ha="center", rotation=90 if cislo == "1" else -25,
                 rotation_mode="anchor", bbox=stitek, zorder=6)

    vrcholy = (
        ((0, 0), "start $(0,0)$ — 0 Kč", (-14, -36), MODRA, "left"),
        ((0, 300), "$(0,300)$ — 2400 Kč", (10, 312), MODRA, "left"),
        ((160, 0), "$(160,0)$ — 2240 Kč\n(sem simplex nešel)", (224, -36), "#6b7b85", "right"),
    )
    for bod, popis, misto, barva, zarovnani in vrcholy:
        osa.scatter(*bod, s=70, color=barva, zorder=5)
        osa.text(*misto, popis, color=barva, fontsize=11, weight="bold", zorder=6,
                 ha=zarovnani, va="top", bbox=stitek)

    osa.scatter(100, 150, s=230, marker="*", color=CERVENA, edgecolor="white",
                linewidth=1.0, zorder=6)
    osa.annotate("$\\mathbf{x}^\\star=(100,150)$ — 2600 Kč\nžádná hrana už zisk nezvýší",
                 xy=(104, 152), xytext=(120, 213), fontsize=11, weight="bold",
                 arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA, zorder=6)
    osa.set(xlabel="chleby $x_1$ [ks/den]", ylabel="bagety $x_2$ [ks/den]",
            xlim=(-18, 228), ylim=(-58, 348))
    osa.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "simplex-chuze.svg", metadata=SVG_METADATA)
    plt.close(fig)


def posun_omezeni() -> None:
    """Uvolnění omezení o 10 kg mouky posune vrchol a zvedne zisk o 100 Kč."""
    fig, ax = plt.subplots(figsize=(6.6, 4.5))

    puvodni = np.array([[0.0, 0.0], [160.0, 0.0], [100.0, 150.0], [0.0, 300.0]])
    rozsirena = np.array([[0.0, 0.0], [180.0, 0.0], [150.0, 75.0], [0.0, 300.0]])

    ax.add_patch(Polygon(rozsirena, closed=True, facecolor=ORANZOVA, alpha=0.35,
                         edgecolor="none", zorder=1))
    ax.add_patch(Polygon(puvodni, closed=True, facecolor=SVETLE_MODRA, alpha=0.95,
                         edgecolor=TYRKYSOVA, linewidth=1.4, zorder=2))

    osa = np.linspace(0, 210, 200)
    ax.plot(osa, (80 - 0.5 * osa) / 0.2, linewidth=2.4, color=TYRKYSOVA, zorder=3)
    ax.plot(osa, (90 - 0.5 * osa) / 0.2, linewidth=2.2, linestyle="--", color=ORANZOVA, zorder=3)
    ax.plot(osa, (600 - 3 * osa) / 2, linewidth=2.4, color=MODRA, zorder=3)

    ax.text(150, 44, "mouka 80 kg", color=TYRKYSOVA, fontsize=10.5, weight="bold",
            rotation=-38, ha="center", va="center")
    ax.text(178, 62, "mouka 90 kg", color="#a86a10", fontsize=10.5, weight="bold",
            rotation=-38, ha="center", va="center")
    ax.text(52, 232, "pec 600 min", color=MODRA, fontsize=10.5, weight="bold",
            rotation=-31, ha="center", va="center")

    ax.scatter([100], [150], s=95, color=CERVENA, zorder=6)
    ax.scatter([150], [75], s=95, facecolor="white", edgecolor=CERVENA,
               linewidth=2.2, zorder=6)
    ax.add_patch(FancyArrowPatch((100, 150), (150, 75), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=2, color=CERVENA, zorder=6))

    ax.annotate("$\\mathbf{x}^\\star=(100,150)$\nzisk 2600 Kč", xy=(100, 150), xytext=(6, 96),
                fontsize=10.5, color=CERVENA, weight="bold",
                arrowprops={"arrowstyle": "->", "color": CERVENA})
    ax.annotate("$(150,75)$\nzisk 2700 Kč", xy=(150, 75), xytext=(146, 148),
                fontsize=10.5, color="#a86a10", weight="bold",
                arrowprops={"arrowstyle": "->", "color": "#a86a10"})

    ax.text(103, 288, "+10 kg mouky  →  +100 Kč,  tedy 10 Kč/kg",
            fontsize=11, color=MODRA, weight="bold", ha="center",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": ORANZOVA, "linewidth": 1.4})

    ax.set(xlabel="chléb $x_1$ [ks]", ylabel="bageta $x_2$ [ks]",
           xlim=(0, 205), ylim=(0, 310))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "posun-omezeni.svg", metadata=SVG_METADATA)
    plt.close(fig)


def stinove_ceny() -> None:
    """Optimální zisk jako po částech lineární funkce dostupné mouky."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    zlomy = np.array([0.0, 60.0, 100.0, 130.0])
    zisky = np.array([0.0, 2400.0, 2800.0, 2800.0])

    ax.axvspan(60, 100, color=SVETLE_MODRA, alpha=0.75, zorder=0)
    ax.plot([100, 130], [2800, 3100], linewidth=1.8, linestyle=":", color=CERVENA, zorder=2)
    ax.plot(zlomy, zisky, linewidth=3.2, color=TYRKYSOVA, zorder=3)
    ax.scatter([60, 100], [2400, 2800], s=70, facecolor="white",
               edgecolor=TYRKYSOVA, linewidth=2.2, zorder=5)
    ax.scatter([80], [2600], s=110, color=CERVENA, zorder=6)

    ax.annotate("dnešní stav\n80 kg → 2600 Kč", xy=(80, 2600), xytext=(84, 1750),
                fontsize=10.5, color=CERVENA, weight="bold", ha="center",
                arrowprops={"arrowstyle": "->", "color": CERVENA})
    ax.text(22, 1500, "směrnice 40 Kč/kg\n(pec ještě nevytížená)", fontsize=10,
            color=MODRA, ha="center")
    ax.annotate("směrnice 10 Kč/kg\n= stínová cena mouky", xy=(72, 2520), xytext=(13, 2930),
                fontsize=10.5, color=TYRKYSOVA, weight="bold", ha="left",
                arrowprops={"arrowstyle": "->", "color": TYRKYSOVA})
    ax.text(115, 2620, "směrnice 0 Kč/kg\nmouka už není\núzkým hrdlem", fontsize=10,
            color=MODRA, ha="center", va="top")
    ax.text(102, 2890, "naivní extrapolace", fontsize=10, color=CERVENA,
            rotation=13, rotation_mode="anchor", ha="left", va="bottom")
    ax.text(80, 340, "cena 10 Kč/kg platí\njen na $60\\leq b_1\\leq100$", fontsize=10.5,
            color="#1676a5", ha="center", weight="bold")
    ax.axvline(100, color="#8a97a0", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(101.5, 1150, "zlom: mění se\nmnožina aktivních\nomezení", fontsize=10,
            color="#52616b", ha="left")

    ax.set(xlabel="dostupná mouka $b_1$ [kg]", ylabel="optimální zisk [Kč]",
           xlim=(0, 130), ylim=(0, 3200))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "stinove-ceny.svg", metadata=SVG_METADATA)
    plt.close(fig)


if __name__ == "__main__":
    linearita()
    matice_tabulka()
    soucin_ax()
    formulace_recept()
    radioterapie_matice()
    l1_vs_mnc()
    celociselne_zaokrouhleni()
    polyedr_poloprostory()
    vrstevnice_posun()
    ctyri_konce()
    simplex_chuze()
    posun_omezeni()
    stinove_ceny()
