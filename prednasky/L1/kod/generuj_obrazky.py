"""Vytvoří ilustrace a grafy pro první přednášku.

Většina obrázků je vektorová (SVG); jeden slide, kde sdělení nese pohyb, je
animovaná smyčka (GIF) — viz `uloz_gif`.
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as Cesta
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from PIL import Image


OUT = Path(__file__).resolve().parents[1] / "obrazky"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "mpc-omm-l1"
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


def spotreba_rychlost() -> None:
    """Spotřeba jako součet dvou protichůdných členů."""
    a, b, c = 400.0, 0.05, 1.5
    rychlost = np.linspace(20, 170, 600)
    casove_ztraty = a / rychlost
    odpor = b * rychlost + c
    spotreba = casove_ztraty + odpor
    optimum = np.sqrt(a / b)
    minimum = a / optimum + b * optimum + c

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axvspan(50, 130, color=SVETLE_MODRA, alpha=0.6, zorder=0)
    ax.text(90, 22.6, "přípustné $50\\leq v\\leq130$", color="#1676a5", ha="center", fontsize=10)
    ax.plot(rychlost, casove_ztraty, linewidth=1.8, linestyle="--", color=ORANZOVA,
            label="$A/v$ — ztráty závislé na čase")
    ax.plot(rychlost, odpor, linewidth=1.8, linestyle=":", color="#3a8a52",
            label="$Bv+C$ — odpor prostředí")
    ax.plot(rychlost, spotreba, linewidth=3, color=TYRKYSOVA, label="$f(v)$ — celková spotřeba")
    ax.scatter([optimum], [minimum], s=90, color=CERVENA, zorder=4)
    ax.annotate(
        f"$v^\\star=\\sqrt{{A/B}}\\approx{optimum:.0f}$ km/h",
        xy=(optimum, minimum),
        xytext=(optimum + 14, minimum + 6.5),
        arrowprops={"arrowstyle": "->", "color": CERVENA},
        color=CERVENA,
        weight="bold",
    )
    ax.set(xlabel="Rychlost $v$ [km/h]", ylabel="Spotřeba [l / 100 km]", xlim=(20, 170), ylim=(0, 24))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "spotreba-rychlost.svg", metadata=SVG_METADATA)
    plt.close(fig)


def kohoutek_anatomie() -> None:
    """Schéma míchání vody a krajina účelové funkce se dvěma cíli."""
    fig, osy = plt.subplots(1, 2, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [1.02, 1]})

    schema = osy[0]
    schema.set(xlim=(0, 12), ylim=(0, 6))
    schema.axis("off")
    zdroje = (
        (3.5, "#f6b6a1", CERVENA, "teplá voda\n60 °C", "$q_\\mathrm{t}$", 3.55),
        (1.0, "#a8d8ef", "#1676a5", "studená voda\n10 °C", "$q_\\mathrm{s}$", 2.45),
    )
    for y, pozadi, okraj, popisek, promenna, cil in zdroje:
        schema.add_patch(FancyBboxPatch(
            (0.25, y), 2.5, 1.5, boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=pozadi, edgecolor=okraj, linewidth=1.8,
        ))
        schema.text(1.5, y + 0.75, popisek, ha="center", va="center", fontsize=11.5, color=MODRA)
        schema.add_patch(FancyArrowPatch(
            (2.95, y + 0.75), (4.75, cil), arrowstyle="-|>", mutation_scale=17,
            linewidth=4, color=okraj, alpha=0.75,
        ))
        schema.text(3.85, cil + (0.55 if y > 2 else -0.55), promenna, ha="center",
                    va="center", fontsize=15, weight="bold", color=okraj)

    schema.add_patch(FancyBboxPatch(
        (4.9, 2.05), 2.5, 1.9, boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="white", edgecolor=MODRA, linewidth=2,
    ))
    schema.text(6.15, 3.35, "kohoutek", ha="center", va="center", fontsize=12.5,
                weight="bold", color=MODRA)
    schema.text(6.15, 2.65, "$\\mathbf{x}=(q_\\mathrm{t},q_\\mathrm{s})$", ha="center", va="center",
                fontsize=13, color=TYRKYSOVA)
    schema.add_patch(FancyArrowPatch(
        (7.6, 3.0), (9.0, 3.0), arrowstyle="-|>", mutation_scale=18, linewidth=4,
        color=TYRKYSOVA, alpha=0.8,
    ))
    schema.text(9.3, 3.62, "$T(x)$ — teplota", ha="left", va="center", fontsize=12, color=MODRA)
    schema.text(9.3, 2.38, "$Q(x)$ — průtok", ha="left", va="center", fontsize=12, color=MODRA)
    schema.text(6.15, 0.75, "proměnné → model → dvě sledované veličiny", ha="center",
                fontsize=11, color="#52616b", style="italic")

    krajina = osy[1]
    osa = np.linspace(0, 12, 160)
    tepla, studena = np.meshgrid(osa, osa)
    celkem = tepla + studena
    teplota = (60 * tepla + 10 * studena) / np.maximum(celkem, 1e-9)
    naklad = (teplota - 38.5) ** 2 + (celkem - 12) ** 2

    obraz = krajina.contourf(tepla, studena, naklad, levels=np.geomspace(0.5, 900, 12),
                             cmap="Blues_r", alpha=0.85)
    krajina.contour(tepla, studena, naklad, levels=np.geomspace(0.5, 900, 12),
                    colors=MODRA, linewidths=0.5, alpha=0.5)
    fig.colorbar(obraz, ax=krajina, fraction=0.045, label="$f(x)$")

    podil = (60 - 38.5) / (38.5 - 10)
    krajina.plot([0, 12], [0, 12 / podil], color=CERVENA, linewidth=2, linestyle="--",
                 label="$T(x)=38{,}5\\,°$C")
    krajina.plot([0, 12], [12, 0], color="#3a8a52", linewidth=2, linestyle="-.",
                 label="$Q(x)=12$ l/min")
    optimum = np.array([12 / (1 + 1 / podil), 12 / (1 + podil)])
    krajina.scatter(*optimum, s=150, marker="*", color=ORANZOVA, edgecolor=MODRA,
                    linewidth=1.2, zorder=5, label="optimum $\\mathbf{x}^\\star$")
    krajina.set(
        xlabel="teplá voda $q_\\mathrm{t}$ [l/min]",
        ylabel="studená voda $q_\\mathrm{s}$ [l/min]",
        xlim=(0, 12), ylim=(0, 12), aspect="equal",
    )
    krajina.set_title("účelová funkce nad přípustným boxem", fontsize=11, color=MODRA)
    krajina.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "kohoutek-anatomie.svg", metadata=SVG_METADATA)
    plt.close(fig)


def gradient_krajina() -> None:
    """Gradient jako směr nejrychlejšího růstu a kroky gradientního sestupu."""
    def hodnota(x, y):
        return 0.8 * x**2 + 1.4 * y**2

    def sklon(x, y):
        return np.array([1.6 * x, 2.8 * y])

    x = np.linspace(-2.4, 2.4, 48)
    y = np.linspace(-2.0, 2.0, 48)
    xx, yy = np.meshgrid(x, y)
    zz = hodnota(xx, yy)
    urovne = np.linspace(0.0, zz.max(), 10)

    krok, cesta = 0.25, [np.array([-2.2, 1.7])]
    for _ in range(16):
        cesta.append(cesta[-1] - krok * sklon(*cesta[-1]))
    cesta = np.array(cesta)

    fig = plt.figure(figsize=(9.2, 4.6))
    prostor = fig.add_subplot(1, 2, 1, projection="3d")
    prostor.plot_surface(xx, yy, zz, cmap="Blues", alpha=0.8, linewidth=0.2,
                         edgecolor="white", antialiased=True, rstride=1, cstride=1)
    prostor.contour(xx, yy, zz, levels=urovne, zdir="z", offset=-3.0,
                    colors=MODRA, linewidths=0.8, alpha=0.55)
    prostor.plot(cesta[:, 0], cesta[:, 1], hodnota(cesta[:, 0], cesta[:, 1]) + 0.12,
                 "o-", color=CERVENA, markersize=3.5, linewidth=2, zorder=10)
    prostor.scatter([0], [0], [-3.0], color=ORANZOVA, marker="*", s=150, depthshade=False)
    prostor.text(-2.5, 2.2, hodnota(-2.2, 1.7) + 1.1, "$\\mathbf{x}_0$", color=CERVENA,
                 fontsize=13, weight="bold")
    prostor.set(xlabel="$x_1$", ylabel="$x_2$", zlim=(-3.0, zz.max()))
    prostor.set_zlabel("$f(\\mathbf{x})$")
    prostor.set_zticks([])
    prostor.set_title("krajina účelové funkce", fontsize=11.5, color=MODRA, pad=0)
    prostor.view_init(elev=30, azim=-58)
    prostor.tick_params(labelsize=8)

    rovina = fig.add_subplot(1, 2, 2)
    rovina.contourf(xx, yy, zz, levels=urovne, cmap="Blues_r", alpha=0.45)
    rovina.contour(xx, yy, zz, levels=urovne, colors=MODRA, linewidths=1.0, alpha=0.55)

    for start, popis in (((1.35, 1.15), True), ((-1.7, -1.25), False)):
        p = np.array(start, dtype=float)
        g = sklon(*p)
        g = g / np.linalg.norm(g)
        rovina.add_patch(FancyArrowPatch(p, p + 0.8 * g, arrowstyle="-|>",
                                         mutation_scale=16, linewidth=2.4, color="#555555"))
        rovina.add_patch(FancyArrowPatch(p, p - 0.8 * g, arrowstyle="-|>",
                                         mutation_scale=16, linewidth=2.4, color=CERVENA))
        if popis:
            kolmice = np.array([-g[1], g[0]])
            nahoru = p + 0.45 * g + 0.45 * kolmice
            dolu = p - 0.5 * g + 0.45 * kolmice
            rovina.text(*nahoru, "$\\nabla f$", color="#444444",
                        fontsize=13, weight="bold", ha="center", va="center")
            rovina.text(*dolu, "$-\\nabla f$", color=CERVENA,
                        fontsize=13, weight="bold", ha="center", va="center")

    rovina.plot(cesta[:, 0], cesta[:, 1], "o-", color=ORANZOVA, markersize=5.5,
                linewidth=2, markeredgecolor="white", markeredgewidth=0.6,
                label="$\\mathbf{x}_{k+1}=\\mathbf{x}_k-\\alpha\\nabla f(\\mathbf{x}_k)$", zorder=4)
    rovina.scatter([0], [0], marker="*", s=200, color=CERVENA, edgecolor="white",
                   linewidth=1.0, zorder=5)
    rovina.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-2.4, 2.4), ylim=(-2.0, 2.3), aspect="equal")
    rovina.set_title("gradient je kolmý na vrstevnice", fontsize=11.5, color=MODRA)
    rovina.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "gradient-krajina.svg", metadata=SVG_METADATA)
    plt.close(fig)


def dimenze() -> None:
    """Jedna, dvě a mnoho proměnných — včetně podoby přípustné množiny."""
    fig, osy = plt.subplots(1, 3, figsize=(12.6, 4.1))
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}

    jedna = osy[0]
    x = np.linspace(-2.6, 2.6, 400)
    y = 0.45 * x**2 + 0.35 * np.sin(2.2 * x) + 1.0
    dolni, horni = 0.4, 2.3
    jedna.axvspan(dolni, horni, color="#cdecee", alpha=0.85, zorder=0)
    jedna.plot(x, y, color=TYRKYSOVA, linewidth=3, zorder=2)
    volne = int(np.argmin(y))
    prípustne = (x >= dolni) & (x <= horni)
    vazane = int(np.argmin(np.where(prípustne, y, np.inf)))
    jedna.scatter([x[volne]], [y[volne]], marker="x", s=90, color="#777777", linewidth=2.5, zorder=4)
    jedna.annotate("volné minimum", xy=(x[volne], y[volne]), xytext=(-1.5, 0.32),
                   arrowprops={"arrowstyle": "->", "color": "#888888"},
                   color="#666666", fontsize=9.5, ha="center")
    jedna.scatter([x[vazane]], [y[vazane]], s=120, color=CERVENA, zorder=5)
    jedna.annotate("optimum na hranici", xy=(x[vazane], y[vazane]), xytext=(0.2, 3.55),
                   arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA,
                   weight="bold", fontsize=10, bbox=stitek)
    jedna.text((dolni + horni) / 2, 0.28, "přípustná množina\ninterval $a\\leq x\\leq b$",
               color=TYRKYSOVA, fontsize=10, ha="center", weight="bold")
    jedna.set(xlabel="$x$ ∈ ℝ", ylabel="$f(x)$", ylim=(0, 4.4), xlim=(-2.6, 2.6))
    jedna.set_title("$n=1$ — křivka a interval", fontsize=13, color=MODRA)
    jedna.grid(alpha=0.25)

    dve = osy[1]
    mrizka = np.linspace(-2.6, 2.6, 220)
    xx, yy = np.meshgrid(mrizka, mrizka)
    zz = 0.5 * (xx + 0.4) ** 2 + 1.1 * (yy - 0.3) ** 2 + 0.6 * np.sin(1.6 * xx) * np.cos(1.4 * yy)
    dve.contourf(xx, yy, zz, levels=12, cmap="Blues_r", alpha=0.75)
    dve.contour(xx, yy, zz, levels=12, colors=MODRA, linewidths=0.7, alpha=0.6)

    vrcholy = np.array([[0.15, -2.5], [2.5, -2.5], [2.5, 2.2], [1.35, 2.5], [-0.2, 0.1]])
    dve.add_patch(Polygon(vrcholy, closed=True, facecolor="#cdecee", edgecolor=TYRKYSOVA,
                          linewidth=2.5, alpha=0.55, zorder=3))
    uvnitr = Cesta(vrcholy).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    volne2 = np.unravel_index(np.argmin(zz), zz.shape)
    vazane2 = np.unravel_index(np.argmin(np.where(uvnitr, zz, np.inf)), zz.shape)
    dve.scatter([xx[volne2]], [yy[volne2]], marker="x", s=90, color="#555555",
                linewidth=2.5, zorder=5)
    dve.scatter([xx[vazane2]], [yy[vazane2]], s=170, marker="*", color=CERVENA,
                edgecolor="white", linewidth=1.0, zorder=6)
    dve.text(1.55, -1.35, "přípustná\nmnožina", color="#046b71", fontsize=10.5,
             ha="center", weight="bold", zorder=7)
    dve.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    dve.set_title("$n=2$ — vrstevnice a oblast", fontsize=13, color=MODRA)

    mnoho = osy[2]
    mnoho.set(xlim=(0, 10), ylim=(0, 10))
    mnoho.axis("off")
    mnoho.text(7.6, 8.0, "?", fontsize=80, color="#e6edf1", ha="center", va="center", zorder=0)
    popisky = ("$x_1$", "$x_2$", "$\\vdots$", "$x_n$")
    for i, popisek in enumerate(popisky):
        y0 = 8.6 - i * 1.05
        if popisek == "$\\vdots$":
            mnoho.text(2.0, y0 + 0.3, popisek, ha="center", va="center", fontsize=16, color=MODRA)
            continue
        mnoho.add_patch(FancyBboxPatch(
            (1.2, y0), 1.6, 0.8, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=SVETLE_MODRA, edgecolor=TYRKYSOVA, linewidth=1.6, zorder=2,
        ))
        mnoho.text(2.0, y0 + 0.4, popisek, ha="center", va="center", fontsize=13, color=MODRA, zorder=3)
    mnoho.text(3.7, 8.2, "$\\mathbf{x}$ ∈ ℝ$^n$", fontsize=16, color=MODRA, weight="bold")
    mnoho.text(3.7, 6.6, "$n=10^3$ voxelů,\n$n=10^6$ vah sítě", fontsize=11.5, color="#52616b")

    mnoho.add_patch(FancyBboxPatch(
        (0.7, 2.7), 8.6, 1.9, boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor="#eef7f8", edgecolor=TYRKYSOVA, linewidth=2, zorder=2,
    ))
    mnoho.text(5.0, 4.05, "přípustná množina", ha="center", va="center",
               fontsize=11.5, color="#046b71", weight="bold", zorder=3)
    mnoho.text(5.0, 3.25, "$g_i(\\mathbf{x})\\leq0,\\quad h_j(\\mathbf{x})=0$", ha="center", va="center",
               fontsize=14, color=MODRA, zorder=3)
    mnoho.text(5.0, 1.35, "nakreslit ji nelze — přípustnost\njen ověříme dosazením",
               fontsize=12, color=CERVENA, ha="center", weight="bold")
    mnoho.set_title("$n\\gg3$ — jen nerovnosti", fontsize=13, color=MODRA)

    fig.tight_layout()
    fig.savefig(OUT / "dimenze.svg", metadata=SVG_METADATA)
    plt.close(fig)


def lp_pekarna() -> None:
    """Školní lineární program: dva výrobky a dvě suroviny."""
    fig, osa = plt.subplots(figsize=(6.6, 5.0))
    x1 = np.linspace(0, 210, 200)
    mouka = (80 - 0.5 * x1) / 0.2
    pec = (600 - 3 * x1) / 2

    oblast = Polygon([[0, 0], [160, 0], [100, 150], [0, 300]], closed=True,
                     facecolor="#cdecee", edgecolor=TYRKYSOVA, linewidth=2, alpha=0.85)
    osa.add_patch(oblast)
    osa.plot(x1, mouka, color=CERVENA, linewidth=2.2, label="mouka: $0{,}5x_1+0{,}2x_2=80$")
    osa.plot(x1, pec, color="#3a8a52", linewidth=2.2, label="pec: $3x_1+2x_2=600$")

    for zisk, styl in ((1200, ":"), (2000, ":"), (2600, "--")):
        osa.plot(x1, (zisk - 14 * x1) / 8, color=MODRA, linewidth=1.4, linestyle=styl, alpha=0.75)
    stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}
    osa.text(150, 145, "vrstevnice zisku", color=MODRA, fontsize=10, rotation=-39,
             rotation_mode="anchor", bbox=stitek)
    osa.add_patch(FancyArrowPatch((55, 40), (108, 78), arrowstyle="-|>", mutation_scale=16,
                                  linewidth=2.4, color=ORANZOVA))
    osa.text(52, 20, "roste zisk", color="#9a6200", fontsize=11, weight="bold", bbox=stitek)

    for vrchol in ((0, 0), (160, 0), (0, 300)):
        osa.scatter(*vrchol, s=55, color=MODRA, zorder=4)
    osa.scatter(100, 150, s=170, marker="*", color=CERVENA, edgecolor="white",
                linewidth=1.0, zorder=5)
    osa.annotate("$\\mathbf{x}^\\star=(100,150)$\nzisk 2600 Kč", xy=(100, 150), xytext=(118, 205),
                 arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA, weight="bold")
    osa.text(14, 118, "přípustná\noblast", color=TYRKYSOVA, weight="bold", fontsize=11, bbox=stitek)
    osa.set(xlabel="chleby $x_1$ [ks/den]", ylabel="bagety $x_2$ [ks/den]",
            xlim=(0, 210), ylim=(0, 330))
    osa.grid(alpha=0.2)
    osa.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "lp-pekarna.svg", metadata=SVG_METADATA)
    plt.close(fig)


def michani_vody() -> None:
    """Dvě veličiny, které při míchání vody sledujeme současně."""
    prutok = np.linspace(0, 12, 240)
    tepla, studena = np.meshgrid(prutok, prutok)
    celkem = tepla + studena
    teplota = (60 * tepla + 10 * studena) / np.maximum(celkem, 1e-9)

    fig, osy = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    panely = (
        (teplota, "Teplota $T(x)$ [°C]", "coolwarm", 38.5, CERVENA, "$T(x)=38{,}5$"),
        (celkem, "Celkový průtok $Q(x)$ [l/min]", "YlGnBu", 12.0, "#1b4d3e", "$Q(x)=12$"),
    )
    for osa, (data, titulek, paleta, cil, barva, popisek) in zip(osy, panely):
        obraz = osa.imshow(data, origin="lower", extent=[0, 12, 0, 12], aspect="equal", cmap=paleta)
        osa.contour(tepla, studena, data, colors="white", linewidths=0.6, alpha=0.7)
        cara = osa.contour(tepla, studena, data, levels=[cil], colors=[barva], linewidths=3)
        osa.clabel(cara, fmt={cil: popisek}, fontsize=10, inline=True)
        osa.set_title(titulek, fontsize=11.5, color=MODRA)
        osa.set_xlabel("teplá voda $q_\\mathrm{t}$ [l/min]")
        fig.colorbar(obraz, ax=osa, fraction=0.045)
    osy[0].set_ylabel("studená voda $q_\\mathrm{s}$ [l/min]")
    fig.tight_layout()
    fig.savefig(OUT / "michani-vody.png", dpi=200)
    plt.close(fig)


def konvexni_funkce() -> None:
    x = np.linspace(-2, 4, 400)
    funkce = (x**2, np.sin(2 * x) + 0.3 * x**2)
    fig, osy = plt.subplots(1, 2, figsize=(10, 3.6))
    for osa, y, titulek in zip(osy, funkce, ("Konvexní", "Nekonvexní")):
        osa.plot(x, y, color=TYRKYSOVA, linewidth=2.5)
        osa.set(title=titulek, xlabel="$x$", ylabel="$f(x)$")
        osa.grid(alpha=0.3)
    body_x = np.array([-1.2, 2.6])
    osy[0].plot(body_x, body_x**2, "o", color=CERVENA, zorder=3)
    osy[0].plot(body_x, np.interp(body_x, body_x, body_x**2), "--", color=CERVENA)
    osy[0].text(0.7, 5.2, "sečna leží nad grafem", color=CERVENA, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "konvexni-funkce.svg", metadata=SVG_METADATA)
    plt.close(fig)


def konvexni_mnozina() -> None:
    fig, osy = plt.subplots(1, 2, figsize=(8, 3.6))
    trojuhelnik = np.array([[0.15, 0.15], [0.85, 0.2], [0.62, 0.85]])
    ucko = np.array([
        [0.15, 0.15], [0.15, 0.85], [0.4, 0.85], [0.4, 0.4],
        [0.6, 0.4], [0.6, 0.85], [0.85, 0.85], [0.85, 0.15],
    ])
    for osa, body, titulek, barva in zip(
        osy,
        (trojuhelnik, ucko),
        ("Konvexní množina", "Nekonvexní množina"),
        ("#91c9d1", "#f4bf75"),
    ):
        osa.add_patch(Polygon(body, closed=True, facecolor=barva, edgecolor="#12355b"))
        osa.set(title=titulek, xlim=(0, 1), ylim=(0, 1), aspect="equal")
        osa.axis("off")
    usecka = np.array([[0.3, 0.28], [0.66, 0.66]])
    osy[0].plot(usecka[:, 0], usecka[:, 1], "o-", color=CERVENA, linewidth=2)
    osy[1].plot([0.28, 0.72], [0.72, 0.72], "o--", color=CERVENA, linewidth=2)
    fig.tight_layout()
    fig.savefig(OUT / "konvexni-mnozina.svg", metadata=SVG_METADATA)
    plt.close(fig)


# --- společná fyzika pro obrázky radioterapie -----------------------------
# Táž geometrie i táž matice dávek jako v živé ukázce na prvním počítačovém
# cvičení (`cviceni/C1/kod/ukazka_radioterapie.py`), aby přednáška a cvičení
# ukazovaly doslova stejného pacienta. Řez má 64 × 64 voxelů, ozařuje se
# z osmi úhlů po dvanácti příčných polohách, takže svazků je 96.

PREDPIS = 60.0        # předepsaná dávka do nádoru [Gy]
TOLERANCE = 1.15      # horní mez v nádoru = 1,15 × předpis
VAHA_MICHA = 3.0      # o kolik víc vadí dávka v míše než ve zdravé tkáni
POLOH = 12            # příčných poloh svazku v jednom úhlu
SIGMA = 2.6           # šířka svazku (odchylka gaussovského profilu) [voxel]
ROZSAH = 18.0         # krajní polohy svazku, ±ROZSAH kolem osy [voxel]
UHLY = [0, 45, 90, 135, 180, 225, 270, 315]
MRIZKA = 64


def _rez():
    """Masky řezu: tělo, nádor, mícha a zdravá tkáň."""
    yy, xx = np.mgrid[0:MRIZKA, 0:MRIZKA]
    stred = (MRIZKA - 1) / 2
    telo = np.hypot(xx - stred, yy - stred) < 30
    nador = np.hypot(xx - stred - 6, yy - stred + 4) < 7
    micha = np.hypot(xx - stred + 8, yy - stred - 6) < 4
    return xx, yy, stred, telo, nador, micha, telo & ~nador & ~micha


def _matice_davek():
    """Matice dávek D: sloupec = mapa dávky od svazku s jednotkovou intenzitou."""
    xx, yy, stred, telo, _, _, _ = _rez()
    sloupce, uhel_svazku, poloha_svazku = [], [], []
    for uhel in np.deg2rad(UHLY):
        pricne = (xx - stred) * np.cos(uhel) + (yy - stred) * np.sin(uhel)
        hloubka = -(xx - stred) * np.sin(uhel) + (yy - stred) * np.cos(uhel)
        for posun in np.linspace(-ROZSAH, ROZSAH, POLOH):
            profil = np.exp(-0.5 * ((pricne - posun) / SIGMA) ** 2)   # šířka svazku
            utlum = np.exp(-0.02 * (hloubka + 32))                    # útlum s hloubkou
            sloupce.append((profil * utlum * telo).ravel())
            uhel_svazku.append(int(round(np.degrees(uhel))))
            poloha_svazku.append(posun)
    return np.array(sloupce).T, np.array(uhel_svazku), np.array(poloha_svazku)


def radioterapie_plany() -> None:
    """Tři skutečně spočítané plány léčby pro úvodní otázku „co je nejlepší“.

    Mapy dávky vznikají ze stejné matice jako na cvičení; plán B je optimum
    konvexního kvadratického programu, A a C jsou plány, které nikdo
    neoptimalizoval. Všechny tři sdílejí barevnou škálu, jinak by se
    neporovnaly.
    """
    import cvxpy as cp

    _, _, _, _, nador, micha, zdrava = _rez()
    D, uhel_svazku, poloha_svazku = _matice_davek()

    # Plán B: optimum — nádor v předepsaném rozmezí, mícha a zdravá tkáň co nejníž.
    intenzity = cp.Variable(D.shape[1], nonneg=True)
    d_nador, d_micha, d_zdrava = D[nador.ravel()], D[micha.ravel()], D[zdrava.ravel()]
    uloha = cp.Problem(
        cp.Minimize(VAHA_MICHA * cp.sum_squares(d_micha @ intenzity)
                    + cp.sum_squares(d_zdrava @ intenzity)),
        [d_nador @ intenzity >= PREDPIS, d_nador @ intenzity <= TOLERANCE * PREDPIS],
    )
    uloha.solve(solver=cp.CLARABEL)
    assert uloha.status == cp.OPTIMAL, f"solver nenašel optimum: {uloha.status}"
    plan_b = (D @ intenzity.value).reshape(MRIZKA, MRIZKA)

    def mapa(vahy: np.ndarray) -> np.ndarray:
        return (D @ vahy).reshape(MRIZKA, MRIZKA)

    # Plán A: opatrný — z každého úhlu jen jediný svazek mířený do středu
    # nádoru. Naškálovaný na tutéž špičku jako optimum, takže je vidět, že
    # mimo střed nádor vychladne; míše se ale vyhne.
    stred_nadoru = np.array([6.0, -4.0])
    opatrny = np.zeros(D.shape[1])
    for uhel in UHLY:
        pricne_nador = (stred_nadoru[0] * np.cos(np.deg2rad(uhel))
                        + stred_nadoru[1] * np.sin(np.deg2rad(uhel)))
        v_uhlu = np.where(uhel_svazku == uhel)[0]
        nejblizsi = np.argmin(np.abs(poloha_svazku[v_uhlu] - pricne_nador))
        opatrny[v_uhlu[nejblizsi]] = 1.0
    plan_a = mapa(opatrny)
    plan_a *= TOLERANCE * PREDPIS / plan_a[nador].max()

    # Plán C: agresivní — všechny svazky stejně, naškálované tak, aby
    # předpis dostal i nejchladnější voxel nádoru. Zaplatí se to míchou.
    plan_c = mapa(np.ones(D.shape[1]))
    plan_c *= PREDPIS / plan_c[nador].min()

    plany = ((plan_a, "Plán A"), (plan_b, "Plán B"), (plan_c, "Plán C"))
    vmax = max(plan.max() for plan, _ in plany)   # společná škála, jinak se plány neporovnají

    fig, osy = plt.subplots(1, 3, figsize=(12, 4.5))
    for osa, (plan, nazev) in zip(osy, plany):
        obraz = osa.imshow(plan, cmap="inferno", vmin=0, vmax=vmax)
        osa.contour(nador, colors="w", linewidths=1.8)
        osa.contour(micha, colors="#00e5ff", linewidths=1.6, linestyles="dashed")
        osa.set_title(nazev, fontsize=13, color=MODRA)
        osa.axis("off")
    osy[0].annotate("nádor", xy=(37.5, 21), xytext=(37.5, 5), color="w", fontsize=11,
                    weight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="w", linewidth=1.1))
    osy[0].annotate("mícha", xy=(23.5, 42), xytext=(13, 56), color="#00e5ff", fontsize=11,
                    weight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="#00e5ff", linewidth=1.1))
    fig.subplots_adjust(left=0.01, right=0.88, wspace=0.04)
    pruh = fig.colorbar(obraz, ax=osy.tolist(), fraction=0.03, pad=0.02)
    pruh.set_label("dávka záření [Gy], společná škála", fontsize=10.5)
    fig.savefig(OUT / "radioterapie-plany.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def radioterapie_model() -> None:
    """Schéma proměnných, cíle a omezení při plánování radioterapie.

    Vlevo tentýž řez jako na předchozím slidu, jen bez dávky. Kolem těla je
    osm směrů, ze kterých ozařovač střílí, a jeden z nich je rozbalený na
    dvanáct příčných poloh — každý pás je jedna nastavitelná intenzita.
    Vpravo tři rámečky s anatomií optimalizační úlohy.
    """
    fig = plt.figure(figsize=(11.6, 5.2))
    rez = fig.add_axes((0.005, 0.02, 0.44, 0.96))
    rez.set(xlim=(-58, 48), ylim=(-44, 44), aspect="equal")
    rez.axis("off")

    telo = Circle((0, 0), 30, facecolor=SEDIVA, edgecolor=MODRA, linewidth=2.2, zorder=2)
    rez.add_patch(telo)

    # Osm směrů ozáření: šipky po kružnici kolem těla.
    for uhel in UHLY:
        smer = np.array([np.sin(np.deg2rad(uhel)), -np.cos(np.deg2rad(uhel))])
        rez.add_patch(FancyArrowPatch(
            -39 * smer, -33 * smer, arrowstyle="-|>", mutation_scale=13,
            linewidth=2.0, color="#1676a5", alpha=0.9, zorder=6,
        ))
    rez.add_patch(FancyArrowPatch(
        (10, 34), (36, 10), arrowstyle="<|-|>", mutation_scale=11,
        linewidth=1.4, color="#1676a5", zorder=6,
        connectionstyle="arc3,rad=-0.3",
    ))
    rez.text(36, 32, "8 úhlů kolem\npacienta", ha="center", va="center",
             fontsize=11, color="#1676a5", weight="bold")

    # Jeden směr rozbalený na dvanáct příčných poloh; pás je široký jako svazek.
    uhel = np.deg2rad(45)
    smer = np.array([np.sin(uhel), -np.cos(uhel)])
    kolmice = np.array([smer[1], -smer[0]])
    for posun in np.linspace(-ROZSAH, ROZSAH, POLOH):
        pas, = rez.plot(*np.column_stack([-32 * smer + posun * kolmice,
                                          32 * smer + posun * kolmice]),
                        color="#4ea8de", alpha=0.5, linewidth=2.1 * SIGMA,
                        solid_capstyle="butt", zorder=3)
        pas.set_clip_path(telo)
    okraj = -31 * smer
    rez.add_patch(FancyArrowPatch(
        okraj - ROZSAH * kolmice, okraj + ROZSAH * kolmice, arrowstyle="<|-|>",
        mutation_scale=11, linewidth=1.4, color=MODRA, zorder=6,
    ))
    rez.text(-43, 22, "12 poloh\nv jednom úhlu", ha="center", va="center",
             fontsize=11, color=MODRA, weight="bold")

    rez.add_patch(Circle((6, 4), 7, facecolor="#e76f51", edgecolor=CERVENA,
                         linewidth=1.8, zorder=4))
    rez.add_patch(Circle((-8, -6), 4, facecolor="#ffd166", edgecolor="#c98200",
                         linewidth=1.8, zorder=4))
    rez.text(6, 4, "nádor", color="white", ha="center", va="center", weight="bold",
             fontsize=11, zorder=5)
    rez.text(-8, -11.5, "mícha", color="#a36700", ha="center", va="top", weight="bold",
             fontsize=10.5, zorder=5)
    rez.annotate("$x_j$ = intenzita\njednoho svazku", xy=tuple(16 * smer + 5.7 * kolmice),
                 xytext=(32, -34), fontsize=11, color=MODRA, ha="center", va="center",
                 zorder=7, arrowprops=dict(arrowstyle="-", color=MODRA, linewidth=1.2,
                                           shrinkB=0))

    popis = fig.add_axes((0.46, 0.03, 0.53, 0.94))
    popis.set(xlim=(0, 5.4), ylim=(0, 6))
    popis.axis("off")
    polozky = [
        (4.75, "PROMĚNNÉ", "intenzity svazků $x_1,\u2026,x_n$", TYRKYSOVA),
        (3.00, "CÍL", "co nejmíň dávky do míchy a zdravé tkáně", CERVENA),
        (1.25, "OMEZENÍ", "nádor dostane předepsanou dávku", ORANZOVA),
    ]
    for y, nadpis, text, barva in polozky:
        popis.add_patch(FancyBboxPatch(
            (0.15, y - 0.58), 5.05, 1.16,
            boxstyle="round,pad=0.10,rounding_size=0.13",
            facecolor="white", edgecolor=barva, linewidth=2,
        ))
        popis.text(0.42, y + 0.14, nadpis, color=barva, fontsize=11.5, weight="bold")
        popis.text(0.42, y - 0.28, text, color=MODRA, fontsize=11.5)
    fig.savefig(OUT / "radioterapie-model.svg", metadata=SVG_METADATA)
    plt.close(fig)


def aktivni_omezeni() -> None:
    """Geometrie dvou proměnných s optimem na aktivním omezení."""
    x = np.linspace(0, 6.5, 220)
    y = np.linspace(0, 6.5, 220)
    xx, yy = np.meshgrid(x, y)
    cil = (xx - 5.0) ** 2 + (yy - 4.0) ** 2
    optimum = np.array([3.5, 2.5])

    bily_stitek = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.6}

    fig, osa = plt.subplots(figsize=(6.8, 5.2))
    oblast = Polygon([[0, 0], [6, 0], [0, 6]], closed=True, facecolor="#cdecee", edgecolor=TYRKYSOVA, alpha=0.8)
    osa.add_patch(oblast)
    osa.contour(xx, yy, cil, levels=[1, 3, 6, 10, 16, 24], colors=MODRA, alpha=0.55)
    osa.plot([0, 6], [6, 0], color=CERVENA, linewidth=3)
    osa.scatter([5], [4], marker="x", s=110, color="#777777", linewidth=2.5, zorder=4)
    osa.scatter(*optimum, s=100, color=CERVENA, edgecolor="white", linewidth=1.2, zorder=5)
    osa.annotate(
        "optimum $\\mathbf{x}^\\star$", xy=optimum, xytext=(0.3, 3.35),
        arrowprops={"arrowstyle": "->", "color": CERVENA},
        color=CERVENA, weight="bold", bbox=bily_stitek,
    )
    osa.annotate(
        "aktivní omezení\n$x_1+x_2=6$", xy=(5.05, 0.95), xytext=(5.35, 2.35),
        arrowprops={"arrowstyle": "->", "color": CERVENA},
        color=CERVENA, ha="center", fontsize=10.5, bbox=bily_stitek,
    )
    osa.text(0.35, 0.3, "přípustná oblast", color=TYRKYSOVA, weight="bold", bbox=bily_stitek)
    osa.annotate(
        "minimum bez omezení", xy=(5, 4), xytext=(4.55, 5.85),
        arrowprops={"arrowstyle": "->", "color": "#777777"},
        color="#666666", fontsize=10, ha="center", bbox=bily_stitek,
    )
    osa.text(2.05, 1.55, "vrstevnice $f$", color=MODRA, fontsize=10, alpha=0.85,
             rotation=-38, bbox=bily_stitek)
    osa.set(
        xlabel="$x_1$",
        ylabel="$x_2$",
        xlim=(-0.15, 7.0),
        ylim=(-0.15, 7.0),
        aspect="equal",
    )
    osa.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT / "aktivni-omezeni.svg", metadata=SVG_METADATA)
    plt.close(fig)


def lokalni_globalni() -> None:
    """Animace: stejná metoda ze dvou startů skončí v různých minimech."""
    x = np.linspace(-3.2, 3.2, 900)
    y = 0.16 * (x**2 - 4) ** 2 + 0.32 * x
    kandidati = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    globalni, lokalni = sorted(kandidati, key=lambda i: y[i])[:2]

    def derivace(bod: float) -> float:
        return 0.64 * bod**3 - 2.56 * bod + 0.32

    def krajina(bod):
        return 0.16 * (np.asarray(bod) ** 2 - 4) ** 2 + 0.32 * np.asarray(bod)

    # Dvě trajektorie gradientního sestupu, každá z jiného startu. Krok je
    # menší než by stačilo, aby byl v animaci vidět sestup, ne skok.
    cesty = {}
    for start in (-3.0, 3.0):
        cesta = [start]
        for _ in range(60):
            cesta.append(cesta[-1] - 0.03 * derivace(cesta[-1]))
        cesty[start] = np.array(cesta)

    def snimek(krok: int, finale: bool):
        fig, osa = plt.subplots(figsize=(9.0, 4.6))
        osa.plot(x, y, color=TYRKYSOVA, linewidth=3, zorder=2)

        if finale:
            osa.annotate("globální minimum", xy=(x[globalni], y[globalni]),
                         xytext=(-2.45, 3.05), arrowprops={"arrowstyle": "->", "color": CERVENA},
                         color=CERVENA, weight="bold")
            osa.annotate("lokální minimum", xy=(x[lokalni], y[lokalni]), xytext=(0.85, 3.05),
                         arrowprops={"arrowstyle": "->", "color": ORANZOVA}, color="#9a6200",
                         weight="bold")

        for start, popisek, barva in ((-3.0, "start 1", CERVENA), (3.0, "start 2", ORANZOVA)):
            cesta = cesty[start][: krok + 1]
            osa.plot(cesta, krajina(cesta), "o", color=MODRA, markersize=4.5, alpha=0.5,
                     zorder=3)
            osa.scatter([start], [krajina(start)], marker="o", facecolor="white",
                        edgecolor=MODRA, s=95, linewidth=2, zorder=6)
            osa.text(start, krajina(start) + 0.52, popisek, color=MODRA, ha="center",
                     weight="bold")
            # aktuální poloha kuličky; v cíli dostane barvu svého minima
            osa.scatter([cesta[-1]], [krajina(cesta[-1])], s=180,
                        color=barva if finale else MODRA,
                        edgecolor="white", linewidth=1.6, zorder=7)

        osa.text(0.0, 5.15, "stejná metoda, jiný počáteční bod → jiný výsledek",
                 color="#52616b", ha="center", fontsize=11, style="italic")
        osa.set(xlabel="$x$", ylabel="$f(x)$", xlim=(-3.4, 3.4), ylim=(-1.5, 5.7))
        osa.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    snimky = [snimek(60, True)]                      # statická záloha: výsledek
    doby = [1800]
    kroky = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 60)
    for krok in kroky:
        snimky.append(snimek(krok, finale=(krok == kroky[-1])))
        doby.append(3000 if krok == kroky[-1] else 220)
    uloz_gif(snimky, doby, "lokalni-globalni", dpi=120)


def typy_reseni() -> None:
    """Čtyři základní možnosti existence a jednoznačnosti řešení."""
    fig, osy = plt.subplots(1, 4, figsize=(13, 3.4))
    x = np.linspace(-2.2, 2.2, 300)

    osy[0].plot(x, x**2, color=TYRKYSOVA, linewidth=2.5)
    osy[0].scatter([0], [0], color=CERVENA, zorder=3)
    osy[0].set_title("Jediné optimum")

    y = np.maximum(np.abs(x) - 0.75, 0) ** 2
    osy[1].plot(x, y, color=TYRKYSOVA, linewidth=2.5)
    osy[1].plot([-0.75, 0.75], [0, 0], color=CERVENA, linewidth=5)
    osy[1].set_title("Více optim")

    osy[2].axvspan(-2.2, -1.0, color="#b7e4c7", alpha=0.75)
    osy[2].axvspan(1.0, 2.2, color="#f4bf75", alpha=0.75)
    osy[2].axvline(-1.0, color=TYRKYSOVA, linewidth=2)
    osy[2].axvline(1.0, color=ORANZOVA, linewidth=2)
    osy[2].text(0, 0.58, "$x\\leq-1$\n a zároveň\n$x\\geq1$", ha="center", va="center", color=MODRA)
    osy[2].set_title("Žádné přípustné řešení")

    osy[3].plot(x, -x, color=TYRKYSOVA, linewidth=2.5)
    osy[3].annotate("$f(x)\\to-\\infty$", xy=(2.05, -2.05), xytext=(0.1, -0.75), arrowprops={"arrowstyle": "->", "color": CERVENA}, color=CERVENA)
    osy[3].set_title("Neomezená úloha")

    for i, osa in enumerate(osy):
        osa.axhline(0, color="#777777", linewidth=0.7)
        osa.set_xlabel("$x$")
        osa.grid(alpha=0.16)
        if i != 2:
            osa.set_ylim((-2.3, 4.9))
        else:
            osa.set_ylim((0, 1))
            osa.set_yticks([])
    osy[0].set_ylabel("$f(x)$")
    fig.tight_layout()
    fig.savefig(OUT / "typy-reseni.svg", metadata=SVG_METADATA)
    plt.close(fig)


def mapa_predmetu() -> None:
    """Deset výukových termínů podle aktuálního DOCX 2026/27."""
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.set(xlim=(0, 12), ylim=(0, 4.3))
    ax.axis("off")
    blocks = [
        (0.1, 2.3, "FORMULACE A MATEMATIKA", "23. 9.  Úvod a konvexita\n30. 9.  Lineární programování\n7. 10.  Nelineární programování, KKT", TYRKYSOVA),
        (6.2, 2.3, "OD GRADIENTU K HEURISTIKÁM", "14. 10.  Gradientní a Newtonovy metody\n21. 10.  Nelder–Mead, SA, tabu, CRS, ES", MODRA),
        (0.1, 0.1, "GENETICKÉ A ROJOVÉ ALGORITMY", "4. 11.  Genetické algoritmy\n11. 11.  Obchodní cestující (TSP)\n18. a 25. 11.  ACO, PSO a rojové algoritmy", "#a06714"),
        (6.2, 0.1, "CHYTRÝ VÝBĚR DALŠÍHO POKUSU", "2. 12.  Bayesovská optimalizace\nGaussovský proces a akviziční funkce\nNávrat k drahé černé skříňce", "#3a8a52"),
    ]
    for x,y,title,body,color in blocks:
        ax.add_patch(FancyBboxPatch((x,y),5.6,1.75,boxstyle="round,pad=0.1",facecolor="#f3f6f8",edgecolor=color,lw=2))
        ax.text(x+.18,y+1.42,title,color=color,weight="bold",fontsize=11)
        ax.text(x+.18,y+1.12,body,color=MODRA,va="top",fontsize=12,linespacing=1.55)
    fig.tight_layout()
    fig.savefig(OUT / "mapa-predmetu.svg", metadata=SVG_METADATA)
    plt.close(fig)


def gin_tonic() -> None:
    """Nakreslí schéma přípustné směsi jako vektorový obrázek."""
    fig, osa = plt.subplots(figsize=(6, 4.5))
    osa.set(xlim=(0, 10), ylim=(0, 8.5), aspect="equal")
    osa.axis("off")

    slozky = [
        (0.8, 5.9, "#d9eef7", "gin"),
        (0.8, 3.3, "#b7e4c7", "tonik"),
        (0.8, 0.7, "#ffe066", "citron"),
    ]
    for x, y, barva, popisek in slozky:
        lahev = FancyBboxPatch(
            (x, y), 1.85, 1.25, boxstyle="round,pad=0.12,rounding_size=0.2",
            facecolor=barva, edgecolor="#12355b", linewidth=1.5,
        )
        osa.add_patch(lahev)
        osa.text(x + 0.92, y + 0.63, popisek, ha="center", va="center", fontsize=13, weight="bold")
        osa.add_patch(FancyArrowPatch(
            (2.8, y + 0.63), (5.15, 4.2), arrowstyle="->",
            mutation_scale=14, linewidth=1.4, color="#12355b",
        ))

    sklenice = Polygon(
        [[5.35, 6.9], [8.6, 6.9], [8.05, 2.1], [5.9, 2.1]],
        closed=True, facecolor="#d9f2f4", edgecolor="#007f86", linewidth=2.2,
    )
    osa.add_patch(sklenice)
    kapalina = Polygon(
        [[5.63, 4.9], [8.32, 4.9], [8.05, 2.32], [5.9, 2.32]],
        closed=True, facecolor="#c5e7c0", edgecolor="none", alpha=0.9,
    )
    osa.add_patch(kapalina)
    for x, y, velikost in ((6.25, 4.1, 0.12), (7.65, 3.85, 0.1), (7.1, 3.15, 0.08)):
        osa.add_patch(plt.Circle((x, y), velikost, fill=False, linewidth=1.2, edgecolor="#007f86"))
    osa.add_patch(plt.Circle((8.2, 6.55), 0.55, facecolor="#ffe066", edgecolor="#d89b00", linewidth=1.4))
    osa.plot([8.2, 8.2], [6.15, 6.95], color="#f7f2d0", linewidth=2)
    osa.text(6.98, 5.65, "Gin & Tonic", ha="center", fontsize=15, weight="bold", color="#12355b")
    osa.text(6.98, 4.55, "$T(\\mathbf{x})$", ha="center", fontsize=18, color="#12355b")
    osa.add_patch(FancyArrowPatch(
        (6.98, 1.95), (6.98, 0.4), arrowstyle="->",
        mutation_scale=17, linewidth=2, color="#007f86",
    ))
    osa.text(6.98, 0.08, "hotový drink", ha="center", fontsize=12, color="#12355b")
    fig.tight_layout()
    fig.savefig(OUT / "gin-tonic.svg", metadata=SVG_METADATA)
    plt.close(fig)


if __name__ == "__main__":
    spotreba_rychlost()
    kohoutek_anatomie()
    gradient_krajina()
    dimenze()
    lp_pekarna()
    michani_vody()
    konvexni_funkce()
    konvexni_mnozina()
    gin_tonic()
    radioterapie_plany()
    radioterapie_model()
    aktivni_omezeni()
    lokalni_globalni()
    typy_reseni()
    mapa_predmetu()

    from revize_priklady import main as revize_priklady
    from revize_matematika import main as revize_matematika
    revize_priklady()
    revize_matematika()
