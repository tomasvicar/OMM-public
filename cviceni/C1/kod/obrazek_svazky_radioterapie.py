"""Vysvetlujici obrazek k radioterapii: co presne jsou ta ladena cisla.

Report kresli vsechny svazky najednou, coz je hezke, ale nepozna se z toho,
co je "jedna poloha". Tenhle obrazek to rozebere na tri patra:

  nahore ......... JEDEN uhel (45 stupnu) dvakrat. Vlevo v rezu: vsechny polohy
                   jako pasy siroke +-sigma, ocislovane po rade, sede vypnute a
                   oranzove zapnute s tloustkou podle intenzity. Vpravo tyz uhel
                   jako pricny profil: gaussovky naskalovane intenzitou a
                   jejich soucet, s vyznacenym pricnym prumetem nadoru a michy.
  dole ........... vsech osm uhlu vedle sebe, tataz konvence, a pod nimi vsechna
                   cisla jako sloupce. Skupiny jsou oddelene mezerou, podbarvenim
                   a smerovou sipkou, aby bylo videt, ktery blok patri kteremu
                   smeru svazku.

Data se neresi znovu, ctou se z `data/radioterapie.json` (vyrabi ho
`predpocet_radioterapie.py`), takze obrazek vznikne za zlomek sekundy.

Spusteni z korene repozitare:
    uv run python cviceni/C1/kod/obrazek_svazky_radioterapie.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

UHEL_DETAIL = 45      # ktery uhel se rozebira nahore
ORANZ = "#e8590c"
NADOR = "#c1121f"
MICHA = "#0077cc"
SEDA = "#c4cbd2"
POPISEK = "#5c5c66"

KOREN = Path(__file__).resolve().parents[1]
D = json.loads((KOREN / "data" / "radioterapie.json").read_text(encoding="utf-8"))
G, IT = D["geometrie"], D["intenzity"]
UHLY, POLOH = G["uhly"], G["poloh"]
SIGMA, ROZSAH = G["sigma"], G["rozsah"]   # sirka svazku a krajni poloha, jako v predpoctu
W = np.array(IT["opt"])
PRAH = 0.005 * W.max()            # stejny prah "prakticky vypnuty" jako v reportu
CX, CY, R_TELO, R_HLAVICE = G["cx"], G["cy"], G["r_telo"], G["r_hlavice"]
POSUNY = np.linspace(-ROZSAH, ROZSAH, POLOH)
ROZTEC = POSUNY[1] - POSUNY[0]


def rez(osa, popisky=False, dosah=48):
    """Spolecny podklad: telo, nador, micha, ctvercovy vyrez a otocena osa y."""
    osa.add_patch(plt.Circle((CX, CY), R_TELO, facecolor="#fbfbfc",
                             edgecolor="#adb5bd", linewidth=1.4, zorder=1))
    for klic, barva, popis, strana, styl in (("nador", NADOR, "NÁDOR", 1, "solid"),
                                            ("micha", MICHA, "mícha", -1, (0, (5, 2.5)))):
        s = G[klic]
        # bily podklad a bila obruba, jinak se kruznice na sytě oranžovém poli ztratí
        osa.add_patch(plt.Circle((s["x"], s["y"]), s["r"], facecolor="#fff", alpha=0.55, zorder=4))
        osa.add_patch(plt.Circle((s["x"], s["y"]), s["r"], facecolor=barva, alpha=0.22,
                                 edgecolor="#fff", linewidth=3.4, zorder=5))
        osa.add_patch(plt.Circle((s["x"], s["y"]), s["r"], facecolor="none", edgecolor=barva,
                                 linewidth=2.2, linestyle=styl, zorder=6))
        if popisky:
            osa.text(s["x"] + strana * (s["r"] + 2.5), s["y"], popis, color=barva,
                     fontsize=12, fontweight="bold", zorder=7,
                     ha="left" if strana > 0 else "right", va="center")
    osa.set_xlim(CX - dosah, CX + dosah)
    osa.set_ylim(CY + dosah, CY - dosah)     # y dolu, at to sedi s obrazkem v reportu
    osa.set_aspect("equal")
    osa.axis("off")


def pas(osa, b, uhel, barva, alfa, zorder=2):
    """Svazek jako pas skutecne sirky (+-sigma) od hlavice po vystup z tela.

    Intenzita se nekresli tloustkou, ale vyrazností: sirka pasu je fyzika (sirka
    svazku), pruhlednost je nastaveni. Prekryvajici se pasy tim ztmavnou presne
    tam, kde se davka scita.
    """
    th = np.deg2rad(uhel)
    kolmo = np.array([np.cos(th), np.sin(th)])
    p_, q_ = np.array([b["x1"], b["y1"]]), np.array([b["x2"], b["y2"]])   # jen uvnitr tela
    rohy = [p_ + SIGMA * kolmo, q_ + SIGMA * kolmo, q_ - SIGMA * kolmo, p_ - SIGMA * kolmo]
    osa.add_patch(plt.Polygon(rohy, closed=True, facecolor=barva, alpha=alfa,
                              edgecolor="none", zorder=zorder))


def vyraznost(w):
    """Intenzita -> pruhlednost pasu; vypnuty svazek zustane jen matnym stinem."""
    return 0.12 + 0.72 * w / W.max() if w > PRAH else 0.09


def svazky_uhlu(u):
    """Indexy, geometrie a intenzity vsech poloh jednoho uhlu."""
    return [(j, G["svazky"][j], W[j]) for j in range(u * POLOH, (u + 1) * POLOH)]


fig = plt.figure(figsize=(15.5, 15.5))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.30, 0.72, 0.90],
              hspace=0.30, wspace=0.16)

# ---------------------------------------------------------------- detail jednoho uhlu
u_det = UHLY.index(UHEL_DETAIL)
ax = fig.add_subplot(gs[0, 0])
rez(ax, popisky=True)
th = np.deg2rad(UHEL_DETAIL)
kolmo = np.array([np.cos(th), np.sin(th)])       # napric svazkem
smer = np.array([-np.sin(th), np.cos(th)])       # podel svazku, od hlavice dovnitr

for j, b, w in svazky_uhlu(u_det):
    zap = w > PRAH
    pas(ax, b, UHEL_DETAIL, ORANZ if zap else SEDA, vyraznost(w))
    poradi = j - u_det * POLOH + 1
    # cislo polohy vne tela u hlavice; sude a liche v jine vzdalenosti, at se neprekryvaji
    h = np.array([b["hx"], b["hy"]]) - 4.5 * smer
    ax.text(h[0], h[1], str(poradi), fontsize=9.5, ha="center", va="center",
            color="#1c1c1e" if zap else "#8b939c",
            fontweight="bold" if zap else "normal", zorder=6)

a, z = G["svazky"][u_det * POLOH], G["svazky"][(u_det + 1) * POLOH - 1]
ax.plot([a["hx"], z["hx"]], [a["hy"], z["hy"]], color="#8b939c", linewidth=3.4,
        solid_capstyle="round", zorder=2)
zapnuto = sum(1 for _, _, w in svazky_uhlu(u_det) if w > PRAH)
ax.set_title(f"Jeden úhel ({UHEL_DETAIL}°) zblízka: {POLOH} poloh, z toho {zapnuto} zapnutých\n"
             f"šířka pásu = šířka svazku (±σ = {SIGMA} voxelu), sytost = intenzita",
             fontsize=14, pad=12)

# ---------------------------------------------------------------- pricny profil tehoz uhlu
ax = fig.add_subplot(gs[0, 1])
osa = np.linspace(-28, 28, 800)
soucet = np.zeros_like(osa)
for j, b, w in svazky_uhlu(u_det):
    g = w * np.exp(-0.5 * ((osa - POSUNY[j - u_det * POLOH]) / SIGMA) ** 2)
    soucet += g
    if w > PRAH:
        ax.plot(osa, g, color=ORANZ, linewidth=1.0, alpha=0.55)
ax.plot(osa, soucet, color="#1c1c1e", linewidth=2.4, label="součet = profil pole")
ax.plot([], [], color=ORANZ, linewidth=1.0, alpha=0.8, label=f"jednotlivé polohy ({POLOH} ks)")

vrch = soucet.max()
for klic, barva, popis, strana in (("nador", NADOR, "nádor", 1), ("micha", MICHA, "mícha", -1)):
    s = G[klic]
    stred = (s["x"] - CX) * kolmo[0] + (s["y"] - CY) * kolmo[1]
    ax.axvspan(stred - s["r"], stred + s["r"], color=barva, alpha=0.13, zorder=0)
    ax.text(stred + strana * s["r"], 1.12 * vrch, popis + " ↔", color=barva, fontsize=12,
            fontweight="bold", ha="left" if strana > 0 else "right", va="center")

for j in range(POLOH):
    zap = W[u_det * POLOH + j] > PRAH
    ax.plot([POSUNY[j]], [0], marker="|", markersize=9,
            color=ORANZ if zap else SEDA, markeredgewidth=2.0)
    ax.text(POSUNY[j], -0.05 * vrch, str(j + 1), fontsize=9.5, ha="center", va="top",
            color="#1c1c1e" if zap else "#8b939c",
            fontweight="bold" if zap else "normal")

ax.set_xlabel("příčná poloha napříč svazkem [voxel]", fontsize=13, labelpad=10)
ax.set_ylabel("intenzita svazku", fontsize=13)
ax.set_xlim(-28, 28)
ax.set_ylim(-0.14 * vrch, 1.22 * vrch)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, loc="center right", framealpha=0.95)
ax.set_title(f"Týž úhel jako příčný profil: {POLOH} čísel, která solver ladí\n"
             "vršek stojí nad nádorem, nad míchou profil klesá", fontsize=14, pad=12)

# ---------------------------------------------------------------- vsech osm uhlu
vnitrni = gs[1, :].subgridspec(1, len(UHLY), wspace=0.05)
for u, uhel in enumerate(UHLY):
    ax = fig.add_subplot(vnitrni[0, u])
    rez(ax, dosah=46)
    suma, zapnuto = 0.0, 0
    for j, b, w in svazky_uhlu(u):
        zap = w > PRAH
        suma, zapnuto = suma + w, zapnuto + int(zap)
        pas(ax, b, uhel, ORANZ if zap else SEDA, vyraznost(w), zorder=3)
    a, z = G["svazky"][u * POLOH], G["svazky"][(u + 1) * POLOH - 1]
    ax.plot([a["hx"], z["hx"]], [a["hy"], z["hy"]], color="#8b939c", linewidth=2.4,
            solid_capstyle="round", zorder=2)
    ax.set_title(f"{uhel}°\n{zapnuto}/{POLOH} zapnuto\nΣ = {suma:.0f}", fontsize=12, pad=8,
                 color="#1c1c1e" if suma > 1 else "#8b939c")

# ------------------------------------------- vsechna cisla po skupinach podle uhlu
MEZERA = 3.5                              # mezera mezi skupinami, ve sirkach sloupce
ax = fig.add_subplot(gs[2, :])
stred_skupin = []
for u, uhel in enumerate(UHLY):
    x0 = u * (POLOH + MEZERA)
    xs = x0 + np.arange(POLOH)
    stred_skupin.append(x0 + (POLOH - 1) / 2)
    ax.axvspan(x0 - 0.8, x0 + POLOH - 0.2, color="#f2f4f7" if u % 2 == 0 else "#ffffff",
               zorder=0)
    ax.bar(xs, W[u * POLOH:(u + 1) * POLOH], width=0.82, color=ORANZ, zorder=2)

uroven = np.array(IT["rovno"])[0]          # srovnavaci hladina rovnomerneho ozareni
ax.axhline(uroven, color=POPISEK, linewidth=1.4, linestyle=(0, (6, 4)), zorder=3)
ax.text(stred_skupin[5], uroven + 1.0, "hladina rovnoměrného ozáření",
        fontsize=10.5, ha="center", va="bottom", color=POPISEK)

for u, uhel in enumerate(UHLY):
    th = np.deg2rad(uhel)
    dx, dy = -np.sin(th), np.cos(th)       # smer svazku; dy je v souradnicich rezu (dolu)
    ax.annotate("", xy=(stred_skupin[u], -0.155), xycoords=("data", "axes fraction"),
                xytext=(-dx * 12, dy * 12), textcoords="offset points",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", linewidth=1.6,
                                color="#1c1c1e" if W[u * POLOH:(u + 1) * POLOH].sum() > PRAH
                                else "#adb5bd"))
    ax.annotate(f"{uhel}°", xy=(stred_skupin[u], -0.30), xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=12, annotation_clip=False,
                color="#1c1c1e" if W[u * POLOH:(u + 1) * POLOH].sum() > PRAH else "#adb5bd")

ax.set_xlim(-1.5, (len(UHLY) - 1) * (POLOH + MEZERA) + POLOH + 0.5)
ax.set_ylim(0, W.max() * 1.12)
ax.set_xticks([])
ax.set_ylabel("intenzita svazku", fontsize=13)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.set_title(f"Všech {len(W)} čísel pohromadě: osm skupin po {POLOH} polohách, "
             "ve stejném pořadí jako řezy nad grafem\n"
             "šipka pod skupinou ukazuje, kudy ten svazek letí — u 225° nezbyl ani jeden",
             fontsize=14, pad=12)

fig.suptitle(f"Radioterapie: co je těch {len(W)} ladených čísel — "
             f"{len(UHLY)} úhlů × {POLOH} příčných poloh",
             fontsize=17, y=0.995)
cesta = KOREN / "obrazky" / "radioterapie-svazky.svg"
fig.savefig(cesta, bbox_inches="tight")
print(f"uloženo {cesta} ({cesta.stat().st_size / 1024:.0f} kB), "
      f"zapnutých svazků celkem {(W > PRAH).sum()} z {len(W)}")
