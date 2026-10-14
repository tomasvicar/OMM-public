"""Obrazky k rekonstrukci CT ve ctvrtem cviceni (report reporty/ct.html).

    uv run python cviceni/C4/kod/ct.py

Rekonstrukce pocita stejnym kodem jako kod/cisla.py (stejna projekce, Adam
s lr = 0,05, 300 kroku, orez na <0, 1>, seed 0 pro sum) a overi, ze RMSE
sedi s data/cisla.json. Navic spocita hustsi radu lambda pro graf RMSE vs.
lambda; ta cisla jen vypise, do cisla.json je nezapisuje.

Vystupy v cviceni/C4/obrazky/:
    ct-schema.svg      paralelni projekce elipsoveho fantomu pod jednim uhlem
    ct-sinogram.png    skutecny rez a jeho sinogram (60 uhlu)
    ct-prubeh.png      rekonstrukce po 5, 20, 50 a 300 krocich vedle originalu
    ct-prubeh.gif      tataz rekonstrukce jako smycka (prvni snimek = vysledek)
    ct-uhly.png        60 vs. 12 uhlu
    ct-tv.png          sum sigma = 1 a penalizace TV pro lambda z cisla.json
    ct-tv-rmse.svg     RMSE vs. lambda (tvar U)
"""

import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from spolecne import DATA, nacti, projekce, tv, uhly  # noqa: E402

OBRAZKY = Path("cviceni/C4/obrazky")
OBRAZKY.mkdir(parents=True, exist_ok=True)
AKCENT, CERV, MODR, ZEL = "#007f86", "#c73e1d", "#1f4e79", "#1a7f37"
SEDA = "#5c5c66"
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})

CISLA = json.loads((DATA / "cisla.json").read_text())["ct"]

ct = nacti("ct-128.png")
n = ct.shape[-1]
yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
kruh = (xx ** 2 + yy ** 2 <= 1).float()


def rmse(x):
    return (((x - ct) ** 2 * kruh).sum() / kruh.sum()).sqrt().item()


def rekonstruuj(pocet_uhlu, sum_sigma=0.0, lam=0.0, kroku=300, lr=0.05, snimky=()):
    """Tataz rekonstrukce jako v cisla.py; navic vrati obrazy po krocich `snimky`."""
    u = uhly(pocet_uhlu)
    torch.manual_seed(0)
    sino = projekce(ct, u).detach()
    sino = sino + sum_sigma * torch.randn_like(sino)
    x = torch.zeros_like(ct, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    ulozene, ztraty = {}, []
    for k in range(1, kroku + 1):
        opt.zero_grad()
        ztrata = ((projekce(x, u) - sino) ** 2).mean() + lam * tv(x)
        ztrata.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        ztraty.append(ztrata.item())
        if k in snimky:
            ulozene[k] = x.detach()[0, 0].clone().numpy()
    return x.detach(), rmse(x.detach()), ulozene, ztraty, sino


def over(rmse_spoctene, rmse_json, popis):
    assert round(rmse_spoctene, 4) == rmse_json, f"{popis}: {rmse_spoctene:.4f} != {rmse_json}"
    print(f"  {popis}: RMSE {rmse_spoctene:.4f} (cisla.json {rmse_json}) OK")


def cislo(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def cislo4(v):
    """RMSE na ctyri desetinna mista bez koncovych nul, jako v cisla.json."""
    return f"{round(v, 4):g}".replace(".", ",")


def uloz_png(fig, jmeno, dpi=150):
    fig.savefig(OBRAZKY / jmeno, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"zapsano {OBRAZKY / jmeno}")


def uloz_svg(fig, jmeno):
    fig.savefig(OBRAZKY / jmeno, bbox_inches="tight", facecolor="white",
                metadata={"Date": None})
    plt.close(fig)
    print(f"zapsano {OBRAZKY / jmeno}")


def obraz(ax, a, popis, pismo=11):
    ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(popis, fontsize=pismo)


# --- 1. schema projekce ---------------------------------------------------
# elipsovy fantom (cx, cy, a, b, natoceni ve stupnich, hodnota); projekce
# elipsy je analyticka: p(t) = 2 h a b sqrt(s^2 - t^2) / s^2
FANTOM = [(0, 0, 0.72, 0.9, 0, 0.5), (-0.22, 0.1, 0.13, 0.32, -18, 0.5),
          (0.25, 0.05, 0.16, 0.24, 18, 0.35), (0.05, -0.5, 0.1, 0.1, 0, 0.4)]


def projekce_elips(theta, t):
    p = np.zeros_like(t)
    for cx, cy, a, b, fi, h in FANTOM:
        th = theta - math.radians(fi)
        s2 = (a * math.cos(th)) ** 2 + (b * math.sin(th)) ** 2
        tt = t - (cx * math.cos(theta) + cy * math.sin(theta))
        p += np.where(tt ** 2 < s2, 2 * h * a * b * np.sqrt(np.clip(s2 - tt ** 2, 0, None)) / s2, 0)
    return p


def schema():
    theta = math.radians(30)
    e = np.array([math.cos(theta), math.sin(theta)])       # smer detektoru (osa t)
    d = np.array([-math.sin(theta), math.cos(theta)])      # smer paprsku
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.set_aspect("equal")
    ax.axis("off")
    for cx, cy, a, b, fi, h in FANTOM:
        barva = plt.cm.gray(0.25 + 0.6 * h / 0.5 if (cx, cy) == (0, 0) else 0.95)
        ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=fi, facecolor=barva,
                             edgecolor="#333", lw=1))
    # paprsky: rovnobezne primky ve smeru d, posunute po e
    for t in np.linspace(-0.9, 0.9, 13):
        z = t * e - 1.35 * d
        k = t * e + 1.2 * d
        ax.add_patch(FancyArrowPatch(z, k, arrowstyle="-|>", mutation_scale=9,
                                     color=CERV, lw=0.9, alpha=0.75))
    # detektor za objektem a profil projekce na nem
    zaklad = 1.3 * d
    ax.plot(*np.array([zaklad - 1.15 * e, zaklad + 1.15 * e]).T, color=MODR, lw=3)
    t = np.linspace(-1.1, 1.1, 300)
    p = projekce_elips(theta, t)
    profil = zaklad[:, None] + np.outer(e, t) + np.outer(d, 0.5 * p / p.max())
    ax.plot(*profil, color=MODR, lw=1.8)
    ax.text(*(zaklad + 1.25 * e + 0.05 * d), "detektor", color=MODR, fontsize=11,
            rotation=math.degrees(theta), ha="left", va="center")
    ax.text(0.55, -1.78, "rovnoběžné paprsky ze zdroje",
            color=CERV, fontsize=11, ha="center", va="top")
    # uhel theta: odklon paprsku od svisle osy u krajniho paprsku
    z = 0.9 * e - 1.35 * d
    ax.plot([z[0], z[0]], [z[1], z[1] + 0.75], color=SEDA, ls="--", lw=1)
    oblouk = np.linspace(math.pi / 2, math.pi / 2 + theta, 20)
    ax.plot(z[0] + 0.5 * np.cos(oblouk), z[1] + 0.5 * np.sin(oblouk), color=SEDA, lw=1)
    ax.text(z[0] + 0.05, z[1] + 0.82, "θ", color=SEDA, fontsize=13, ha="center")
    ax.set_xlim(-1.75, 1.95)
    ax.set_ylim(-2.0, 2.05)
    # vpravo sinogram fantomu, zvyrazneny radek pro theta = 30 stupnu
    th = np.radians(np.arange(0, 180, 1.0))
    tt = np.linspace(-1.1, 1.1, 220)
    sino = np.array([projekce_elips(a, tt) for a in th])
    ax2.imshow(sino, cmap="gray", aspect="auto", extent=[tt[0], tt[-1], 180, 0])
    ax2.axhline(30, color=MODR, lw=2)
    ax2.text(1.08, 27, "θ = 30°: jeden řádek", color="white", fontsize=10, ha="right", va="bottom")
    ax2.set_xlabel("poloha na detektoru t")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax2.set_ylabel("úhel θ [°]")
    fig.tight_layout()
    uloz_svg(fig, "ct-schema.svg")


# --- 2. rez a sinogram ------------------------------------------------------
def rez_a_sinogram(sino60):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    obraz(a1, ct[0, 0].numpy(), "řez 128 × 128 pixelů (to hledáme)")
    a2.imshow(sino60.numpy(), cmap="gray", aspect="auto", interpolation="nearest",
              extent=[0, n, 180, 0])
    a2.set_xlabel("poloha na detektoru [px]")
    a2.set_ylabel("úhel θ [°]  (60 úhlů po 3°)")
    fig.tight_layout()
    uloz_png(fig, "ct-sinogram.png")


# --- 3. prubeh rekonstrukce ----------------------------------------------
KROKY = (5, 20, 50, 300)


def prubeh(snimky):
    fig, osy = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, k in zip(osy, KROKY):
        obraz(ax, snimky[k], f"krok {k}, RMSE {cislo4(rmse(torch.tensor(snimky[k])))}", 15)
    obraz(osy[-1], ct[0, 0].numpy(), "originál", 15)
    fig.tight_layout()
    uloz_png(fig, "ct-prubeh.png")


def gif(snimky, ztraty):
    """Prvni snimek = vysledek po 300 krocich, pak cela cesta od nuly,
    posledni snimek ma dlouhou vydrz."""
    kroky = sorted(snimky)
    poradi = [300] + kroky
    obrazky = []
    for k in poradi:
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
        obraz(a, snimky[k], f"krok {k}")
        b.semilogy(range(1, 301), ztraty, color="#c9cdd3", lw=1.5)
        b.semilogy(range(1, k + 1), ztraty[:k], color=AKCENT, lw=2)
        b.plot(k, ztraty[k - 1], "o", color=AKCENT)
        b.set_xlabel("krok Adamu")
        b.set_ylabel("ztráta (MSE sinogramu)")
        b.set_xlim(0, 305)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor="white")
        plt.close(fig)
        obrazky.append(Image.open(buf).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=128))
    doby = [2500] + [120] * (len(kroky) - 1) + [3000]
    obrazky[0].save(OBRAZKY / "ct-prubeh.gif", save_all=True, append_images=obrazky[1:],
                    duration=doby, loop=0, optimize=True)
    print(f"zapsano {OBRAZKY / 'ct-prubeh.gif'}")


# --- 4. pocet uhlu -----------------------------------------------------------
def uhly_obr(x60, r60, x12, r12):
    fig, osy = plt.subplots(1, 3, figsize=(10, 3.7))
    obraz(osy[0], x60[0, 0].numpy(), f"60 úhlů, RMSE {cislo4(r60)}")
    obraz(osy[1], x12[0, 0].numpy(), f"12 úhlů, RMSE {cislo4(r12)}")
    obraz(osy[2], ct[0, 0].numpy(), "originál")
    fig.tight_layout()
    uloz_png(fig, "ct-uhly.png")


# --- 5. sum a TV ----------------------------------------------------------
def tv_obr(vysledky):
    fig, osy = plt.subplots(1, len(vysledky), figsize=(13, 3.4))
    for ax, (lam, (x, r)) in zip(osy, vysledky.items()):
        obraz(ax, x[0, 0].numpy(), f"λ = {cislo4(lam)}, RMSE {cislo4(r)}", 15)
    fig.tight_layout()
    uloz_png(fig, "ct-tv.png")


def tv_rmse(rada, tabulka):
    lam = np.array(sorted(rada))
    r = np.array([rada[k] for k in lam])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(lam, r, "-", color=AKCENT, lw=2)
    ax.plot(lam, r, "o", color=AKCENT, ms=4)
    for k, v in tabulka.items():
        ax.plot(k, v, "o", color=CERV if k == min(tabulka, key=tabulka.get) else MODR, ms=8, zorder=3)
        ax.annotate(cislo4(v), (k, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10)
    ax.axhline(CISLA["zakladni"]["rmse"], color=SEDA, ls=":", lw=1)
    ax.text(lam[-1], CISLA["zakladni"]["rmse"] + 0.002, "bez šumu, λ = 0", color=SEDA,
            ha="right", va="bottom", fontsize=10)
    ax.text(0.0003, r[0] + 0.012, "málo: zůstane šum", color=SEDA, fontsize=10)
    ax.text(lam[-1], r[-1] - 0.012, "moc: rozmazání", color=SEDA, fontsize=10, ha="right")
    ax.set_xlabel("váha penalizace λ")
    ax.set_ylabel("RMSE uvnitř kruhu")
    ax.set_ylim(0, max(r) * 1.18)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: cislo(v, 3)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: cislo(v, 2)))
    fig.tight_layout()
    uloz_svg(fig, "ct-tv-rmse.svg")


if __name__ == "__main__":
    schema()

    print("rekonstrukce 60 uhlu")
    snimky_kroky = set(KROKY) | {1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 75,
                                 90, 110, 130, 160, 200, 250, 300}
    x60, r60, snimky, ztraty, sino60 = rekonstruuj(60, snimky=snimky_kroky)
    over(r60, CISLA["zakladni"]["rmse"], "60 uhlu")
    rez_a_sinogram(sino60)
    prubeh(snimky)
    gif(snimky, ztraty)

    x12, r12, *_ = rekonstruuj(12)
    over(r12, CISLA["malo_uhlu"]["rmse"], "12 uhlu")
    uhly_obr(x60, r60, x12, r12)

    print("sum sigma = 1 a TV")
    tabulka, vysledky = {}, {}
    for klic, rmse_json in CISLA["sum"]["tv"].items():
        lam = float(klic)
        x, r, *_ = rekonstruuj(60, CISLA["sum"]["sigma"], lam)
        over(r, rmse_json, f"lambda {klic}")
        tabulka[lam], vysledky[lam] = r, (x, r)
    tv_obr(vysledky)

    rada = dict(tabulka)
    for lam in (0.0003, 0.0006, 0.0015, 0.002, 0.003, 0.006, 0.008):
        rada[lam] = rekonstruuj(60, CISLA["sum"]["sigma"], lam)[1]
    print("RMSE vs. lambda (navic, jen pro graf):",
          ", ".join(f"{k:g}: {rada[k]:.4f}" for k in sorted(rada)))
    tv_rmse(rada, tabulka)
