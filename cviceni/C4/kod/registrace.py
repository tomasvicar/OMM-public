"""Registrace MRI ctvrteho cviceni: obrazky pro report registrace.html.

    uv run python cviceni/C4/kod/registrace.py

Tuha registrace: posun (tx, ty) a rotace theta, tedy tri parametry. Pohnuty
snimek vznikne ze znameho MRI rezu se znamou pravdou (tx = 0,10, ty = -0,06,
theta = 12 stupnu) plus sum sigma = 0,02 (seed 0). Hleda se p, aby
transformuj(mri, p) ~ pohnuty; ztrata je MSE, Adam s lr = 0,02, 300 kroku,
start (0, 0, 0). Gradient pocita autograd pres bilinearni interpolaci
(grid_sample) a cos/sin.

Past: pri pravde theta = 150 stupnu skonci start z nuly v lokalnim minimu
kolem -30 stupnu (mozek je zhruba elipsa, otoceni o 180 stupnu ji skoro
prekryje). Start ze 180 stupnu najde globalni minimum.

Skript prepocita oba behy, overi je proti data/cisla.json (cisla.py)
a nakresli:
    registrace-snimky.png       referencni a pohnuty snimek, prekryti pred a po
    registrace-konvergence.svg  MSE a parametry behem 300 kroku
    registrace-rez.svg          MSE pres theta pri pravde 150 stupnu + drahy Adama
    registrace-prubeh.gif       animace prekryti behem registrace
                                (prvni snimek = vysledek, konec s dlouhou vydrzi)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from spolecne import DATA, nacti, transformuj  # noqa: E402

OBRAZKY = Path("cviceni/C4/obrazky")
SVG = {"Date": None}
plt.rcParams.update({"svg.hashsalt": "mpc-omm-c4-registrace", "svg.fonttype": "path",
                     "font.size": 10})
AKCENT, CERV, MODR, ZEL = "#007f86", "#c73e1d", "#1f4e79", "#1a7f37"

CISLA = json.loads((DATA / "cisla.json").read_text())
REG, PAST = CISLA["registrace"], CISLA["registrace_past"]
PRAVDA = REG["pravda"]


def c(x, d=2):
    """Cislo s desetinnou carkou."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ data ----
# stejne poradi nahodnych cisel jako cisla.py: nejdriv sum pro 12, pak pro 150 stupnu
torch.manual_seed(REG["seed"])
MRI = nacti(REG["obraz"])
P_PRAVDA = torch.tensor([PRAVDA["tx"], PRAVDA["ty"], math.radians(PRAVDA["theta_stupne"])])
POHNUTY = (transformuj(MRI, P_PRAVDA) + REG["sum_sigma"] * torch.randn_like(MRI)).detach()
P_PRAVDA_150 = torch.tensor([PRAVDA["tx"], PRAVDA["ty"], math.radians(PAST["pravda_theta_stupne"])])
POHNUTY_150 = (transformuj(MRI, P_PRAVDA_150) + REG["sum_sigma"] * torch.randn_like(MRI)).detach()


def registruj(cil, start, kroku=REG["kroku"], lr=REG["lr"]):
    """Adam nad tremi parametry; vrati drahu parametru a MSE po krocich."""
    p = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    drahy, ztraty = [p.detach().clone()], []
    for _ in range(kroku):
        opt.zero_grad()
        ztrata = ((transformuj(MRI, p) - cil) ** 2).mean()
        ztrata.backward()
        opt.step()
        drahy.append(p.detach().clone())
        ztraty.append(ztrata.item())
    return torch.stack(drahy), np.array(ztraty)


DRAHA, ZTRATY = registruj(POHNUTY, REG["start"])
P_KONEC = DRAHA[-1]
assert abs(P_KONEC[0] - REG["vysledek"]["tx"]) < 1e-3
assert abs(P_KONEC[1] - REG["vysledek"]["ty"]) < 1e-3
assert abs(math.degrees(P_KONEC[2]) - REG["vysledek"]["theta_stupne"]) < 0.05
assert abs(ZTRATY[-1] - REG["mse"]) < 2e-6

DRAHA_0, ZTRATY_0 = registruj(POHNUTY_150, [0.0, 0.0, 0.0])                  # past
DRAHA_180, ZTRATY_180 = registruj(POHNUTY_150, [0.0, 0.0, math.pi])
assert abs(math.degrees(DRAHA_0[-1, 2]) - PAST["z_nuly"]["theta_stupne"]) < 0.05
assert abs(ZTRATY_0[-1] - PAST["z_nuly"]["mse"]) < 2e-6
assert abs(math.degrees(DRAHA_180[-1, 2]) % 360 - 150) < 0.05


# ---------------------------------------------------- snimky pred a po ----
def prekryti(a, b):
    """Cil a zelene, transformovany snimek b purpurove; kde sedi, je seda."""
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    return np.stack([b, a, b], axis=-1)


def obr_snimky():
    mri = MRI[0, 0].numpy()
    pohnuty = POHNUTY[0, 0].numpy()
    po = transformuj(MRI, P_KONEC)[0, 0].numpy()
    panely = [
        (mri, "referenční", "gray"),
        (pohnuty, "pohnutý", "gray"),
        (prekryti(pohnuty, mri), "překrytí před", None),
        (prekryti(pohnuty, po), "překrytí po", None),
    ]
    fig, osy = plt.subplots(1, 4, figsize=(10, 2.85))
    for ax, (obr, popis, cmap) in zip(osy, panely):
        ax.imshow(obr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(popis, fontsize=11)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OBRAZKY / "registrace-snimky.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------- konvergence ----
def obr_konvergence():
    k = np.arange(len(DRAHA))
    fig, osy = plt.subplots(1, 3, figsize=(10, 2.9))
    ax = osy[0]
    ax.semilogy(np.arange(1, len(ZTRATY) + 1), ZTRATY, color=AKCENT, lw=2)
    ax.axhline(REG["sum_sigma"] ** 2, color="#6b7280", ls="--", lw=1)
    ax.text(len(ZTRATY), REG["sum_sigma"] ** 2 * 1.25, "σ² šumu", ha="right", va="bottom",
            fontsize=9, color="#6b7280")
    ax.set_xlabel("krok")
    ax.set_ylabel("MSE")
    ax = osy[1]
    for i, (jm, barva, pr) in enumerate([("$t_x$", MODR, PRAVDA["tx"]),
                                          ("$t_y$", CERV, PRAVDA["ty"])]):
        ax.plot(k, DRAHA[:, i], color=barva, lw=2, label=jm)
        ax.axhline(pr, color=barva, ls="--", lw=1)
    ax.set_xlabel("krok")
    ax.set_ylabel("posun (normované souřadnice)")
    ax.legend(loc="center right", fontsize=9)
    ax = osy[2]
    ax.plot(k, np.degrees(DRAHA[:, 2].numpy()), color=ZEL, lw=2, label=r"$\theta$")
    ax.axhline(PRAVDA["theta_stupne"], color=ZEL, ls="--", lw=1)
    ax.set_xlabel("krok")
    ax.set_ylabel("rotace θ [°]")
    ax.text(len(DRAHA) - 1, PRAVDA["theta_stupne"] - 0.6, "pravda", ha="right", va="top",
            fontsize=9, color="#6b7280")
    for ax in osy:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, len(DRAHA) - 1)
    fig.tight_layout()
    fig.savefig(OBRAZKY / "registrace-konvergence.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ rez ztratou pres theta ----
def transformuj_davku(obr, tx, ty, th):
    """Totez co transformuj, ale pro davku parametru najednou (rychlejsi rez)."""
    cth, sth = torch.cos(th), torch.sin(th)
    theta = torch.stack([torch.stack([cth, -sth, tx], 1), torch.stack([sth, cth, ty], 1)], 1)
    mrizka = F.affine_grid(theta, (len(th), 1, *obr.shape[-2:]), align_corners=False)
    return F.grid_sample(obr.expand(len(th), -1, -1, -1), mrizka, align_corners=False)


def rez():
    """MSE pres theta: (1) pri spravnem posunu, (2) s nejlepsim posunem pro kazde theta."""
    stupne = np.arange(-180, 181, 2.0)
    th = torch.tensor(np.radians(stupne), dtype=torch.float32)
    n = len(th)
    with torch.no_grad():
        pri_pravde = ((transformuj_davku(MRI, torch.full((n,), PRAVDA["tx"]),
                                         torch.full((n,), PRAVDA["ty"]), th)
                       - POHNUTY_150) ** 2).mean(dim=(1, 2, 3)).numpy()
    # nejlepsi posun: pro kazde theta zvlast Adam jen nad (tx, ty), tri starty
    nejlepsi = np.full(n, np.inf)
    for start in ([PRAVDA["tx"], PRAVDA["ty"]], [-PRAVDA["tx"], -PRAVDA["ty"]], [0.0, 0.0]):
        t = torch.tensor([start] * n, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([t], lr=0.01)
        for _ in range(300):
            opt.zero_grad()
            mse = ((transformuj_davku(MRI, t[:, 0], t[:, 1], th) - POHNUTY_150) ** 2
                   ).mean(dim=(1, 2, 3))
            mse.sum().backward()
            opt.step()
        with torch.no_grad():
            mse = ((transformuj_davku(MRI, t[:, 0], t[:, 1], th) - POHNUTY_150) ** 2
                   ).mean(dim=(1, 2, 3)).numpy()
        nejlepsi = np.minimum(nejlepsi, mse)
    return stupne, pri_pravde, nejlepsi


def obr_rez():
    stupne, pri_pravde, nejlepsi = rez()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(stupne, nejlepsi, color=AKCENT, lw=2.2, label="s nejlepším posunem pro každé θ")
    ax.plot(stupne, pri_pravde, color="#9ca3af", lw=1.5, ls="--",
            label="při správném posunu $t_x$, $t_y$")

    def zabal(x):
        return (x + 180) % 360 - 180

    th0 = zabal(np.degrees(DRAHA_0[:, 2].numpy()))
    th180 = zabal(np.degrees(DRAHA_180[:, 2].numpy()))
    ax.plot(th0[1:], ZTRATY_0, ".", color=CERV, ms=3.5, label="Adam ze startu 0°")
    ax.plot(th180[1:], ZTRATY_180, ".", color=ZEL, ms=3.5, label="Adam ze startu 180°")
    lok = PAST["z_nuly"]
    ax.annotate(f"lokální minimum\nθ = {c(lok['theta_stupne'])}°, MSE = {c(lok['mse'], 4)}",
                xy=(lok["theta_stupne"], lok["mse"]), xytext=(-125, 0.0035),
                fontsize=9, color=CERV, arrowprops=dict(arrowstyle="->", color=CERV))
    glob = next(m for m in PAST["multistart"] if m["start_stupne"] == 180)
    ax.annotate(f"globální minimum\nθ = 150°, MSE = {c(glob['mse'], 4)}",
                xy=(150, glob["mse"]), xytext=(55, 0.0035),
                fontsize=9, color=ZEL, ha="right", arrowprops=dict(arrowstyle="->", color=ZEL))
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 0.045)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_xlabel("rotace θ [°]")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(OBRAZKY / "registrace-rez.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)
    i = int(np.argmin(np.abs(stupne - (-30))))
    print(f"rez: nejlepsi MSE u -30 st. {nejlepsi[i]:.4f}, pri spravnem posunu {pri_pravde[i]:.4f}")


# --------------------------------------------------------------- animace ----
def obr_gif():
    pohnuty = POHNUTY[0, 0].numpy()
    kroky = list(range(0, 41, 2)) + list(range(45, 101, 5)) + list(range(120, 301, 30))
    pismo = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 14)
    snimky = []
    for k in kroky:
        with torch.no_grad():
            obr = transformuj(MRI, DRAHA[k])[0, 0].numpy()
        rgb = (prekryti(pohnuty, obr) * 255).astype(np.uint8)
        im = Image.fromarray(rgb).resize((256, 256), Image.NEAREST)
        platno = Image.new("RGB", (256, 282), "white")
        platno.paste(im, (0, 0))
        d = ImageDraw.Draw(platno)
        p = DRAHA[k]
        d.text((8, 261), f"krok {k}", fill=(28, 28, 30), font=pismo)
        d.text((100, 261), f"θ = {math.degrees(p[2]):.1f}°".replace(".", ","), fill=(28, 28, 30),
               font=pismo)
        snimky.append(platno.quantize(colors=96, dither=Image.Dither.NONE))
    # prvni snimek = vysledny stav (v PDF a tisku se animace neprehraje)
    snimky = [snimky[-1]] + snimky
    trvani = [1500] + [80] * (len(snimky) - 2) + [3000]
    snimky[0].save(OBRAZKY / "registrace-prubeh.gif", save_all=True,
                   append_images=snimky[1:], duration=trvani, loop=0, optimize=True)


if __name__ == "__main__":
    OBRAZKY.mkdir(parents=True, exist_ok=True)
    print(f"registrace: tx = {P_KONEC[0]:.4f}, ty = {P_KONEC[1]:.4f}, "
          f"theta = {math.degrees(P_KONEC[2]):.2f} st., MSE = {ZTRATY[-1]:.6f}")
    print(f"past ze 0: theta = {math.degrees(DRAHA_0[-1, 2]):.2f} st., MSE = {ZTRATY_0[-1]:.6f}; "
          f"ze 180: theta = {math.degrees(DRAHA_180[-1, 2]) % 360:.2f} st., "
          f"MSE = {ZTRATY_180[-1]:.6f}")
    th = np.degrees(DRAHA[:, 2].numpy())
    venku = np.nonzero(np.abs(th - PRAVDA["theta_stupne"]) >= 0.1)[0]
    nad = np.nonzero(ZTRATY >= 1.05 * ZTRATY[-1])[0]
    print(f"theta trvale do 0,1 st. od pravdy od kroku {venku[-1] + 1}, "
          f"nejvetsi prekmit {th.max():.2f} st. v kroku {th.argmax()}; "
          f"MSE trvale pod 1,05 x konecne od kroku {nad[-1] + 2}")
    obr_snimky()
    obr_konvergence()
    obr_rez()
    obr_gif()
    print("hotovo:", ", ".join(sorted(p.name for p in OBRAZKY.glob("registrace-*"))))
