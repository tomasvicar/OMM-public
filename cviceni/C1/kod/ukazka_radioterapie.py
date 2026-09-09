"""Planovani radioterapie jako konvexni optimalizacni uloha (ukazka pro C1).

Zadani slovy: v rezu pacientem lezi nador a hned vedle nej micha, kterou nesmime
prezarit. Ozarovac vysila uzke svazky z osmi uhlu a u kazdeho se da nastavit
intenzita. Jak je zvolit, aby nador dostal predepsanou davku a micha co nejmene?

    promenne     w_j >= 0, intenzita j-teho svazku (8 uhlu x 12 poloh = 96)
    data         D_ij ... davka do voxelu i od svazku j s jednotkovou intenzitou,
                 davka v celem rezu je tedy vektor d = D @ w
    minimize     3 * ||d_micha||^2 + ||d_zdrava||^2
    subject to   60 <= d_nador <= 69   (predpis 60 Gy, tolerance +15 %)
                 w >= 0                (zaporna intenzita neexistuje)

Ucelova funkce je soucet ctvercu linearnich funkci a omezeni jsou linearni, jde
tedy o konvexni kvadraticky program: kazde lokalni minimum je i globalni a
solver vraci certifikat optimality. Vaha 3 u michy je klinicka preference, ne
fyzika - je to prvni misto, kde se da s modelem hrat. Srovnavaci plan
"rovnomerne ozareni" ma vsechny svazky stejne, naskalovane tak, aby nador dostal
tutez predepsanou davku. Spousteni z korene repozitare:
    uv run python cviceni/C1/kod/ukazka_radioterapie.py
"""

import time
from pathlib import Path

import numpy as np
import cvxpy as cp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREDPIS = 60.0        # predepsana davka do nadoru [Gy]
TOLERANCE = 1.15      # horni mez v nadoru = 1.15 * predpis
VAHA_MICHA = 3.0      # o kolik vic vadi davka v miche nez ve zdrave tkani
POLOH = 12            # pricnych poloh svazku v jednom uhlu
SIGMA = 2.6           # sirka svazku (smerodatna odchylka gaussovskeho profilu) [voxel]
ROZSAH = 18.0         # krajni polohy svazku, +-ROZSAH kolem osy [voxel]

# --- geometrie rezu: 64 x 64 voxelu, kruhove telo, nador a micha vedle sebe ---
N = 64
yy, xx = np.mgrid[0:N, 0:N]
cx = cy = (N - 1) / 2
telo = np.hypot(xx - cx, yy - cy) < 30
nador = np.hypot(xx - cx - 6, yy - cy + 4) < 7
micha = np.hypot(xx - cx + 8, yy - cy - 6) < 4
zdrava = telo & ~nador & ~micha

# --- matice davek D: sloupec = mapa davky od jednoho svazku s intenzitou 1 ---
sloupce = []
UHLY = [0, 45, 90, 135, 180, 225, 270, 315]
for uhel in np.deg2rad(UHLY):
    pricne = (xx - cx) * np.cos(uhel) + (yy - cy) * np.sin(uhel)   # napric svazkem
    hloubka = -(xx - cx) * np.sin(uhel) + (yy - cy) * np.cos(uhel)  # podel svazku
    for posun in np.linspace(-ROZSAH, ROZSAH, POLOH):
        profil = np.exp(-0.5 * ((pricne - posun) / SIGMA) ** 2)     # sirka svazku
        utlum = np.exp(-0.02 * (hloubka + 32))                      # utlum s hloubkou
        sloupce.append((profil * utlum * telo).ravel())
D = np.array(sloupce).T
print(f"matice davek D: {D.shape[0]} voxelu x {D.shape[1]} svazku")

# --- reseni konvexniho QP ---
w = cp.Variable(D.shape[1], nonneg=True)
davka_nador, davka_micha, davka_zdrava = D[nador.ravel()], D[micha.ravel()], D[zdrava.ravel()]
uloha = cp.Problem(
    cp.Minimize(VAHA_MICHA * cp.sum_squares(davka_micha @ w) + cp.sum_squares(davka_zdrava @ w)),
    [davka_nador @ w >= PREDPIS, davka_nador @ w <= TOLERANCE * PREDPIS],
)
cas = time.perf_counter()
uloha.solve(solver=cp.CLARABEL)
cas = time.perf_counter() - cas
print(f"status: {uloha.status}, doba reseni: {cas:.1f} s")

# --- plany k porovnani ---------------------------------------------------
# Krome optima jeste tri plany, ktere nikdo neoptimalizoval: rovnomerne ozareni
# a dva "od stolu". Vsechny tri se skaluji tak, aby nejchladnejsi voxel nadoru
# dostal predepsanych 60 Gy -- lisi se jen tim, ktere svazky jsou zapnute.
STRED_NADOR = np.array([6.0, -4.0])   # stred nadoru vuci izocentru [voxel]
R_NADOR = 7

trefa_nadoru, uhel_svazku = [], []
for uhel in np.deg2rad(UHLY):
    pricne_nador = STRED_NADOR[0] * np.cos(uhel) + STRED_NADOR[1] * np.sin(uhel)
    for posun in np.linspace(-ROZSAH, ROZSAH, POLOH):
        trefa_nadoru.append(abs(posun - pricne_nador) < R_NADOR)   # osa svazku jde nadorem
        uhel_svazku.append(int(round(np.degrees(uhel))))
trefa_nadoru, uhel_svazku = np.array(trefa_nadoru), np.array(uhel_svazku)


def naskaluj(w_):
    """Plan naskalovany tak, aby nejchladnejsi voxel nadoru dostal predpis."""
    plan = (D @ w_).reshape(N, N)
    return plan * (PREDPIS / plan[nador].min())


plan_opt = (D @ w.value).reshape(N, N)
PLANY = [
    ("Rovnoměrné ozáření", naskaluj(np.ones(D.shape[1]))),
    ("Mířím na nádor", naskaluj(trefa_nadoru.astype(float))),
    ("Jen dva kolmé směry", naskaluj((trefa_nadoru & np.isin(uhel_svazku, [0, 90])).astype(float))),
    ("Optimalizovaný plán", plan_opt),
]
plan_rovno = PLANY[0][1]

ZKRATKY = ["rovnoměrně", "na nádor", "2 směry", "optimum"]
hlavicka = "".join(f"{z:>14}" for z in ZKRATKY)
print(f"\n{'veličina':<34}{hlavicka}\n" + "-" * (34 + 14 * len(PLANY)))
for popis, f in [("dávka v nádoru, min [Gy]", lambda p: p[nador].min()),
                 ("dávka v nádoru, max [Gy]", lambda p: p[nador].max()),
                 ("mícha, maximální dávka [Gy]", lambda p: p[micha].max()),
                 ("mícha, střední dávka [Gy]", lambda p: p[micha].mean()),
                 ("zdravá tkáň, střední dávka [Gy]", lambda p: p[zdrava].mean())]:
    print(f"{popis:<34}" + "".join(f"{f(pl):>14.1f}" for _, pl in PLANY))
print(f"\nhorní mez v nádoru je {TOLERANCE * PREDPIS:.0f} Gy; plány, které ji překročí, "
      f"jsou NEPŘÍPUSTNÉ,\nať mají v míše jakkoli hezké číslo")

# --- obrazek 1: ctyri mapy davky ve spolecne skale ---
plt.rcParams.update({"font.size": 12})
MEZ = TOLERANCE * PREDPIS
vmax = max(pl.max() for _, pl in PLANY)      # spolecna skala, jinak se plany neporovnaji
BARVY = ["#5c5c66", "#e8590c", "#7c3aed", "#1a7f37"]

fig, osy = plt.subplots(2, 2, figsize=(11.2, 10.8))   # 2x2, at se vejdou dvouradkove titulky
osy = osy.ravel()
for osa, (nazev, plan), barva in zip(osy, PLANY, BARVY):
    obr = osa.imshow(plan, cmap="inferno", vmin=0, vmax=vmax)
    osa.contour(nador, colors="w", linewidths=2.2)
    osa.contour(micha, colors="#00e5ff", linewidths=2.0, linestyles="dashed")
    prekrocil = plan[nador].max() > MEZ + 0.05
    osa.set_title(f"{nazev}\nmícha max {plan[micha].max():.0f} · "
                  f"nádor max {plan[nador].max():.0f} Gy", fontsize=14,
                  color="#b42318" if prekrocil else "#1c1c1e")
    if prekrocil:
        osa.text(0.5, -0.03, f"NEPŘÍPUSTNÝ: mez je {MEZ:.0f} Gy", transform=osa.transAxes,
                 ha="center", va="top", fontsize=12.5, fontweight="bold", color="#b42318")
    osa.axis("off")
if True:                                     # popsat struktury jen v prvni mape
    osy[0].annotate("nádor", xy=(37.5, 21), xytext=(37.5, 7), color="w", fontsize=13,
                    fontweight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="w", linewidth=1.2))
    osy[0].annotate("mícha", xy=(23.5, 42), xytext=(15, 55), color="#00e5ff", fontsize=13,
                    fontweight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="#00e5ff", linewidth=1.2))
fig.subplots_adjust(wspace=0.02, hspace=0.22)
plt.colorbar(obr, ax=osy.tolist(), shrink=0.65, pad=0.02, label="dávka [Gy], společná škála")
cesta = Path(__file__).resolve().parents[1] / "obrazky" / "radioterapie.png"
fig.savefig(cesta, dpi=120, bbox_inches="tight")
print(f"\nobrázek uložen: {cesta}")

# --- obrazek 2: dose-volume histogram vsech ctyr planu, samostatne a velky ---
# Cist se ma takhle: pro kazdou strukturu, jaky PODIL jejiho objemu dostal aspon
# danou davku. Krivka nadoru ma stat co nejvic vpravo a byt svisla, krivka michy
# co nejvic vlevo. Barva je plan, styl je struktura.
fig, osa = plt.subplots(figsize=(13.0, 7.6))
osa.axvspan(MEZ, vmax, color="#b42318", alpha=0.05, zorder=0)
for (nazev, plan), barva in zip(PLANY, BARVY):
    for maska, styl in [(nador, "-"), (micha, "--")]:
        hodnoty = np.sort(plan[maska])[::-1]
        osa.plot(hodnoty, np.linspace(0, 100, hodnoty.size), styl, color=barva, lw=2.8,
                 label=nazev if styl == "-" else None, zorder=3)
osa.axvline(PREDPIS, color="k", lw=1.2, ls=":", zorder=2)
osa.axvline(MEZ, color="#b42318", lw=1.8, ls=":", zorder=2)
osa.text(PREDPIS - 1.2, 103, f"předpis {PREDPIS:.0f} Gy", fontsize=12.5, ha="right",
         va="bottom", color="#1c1c1e")
osa.text(MEZ + 1.2, 103, f"horní mez {MEZ:.0f} Gy — vpravo odsud je plán nepřípustný",
         fontsize=12.5, ha="left", va="bottom", color="#b42318", fontweight="bold")
legenda = osa.legend(fontsize=13, loc="center left", bbox_to_anchor=(0.30, 0.62),
                     title="plán (barva)", framealpha=0.95)
legenda.get_title().set_fontsize(13)
osa.add_artist(legenda)
styly = [plt.Line2D([], [], color="0.3", lw=2.8, ls="-", label="nádor — má být vpravo a svislá"),
         plt.Line2D([], [], color="0.3", lw=2.8, ls="--", label="mícha — má být co nejvíc vlevo")]
osa.legend(handles=styly, fontsize=13, loc="center left", bbox_to_anchor=(0.30, 0.30),
           title="struktura (styl čáry)", framealpha=0.95).get_title().set_fontsize(13)
osa.set(xlim=(0, vmax), ylim=(0, 101))
osa.set_xlabel("dávka [Gy]", fontsize=14)
osa.set_ylabel("podíl objemu struktury s aspoň touto dávkou [%]", fontsize=14)
osa.set_title("Dose-volume histogram: jak se ozařovací plán čte na klinice", fontsize=16, pad=28)
osa.tick_params(labelsize=12.5)
osa.grid(alpha=0.3)
cesta_dvh = Path(__file__).resolve().parents[1] / "obrazky" / "radioterapie-dvh.png"
fig.savefig(cesta_dvh, dpi=120, bbox_inches="tight")
print(f"obrázek uložen: {cesta_dvh}")

# --- kontrola dosazenim zpet: musi umet selhat, proto pocitame znovu z w ---
davka = (D @ w.value)[nador.ravel()]
assert uloha.status == cp.OPTIMAL, f"solver nenasel optimum: {uloha.status}"
assert davka.min() >= PREDPIS - 1e-6, "nador nedostal predepsanou davku"
assert davka.max() <= TOLERANCE * PREDPIS + 1e-6, "prekrocena horni mez v nadoru"
assert w.value.min() >= -1e-8, "zaporna intenzita svazku"
assert plan_opt[micha].max() < 0.5 * plan_rovno[micha].max(), "optimalizace miche nepomohla"
print("kontrola dosazenim zpet prosla")
