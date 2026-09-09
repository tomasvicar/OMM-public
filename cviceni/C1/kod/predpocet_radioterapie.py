"""Predpocet rodiny ozarovacich planu pro interaktivni hriste v reportu.

Uloha je tataz jako v `ukazka_radioterapie.py` (konvexni QP, 96 promennych),
jen se resi opakovane pro celou mrizku hodnot vahy michy. V prohlizeci se
96promenna uloha resit neda, takze report jen kresli hotove vysledky odsud.

JSON obsahuje dvoje data:
  * `geometrie` + `intenzity` ... poloha vsech 96 svazku v rezu (usecky orezane
    telem pacienta) a jejich intenzity ve dvou planech, rovnomernem a
    optimalizovanem (vaha michy 3). Z toho report kresli obrazek toho, co presne
    je promennou ulohy: osm uhlu krat 12 poloh, u kazde jedno cislo.
  * `plany` ... Paretova fronta pres celou mrizku vah (cim vic vazi micha, tim
    min dostane a tim vic dostane zdrava tkan) vcetne DVH krivek. Report ji dnes
    nekresli, data se drzi pro pripad, ze by se interaktivni varianta vratila.

Vyrabi `cviceni/C1/data/radioterapie.json`, ktery je zaroven vlozeny inline
v `cviceni/C1/reporty/radioterapie.html` jako <script id="radioterapie-data">.
Kdyz se JSON prepocita, je potreba obsah v reportu vymenit taky.

Spousteni z korene repozitare (bezi zhruba pul druhe minuty):
    uv run python cviceni/C1/kod/predpocet_radioterapie.py
"""

import json
import time
from pathlib import Path

import numpy as np
import cvxpy as cp

PREDPIS = 60.0        # predepsana davka do nadoru [Gy]
TOLERANCE = 1.15      # horni mez v nadoru = 1.15 * predpis
PRAH_ZDRAVA = 45.0    # klinicky prah: podil zdrave tkane nad touhle davkou [Gy]
VAHY = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
        300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
BODU_DVH = 60         # kolik bodu na jednu krivku dose-volume histogramu
VAHA_OBRAZEK = 3.0    # ktery plan se kresli jako "optimalizovany" (jako v reportu)

# Svazku je zamerne malo a jsou siroke, aby byly na obrazku rozeznatelne jeden po
# druhem. Roztec poloh je mensi nez sirka svazku, takze sousedi se prekryvaji a
# pole je spojite; uzsi a hustsi mrizka by michu setrila o par Gy lip, ale na
# obrazku by z ni byla nerozlisitelna sedivka.
POLOH = 12            # pricnych poloh svazku v jednom uhlu
SIGMA = 2.6           # sirka svazku (smerodatna odchylka gaussovskeho profilu) [voxel]
ROZSAH = 18.0         # krajni polohy svazku, +-ROZSAH kolem osy [voxel]

# --- geometrie rezu a matice davek: presna kopie z ukazka_radioterapie.py ---
N = 64
yy, xx = np.mgrid[0:N, 0:N]
cx = cy = (N - 1) / 2
telo = np.hypot(xx - cx, yy - cy) < 30
nador = np.hypot(xx - cx - 6, yy - cy + 4) < 7
micha = np.hypot(xx - cx + 8, yy - cy - 6) < 4
zdrava = telo & ~nador & ~micha

UHLY = [0, 45, 90, 135, 180, 225, 270, 315]
POSUNY = np.linspace(-ROZSAH, ROZSAH, POLOH)
R_TELO = 30.0         # polomer rezu telem [voxely]
R_HLAVICE = 44.0      # vzdalenost banku hlavic od izocentra, vne tela

sloupce, svazky = [], []
for uhel in np.deg2rad(UHLY):
    pricne = (xx - cx) * np.cos(uhel) + (yy - cy) * np.sin(uhel)    # napric svazkem
    hloubka = -(xx - cx) * np.sin(uhel) + (yy - cy) * np.cos(uhel)  # podel svazku
    for posun in POSUNY:
        profil = np.exp(-0.5 * ((pricne - posun) / SIGMA) ** 2)     # sirka svazku
        utlum = np.exp(-0.02 * (hloubka + 32))                      # utlum s hloubkou
        sloupce.append((profil * utlum * telo).ravel())
        # osa svazku je primka {pricne == posun}; svazek po ni jde ve smeru rustu
        # hloubky, tedy smerem d. Body pro kresleni: hlavice -> vstup -> vystup.
        kolmo = np.array([np.cos(uhel), np.sin(uhel)])
        smer = np.array([-np.sin(uhel), np.cos(uhel)])
        stred = np.array([cx, cy]) + posun * kolmo
        pul = np.sqrt(R_TELO ** 2 - posun ** 2)     # pulka tetivy telem
        # Vsechny hlavice jednoho uhlu lezi na PRIMCE kolme na svazek ve vzdalenosti
        # R_HLAVICE od izocentra, ne na kruznici -- svazky jsou rovnobezne, takze
        # vychazeji ze spolecne roviny. Osm takovych banku tvori osmiuhelnik.
        hlavice = stred - R_HLAVICE * smer
        svazky.append({
            "u": UHLY.index(int(round(np.degrees(uhel)))),
            "hx": hlavice[0], "hy": hlavice[1],
            "x1": (stred - pul * smer)[0], "y1": (stred - pul * smer)[1],
            "x2": (stred + pul * smer)[0], "y2": (stred + pul * smer)[1],
        })
D = np.array(sloupce).T
assert len(svazky) == D.shape[1] == len(UHLY) * POLOH
davka_nador, davka_micha, davka_zdrava = D[nador.ravel()], D[micha.ravel()], D[zdrava.ravel()]

# podil objemu, ke kteremu se odecita davka; krivka DVH je tim popsana uplne
OBJEM = np.linspace(0.0, 100.0, BODU_DVH)


def cifry(pole, n=3):
    """Zaokrouhli na n platnych cifer a vrati cisty seznam floatu (kvuli velikosti JSONu)."""
    return [float(f"{v:.{n}g}") for v in np.asarray(pole, dtype=float)]


def dvh(davka):
    """Davka odectena na mrizce podilu objemu -- inverzni funkce k dose-volume histogramu."""
    return cifry(np.percentile(davka, 100.0 - OBJEM))


def shrnuti(plan, vaha=None):
    """Klicova cisla jednoho planu plus jeho tri DVH krivky."""
    dn, dm, dz = plan[nador.ravel()], plan[micha.ravel()], plan[zdrava.ravel()]
    zaznam = {
        "micha_max": round(float(dm.max()), 3),
        "micha_stred": round(float(dm.mean()), 3),
        "nador_min": round(float(dn.min()), 3),
        "nador_max": round(float(dn.max()), 3),
        "nador_pokryti": round(float(100.0 * (dn >= PREDPIS - 1e-6).mean()), 2),
        "zdrava_stred": round(float(dz.mean()), 3),
        "zdrava_nad_prah": round(float(100.0 * (dz > PRAH_ZDRAVA).mean()), 3),
        "dvh": {"nador": dvh(dn), "micha": dvh(dm), "zdrava": dvh(dz)},
    }
    if vaha is not None:
        zaznam["vaha"] = vaha
    return zaznam


# --- rucne sestavene plany, ktere si student zkusi drive, nez neco optimalizuje ---
# Oba jsou "rozumne" a oba selzou, kazdy jinak: mireni na nador rozbije homogenitu
# v nadoru, dva kolme smery sice srazi michu pod optimum, ale za cenu 97 Gy v nadoru.
# Intenzita je u obou stejna pro vsechny zapnute svazky a skaluje se tak, aby nador
# dostal predepsanych 60 Gy v nejchladnejsim voxelu -- presne to, co by clovek udelal.
STRED_NADOR = np.array([6.0, -4.0])   # stred nadoru vuci izocentru [voxel]
R_NADOR = 7                           # polomer nadoru [voxel]

trefa_nadoru, uhel_svazku = [], []
for uhel in np.deg2rad(UHLY):
    pricne_nador = STRED_NADOR[0] * np.cos(uhel) + STRED_NADOR[1] * np.sin(uhel)
    for posun in POSUNY:
        trefa_nadoru.append(abs(posun - pricne_nador) < R_NADOR)   # osa svazku jde nadorem
        uhel_svazku.append(int(round(np.degrees(uhel))))
trefa_nadoru = np.array(trefa_nadoru)
uhel_svazku = np.array(uhel_svazku)

RUCNI = [
    ("mirim", "Mířím na nádor",
     "všechny svazky, jejichž osa protne nádor, stejně silné",
     trefa_nadoru),
    ("kolme", "Jen dva kolmé směry",
     "totéž, ale pouze z 0° a 90°",
     trefa_nadoru & np.isin(uhel_svazku, [0, 90])),
]


def rucni_plan(maska):
    """Zapnute svazky stejne silne, naskalovane na predpis v nejchladnejsim voxelu nadoru."""
    w = maska.astype(float)
    assert w.sum() > 0, "prazdny rucni plan"
    return w * (PREDPIS / (D @ w)[nador.ravel()].min())


# --- srovnavaci plan: vsechny svazky stejne, naskalovane na predpis v nadoru ---
w_rovno = np.ones(D.shape[1])
plan_rovno = D @ w_rovno
skala = PREDPIS / plan_rovno[nador.ravel()].min()
plan_rovno, w_rovno = plan_rovno * skala, w_rovno * skala

# --- rodina optimalizovanych planu; vaha je Parameter, uloha se sestavuje jednou ---
w = cp.Variable(D.shape[1], nonneg=True)
vaha = cp.Parameter(nonneg=True)
uloha = cp.Problem(
    cp.Minimize(vaha * cp.sum_squares(davka_micha @ w) + cp.sum_squares(davka_zdrava @ w)),
    [davka_nador @ w >= PREDPIS, davka_nador @ w <= TOLERANCE * PREDPIS],
)

plany, w_opt, zacatek = [], None, time.perf_counter()
print(f"{'váha míchy':>12}{'status':>10}{'čas [s]':>9}{'mícha max':>11}"
      f"{'nádor min':>11}{'nádor max':>11}{'zdravá >45 Gy':>15}")
for v in VAHY:
    vaha.value = v
    cas = time.perf_counter()
    uloha.solve(solver=cp.CLARABEL)
    cas = time.perf_counter() - cas
    assert uloha.status == cp.OPTIMAL, f"váha {v}: solver skončil jako {uloha.status}"
    plan = D @ w.value
    zaznam = shrnuti(plan, vaha=v)
    # kontrola dosazenim zpet: predpis v nadoru musi platit u kazde vahy
    assert zaznam["nador_min"] >= PREDPIS - 1e-3, f"váha {v}: nádor nedostal předpis"
    assert zaznam["nador_max"] <= TOLERANCE * PREDPIS + 1e-3, f"váha {v}: překročena horní mez"
    assert w.value.min() >= -1e-8, f"váha {v}: záporná intenzita svazku"
    plany.append(zaznam)
    if v == VAHA_OBRAZEK:
        w_opt = w.value.copy()
    print(f"{v:12g}{uloha.status:>10}{cas:9.1f}{zaznam['micha_max']:11.2f}"
          f"{zaznam['nador_min']:11.2f}{zaznam['nador_max']:11.2f}"
          f"{zaznam['zdrava_nad_prah']:15.2f}")

# fronta musi byt monotonni v tom, co se skutecne minimalizuje: vyssi vaha = nizsi micha
for a, b in zip(plany, plany[1:]):
    assert b["micha_max"] <= a["micha_max"] + 1e-6, "fronta není monotónní v dávce do míchy"

assert w_opt is not None, f"váha {VAHA_OBRAZEK} není v mřížce VAHY"
for s_ in svazky:                       # souradnice staci na desetinu voxelu
    for k in ("hx", "hy", "x1", "y1", "x2", "y2"):
        s_[k] = round(float(s_[k]), 1)

rucni = []
print(f"\n{'ruční plán':<22}{'svazků':>8}{'nádor min':>11}{'nádor max':>11}"
      f"{'mícha max':>11}{'zdravá stř':>12}")
for klic, nazev, popis, maska in RUCNI:
    w_r = rucni_plan(maska)
    z = shrnuti(D @ w_r)
    # tyhle plany predpis do nadoru splni jen zdola; horni mez naopak porusi, coz je pointa
    assert z["nador_min"] >= PREDPIS - 1e-3, f"{klic}: nádor nedostal předpis"
    rucni.append({"klic": klic, "nazev": nazev, "popis": popis,
                  "w": cifry(w_r), "cisla": z})
    print(f"{nazev:<22}{int(maska.sum()):>8}{z['nador_min']:>11.1f}{z['nador_max']:>11.1f}"
          f"{z['micha_max']:>11.1f}{z['zdrava_stred']:>12.1f}")

data = {
    "predpis": PREDPIS,
    "tolerance": TOLERANCE,
    "prah_zdrava": PRAH_ZDRAVA,
    "objem_procenta": cifry(OBJEM, 4),
    "rovnomerne": shrnuti(plan_rovno),
    "plany": plany,
    "geometrie": {
        "uhly": UHLY, "poloh": len(POSUNY), "sigma": SIGMA, "rozsah": ROZSAH,
        "r_telo": R_TELO, "r_hlavice": R_HLAVICE,
        "cx": cx, "cy": cy,
        "nador": {"x": cx + 6, "y": cy - 4, "r": 7},
        "micha": {"x": cx - 8, "y": cy + 6, "r": 4},
        "svazky": svazky,
    },
    "intenzity": {"vaha": VAHA_OBRAZEK, "opt": cifry(w_opt), "rovno": cifry(w_rovno),
                  "rucni": rucni},
}
cesta = Path(__file__).resolve().parents[1] / "data" / "radioterapie.json"
cesta.parent.mkdir(parents=True, exist_ok=True)
cesta.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"\ncelkem {time.perf_counter() - zacatek:.0f} s, "
      f"uloženo {cesta} ({cesta.stat().st_size / 1024:.1f} kB)")
