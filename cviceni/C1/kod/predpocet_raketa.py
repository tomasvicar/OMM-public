"""Predpocet rodiny reseni pro interaktivni hriste u ukazky s raketou.

Hriste v reportu necha uzivatele nakreslit vlastni prubeh tahu a uhlu a odletet
s nim sestup. Optimalni reseni se v prohlizeci pocitat neda (SOCP se 400+
promennymi), takze se predpocita tady: pro mrizku hodnot (Tmax, doba letu) se
ulozi optimalni profil tahu po krocich. Trajektorii si prohlizec dopocita sam
tymiz diskretizovanymi rovnicemi, ktere jsou v modelu - proto staci ulozit
vektor tahu, ne polohy.

Spusteni z korene repozitare:

    uv run python cviceni/C1/kod/predpocet_raketa.py

Skript zapisuje dve veci:
  1. cviceni/C1/data/raketa.json          ... samotna data
  2. cviceni/C1/reporty/raketa.html       ... tentyz JSON vlozi mezi znacky
     <script id="raketa-data" ...> ... </script>, aby report fungoval offline
     a dal se zabalit do jednoho souboru.

Model je totozny s kod/ukazka_raketa.py, jen s promennym Tmax a poctem kroku.
Nepripustne kombinace se ukladaji jako null - `infeasible` je u teto ukazky
soucast sdeleni, ne chyba.
"""

import json
import pathlib
import re

import cvxpy as cp
import numpy as np

KOREN = pathlib.Path(__file__).resolve().parents[3]
JSON_SOUBOR = KOREN / "cviceni/C1/data/raketa.json"
REPORT = KOREN / "cviceni/C1/reporty/raketa.html"

dt = 1.0                                          # delka kroku [s]
g, m = 9.81, 2000.0                               # tihove zrychleni, hmotnost
gv = np.array([0.0, g])
Tmin = 5000.0                                     # motor nejde vypnout
p0 = np.array([1200.0, 1500.0])
v0 = np.array([-60.0, -80.0])
sklon = 0.5                                       # pristavaci kuzel: y >= sklon*|x|

TAHY = [20000, 22000, 24000, 25000, 26000, 28000, 30000, 33000, 36000, 40000]
DOBY = [25, 30, 35, 40, 45, 50, 60, 70, 80]
VYCHOZI_TAH, VYCHOZI_DOBA = TAHY.index(30000), DOBY.index(60)


def vyres(N, Tmax):
    """Minimalne palivova uloha pro N kroku a horni mez tahu Tmax."""
    p = cp.Variable((N + 1, 2))
    v = cp.Variable((N + 1, 2))
    T = cp.Variable((N, 2))
    sigma = cp.Variable(N)                        # lossless convexification

    omezeni = [
        p[0] == p0, v[0] == v0,
        p[N] == 0, v[N] == 0,
        p[:, 1] >= 0,
        p[:, 1] >= sklon * cp.abs(p[:, 0]),
        cp.norm(T, axis=1) <= sigma,
        sigma >= Tmin, sigma <= Tmax,
    ]
    for k in range(N):
        omezeni += [
            v[k + 1] == v[k] + dt * (T[k] / m - gv),
            p[k + 1] == p[k] + dt * (v[k] + v[k + 1]) / 2,
        ]
    uloha = cp.Problem(cp.Minimize(cp.sum(sigma) * dt), omezeni)
    uloha.solve()
    return uloha, T.value


def main():
    reseni, pocet_ok = [], 0
    for Tmax in TAHY:
        radek = []
        for N in DOBY:
            uloha, T = vyres(N, float(Tmax))
            if uloha.status != cp.OPTIMAL:
                radek.append(None)
                print(f"Tmax {Tmax / 1000:4.0f} kN, {N:2d} s -> {uloha.status}")
                continue
            pocet_ok += 1
            velikost = np.linalg.norm(T, axis=1)
            radek.append({
                "impuls": round(float(uloha.value)),      # [N*s]
                "tahMin": round(float(velikost.min())),
                "tahMax": round(float(velikost.max())),
                # vektor tahu po krocich, zaokrouhleny na newtony: [Tx0,Ty0,Tx1,...]
                "T": [round(float(x)) for x in T.reshape(-1)],
            })
            print(f"Tmax {Tmax / 1000:4.0f} kN, {N:2d} s -> optimal, "
                  f"{uloha.value / 1e6:.3f} MN*s")
        reseni.append(radek)

    data = {
        "dt": dt, "g": g, "m": m, "Tmin": Tmin, "sklon": sklon,
        "p0": list(p0), "v0": list(v0),
        "tahy": TAHY, "doby": DOBY,
        "vychozi": [VYCHOZI_TAH, VYCHOZI_DOBA],
        "reseni": reseni,
    }
    text = json.dumps(data, separators=(",", ":"))
    JSON_SOUBOR.parent.mkdir(parents=True, exist_ok=True)
    JSON_SOUBOR.write_text(text + "\n", encoding="utf-8")
    print(f"\n{pocet_ok} z {len(TAHY) * len(DOBY)} kombinaci ma reseni")
    print(f"{JSON_SOUBOR} zapsan, {len(text) / 1024:.1f} kB")

    # tentyz JSON vlozit do reportu, aby hriste fungovalo bez site i ze souboru
    html = REPORT.read_text(encoding="utf-8")
    znacka = re.compile(
        r'(<script id="raketa-data" type="application/json">)(.*?)(</script>)',
        re.S)
    if not znacka.search(html):
        print(f"POZOR: {REPORT} nema blok <script id=\"raketa-data\">, nevkladam")
        return
    REPORT.write_text(znacka.sub(lambda t: t.group(1) + text + t.group(3), html),
                      encoding="utf-8")
    print(f"{REPORT} aktualizovan")


if __name__ == "__main__":
    main()
