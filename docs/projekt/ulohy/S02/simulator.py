"""Detektor QRS podle Pana a Tompkinse se šesti nastavitelnými parametry.

Rozhraní
--------
Parametry detektoru jsou vektor šesti čísel (pořadí je pevné):

    p = (f_dolni, f_horni, okno, prah, refrakter_ms, adaptace)

    f_dolni      dolní mezní frekvence pásmové propusti [Hz]   meze 0.5 .. 20
    f_horni      horní mezní frekvence pásmové propusti [Hz]   meze 8 .. 60
                 (musí platit f_dolni < f_horni)
    okno         délka klouzavého průměru ve vzorcích           meze 5 .. 60
                 (zaokrouhlí se na celé číslo; při 250 Hz je 1 vzorek 4 ms)
    prah         poloha prahu mezi úrovní šumu a úrovní QRS     meze 0.05 .. 0.8
    refrakter_ms doba po detekci, kdy se další úder nehledá [ms] meze 100 .. 400
                 (zaokrouhlí se na celé vzorky)
    adaptace     rychlost, jakou se úrovně QRS a šumu přizpůsobují
                 novým špičkám (0 = vůbec, 1 = jen poslední špička) meze 0.01 .. 0.5

Funkce:

    detekuj(ekg, fs, p)          -> indexy detekovaných R vln (np.ndarray int)
    f1_zaznamu(det, r, fs)       -> (F1, TP, FP, FN), zásah = do ±50 ms od R
    vyhodnot(p, zaznamy)         -> dict s průměrným a nejhorším F1 přes záznamy
    nacti_sadu(adresar)          -> seznam záznamů {"ekg", "fs", "r", "extrasystola", "nazev"}
                                    relativní cesta se bere vůči složce simulátoru, holé
                                    jméno sady vůči data/: nacti_sadu("klid-trenink")

Výchozí parametry z literatury (Pan a Tompkins 1985) jsou VYCHOZI.
Simulátor je deterministický: stejné parametry a data dají vždy stejný výsledek.
Nepřípustné parametry (mimo meze, f_dolni >= f_horni) vyvolají ValueError.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

NAZVY = ("f_dolni", "f_horni", "okno", "prah", "refrakter_ms", "adaptace")
MEZE = np.array([[0.5, 20.0], [8.0, 60.0], [5, 60], [0.05, 0.8], [100, 400], [0.01, 0.5]])
VYCHOZI = np.array([5.0, 15.0, 37, 0.25, 200, 0.125])


def zkontroluj(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (6,) or not np.all(np.isfinite(p)):
        raise ValueError("p musí být 6 konečných čísel")
    if np.any(p < MEZE[:, 0]) or np.any(p > MEZE[:, 1]):
        raise ValueError(f"parametry mimo meze: {p}")
    if p[0] >= p[1]:
        raise ValueError("musí platit f_dolni < f_horni")
    return p


@lru_cache(maxsize=64)
def _pasmova_propust(f_lo, f_hi, fs):
    return butter(2, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")


def detekuj(ekg, fs, p):
    """Vrátí indexy vzorků, kde detektor hlásí R vlnu."""
    f_lo, f_hi, okno, prah, refr_ms, alfa = zkontroluj(p)
    okno = int(round(okno))
    refr = int(round(refr_ms * fs / 1000))
    # 1) pásmová propust (nulová fáze), 2) derivace, 3) kvadrát, 4) klouzavý průměr
    bp = sosfiltfilt(_pasmova_propust(f_lo, f_hi, fs), ekg)
    s = np.gradient(bp) ** 2
    c = np.concatenate([np.zeros(okno), np.cumsum(s)])
    mwi = (c[okno:] - c[:-okno]) / okno  # kauzální okno: průměr posledních `okno` vzorků
    kandidati, _ = find_peaks(mwi)
    hodnoty = mwi[kandidati].tolist()
    kandidati = kandidati.tolist()
    # 5) adaptivní práh: úroveň QRS (spk) a úroveň šumu (npk), učení z prvních 2 s
    n0 = int(2 * fs)
    spk = 0.25 * float(mwi[:n0].max())
    npk = 0.5 * float(mwi[:n0].mean())
    thr = npk + prah * (spk - npk)
    pruchod = okno + int(round(0.075 * fs))  # kde hledat R zpětně od špičky mwi

    det = []
    rr = []  # posledních 8 intervalů mezi údery (ve vzorcích)
    limit_rr = 0.0  # 1.66 x průměr rr
    posledni = -10**9
    ref, j_ref = posledni, 0  # od kdy (čas, index kandidáta) běží zpětné hledání
    for j in range(len(kandidati)):
        k, v = kandidati[j], hodnoty[j]
        # zpětné hledání: dlouho nic -> vezmi nejvyšší špičku nad polovičním prahem
        if rr and k - ref > limit_rr:
            okno_j = [i for i in range(j_ref, j) if kandidati[i] - posledni >= refr]
            ref, j_ref = k, j
            if okno_j:
                i_max = max(okno_j, key=lambda i: hodnoty[i])
                if hodnoty[i_max] > 0.5 * thr:
                    kk = kandidati[i_max]
                    rr = (rr + [kk - posledni])[-8:]
                    limit_rr = 1.66 * sum(rr) / len(rr)
                    det.append(kk)
                    posledni = kk
                    ref, j_ref = kk, i_max + 1
                    spk = 0.25 * hodnoty[i_max] + 0.75 * spk
                    thr = npk + prah * (spk - npk)
        if k - posledni < refr:
            continue
        if v > thr:
            if det:
                rr = (rr + [k - posledni])[-8:]
                limit_rr = 1.66 * sum(rr) / len(rr)
            det.append(k)
            posledni = k
            ref, j_ref = k, j + 1
            spk = alfa * v + (1 - alfa) * spk
        else:
            npk = alfa * v + (1 - alfa) * npk
        thr = npk + prah * (spk - npk)

    # poloha R: maximum |pásmově propuštěného signálu| těsně před špičkou mwi
    abs_bp = np.abs(bp)
    r = [max(0, k - pruchod) + int(np.argmax(abs_bp[max(0, k - pruchod): k + 1])) for k in det]
    return np.array(r, dtype=int)


def f1_zaznamu(det, r, fs, tolerance_ms=50.0):
    """Spáruje detekce se skutečnými R (každé R nejvýš jednou) a vrátí (F1, TP, FP, FN).

    Párování maximalizuje počet zásahů i pro libovolnou toleranci: detekce
    se procházejí podle času a každá dostane nejdřívější dosud nepoužitou
    vlnu R v okně ±tolerance (okna mají stejnou délku, proto je takové
    hladové párování největší možné).
    """
    tol = tolerance_ms * fs / 1000
    r = sorted(int(x) for x in r)
    tp, i = 0, 0
    for d in sorted(int(x) for x in det):
        while i < len(r) and r[i] < d - tol:  # tuhle R už žádná další detekce nezasáhne
            i += 1
        if i < len(r) and r[i] <= d + tol:
            tp += 1
            i += 1
    fp, fn = len(det) - tp, len(r) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    return f1, tp, fp, fn


def vyhodnot(p, zaznamy):
    """Spustí detektor na všech záznamech. Vrací průměrné F1, nejhorší F1 a součty."""
    f1, tp, fp, fn = [], 0, 0, 0
    for z in zaznamy:
        det = detekuj(z["ekg"], z["fs"], p)
        a, b, c, d = f1_zaznamu(det, z["r"], z["fs"])
        f1.append(a)
        tp, fp, fn = tp + b, fp + c, fn + d
    f1 = np.array(f1)
    return {"f1_prumer": float(f1.mean()), "f1_min": float(f1.min()),
            "f1": f1.tolist(), "tp": tp, "fp": fp, "fn": fn}


def nacti_sadu(adresar):
    """Načte všechny záznamy *.npz z adresáře (klíče ekg [mV], fs [Hz], r [indexy], extrasystola)."""
    adresar = Path(adresar)
    if not adresar.is_absolute():
        zde = Path(__file__).resolve().parent
        adresar = zde / adresar if (zde / adresar).is_dir() else zde / "data" / adresar
    if not adresar.is_dir():
        raise FileNotFoundError(f"sada {adresar} neexistuje")
    zaznamy = []
    for f in sorted(adresar.glob("*.npz")):
        d = np.load(f)
        zaznamy.append({"ekg": d["ekg"].astype(float), "fs": float(d["fs"]), "r": d["r"].astype(int),
                        "extrasystola": d["extrasystola"].astype(bool), "nazev": f.stem})
    return zaznamy


if __name__ == "__main__":
    import sys, time
    sada = nacti_sadu(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "klid-trenink")
    t = time.perf_counter()
    v = vyhodnot(VYCHOZI, sada)
    print(f"výchozí parametry: F1 průměr {v['f1_prumer']:.4f}, min {v['f1_min']:.4f}, "
          f"TP/FP/FN {v['tp']}/{v['fp']}/{v['fn']}  ({time.perf_counter() - t:.3f} s)")
