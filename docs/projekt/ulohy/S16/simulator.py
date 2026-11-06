"""Urgentní příjem na jeden týden – diskrétně-událostní simulace (úloha S16).

Den má šest čtyřhodinových bloků (0–4, 4–8, 8–12, 12–16, 16–20, 20–24 h).
Rozpis říká, kolik lékařů a kolik sester je v každém bloku; stejný rozpis
platí všech sedm dní v týdnu.

* Příchody: nehomogenní Poissonův proces – klidná noc, dopolední a večerní
  špička, vynásobeno koeficientem dne v týdnu. Asi 0,7× za týden přijede
  hromadné neštěstí: 6–12 vážně zraněných (červení a žlutí) během půl hodiny.
* Triáž: červení (0), žlutí (1), zelení (2), limit čekání 10 / 60 / 240 min.
* Ošetření: (1) vyšetření – lékař a sestra současně, (2) péče – jen sestra.
  Doby jsou lognormální se střední hodnotou podle kategorie.
* Čekání = od příchodu do začátku vyšetření. Zelený, který čeká déle než
  jeho (náhodná) trpělivost, odejde neošetřen.
* Přednost: červené vyšetření > péče > žluté > zelené vyšetření, v rámci
  kategorie podle příchodu. Rozpracovaný úkon se při změně bloku dodělá.
* Fast-track (fast_track=True): čeká-li zelený a je volná sestra, ale ne
  lékař, jedna sestra (nejvýš jedna naráz) ho ošetří sama, bez lékaře.

Kvalita rozpisu (minimalizace): vážený podíl pacientů týdne, kteří čekali
déle, než smí, nebo odešli neošetřeni, s vahami 5 / 2 / 1:
    cil = sum_k v_k * (nad_limitem_k + odesli_k) / sum_k v_k * pocet_k

Rozhraní
--------
simuluj(rozpis, seed, instance="hlavni", fast_track=False) -> dict
    rozpis     – celá čísla tvaru (2, 6): řádek 0 lékaři, řádek 1 sestry;
    seed       – který týden; stejný seed = stejní pacienti se stejnými
                 dobami a trpělivostí, ať je rozpis jakýkoli;
    instance   – "hlavni" nebo "vecer" (večerní špička později a silnější);
    vrací dict: 'cil', 'pocet', 'nad_limitem', 'odesli' (po kategoriích),
    'cekani', 'kategorie', 'prichod' (po pacientech, minuty).

Rozpočet 56 lidí-bloků za den simulátor nekontroluje – dodržet ho je věc
řešení. Blok bez lékaře nebo bez sestry (nebo se záporným počtem)
simulátor odmítne výjimkou ValueError: vyšetření by se v něm nedalo
začít.
"""
import heapq
from collections import deque

import numpy as np

TYDEN, BLOK = 7 * 24 * 60, 4 * 60          # minuty
LIMIT = np.array([10.0, 60.0, 240.0])
VAHA = np.array([5.0, 2.0, 1.0])

INSTANCE = {
    "hlavni": {
        # intenzita [pacient/h]: noc + gaussovské špičky [výška, hodina, šířka]
        "noc": 2.5, "spicky": [[10.0, 11.0, 2.5], [7.0, 19.0, 2.0]],
        "den_v_tydnu": [1.15, 1.05, 1.0, 1.0, 1.05, 0.9, 0.85],   # po … ne
        "podil_kategorii": [0.08, 0.37, 0.55],
        "vysetreni_min": [45.0, 25.0, 12.0],     # střední doba (lékař + sestra)
        "pece_min": [60.0, 30.0, 10.0],          # střední doba (sestra)
        "variacni_koef": 0.6,
        "fast_track_min": 20.0,                  # sestra sama, zelený pacient
        "trpelivost_min": 200.0,                 # střední trpělivost zelených
        "nestesti_za_tyden": 0.7,
    },
}
INSTANCE["vecer"] = dict(INSTANCE["hlavni"], spicky=[[8.0, 10.5, 2.5], [10.0, 21.0, 2.0]])


def intenzita(hodina, p):
    lam = p["noc"] + 0.0 * hodina
    for a, mu, s in p["spicky"]:
        for posun in (-24.0, 0.0, 24.0):          # špička přes půlnoc
            lam = lam + a * np.exp(-0.5 * ((hodina - mu - posun) / s) ** 2)
    return lam


def _lognorm(rng, stred, cv, n):
    s2 = np.log(1.0 + cv * cv)
    return rng.lognormal(np.log(stred) - 0.5 * s2, np.sqrt(s2), n)


def generuj_pacienty(seed, instance="hlavni"):
    """Pacienti jednoho týdne: časy, kategorie, doby, trpělivost."""
    p = INSTANCE[instance]
    rng = np.random.default_rng(seed)
    lam_max = intenzita(np.linspace(0, 24, 2401), p).max() * max(p["den_v_tydnu"])
    t = np.sort(rng.uniform(0, TYDEN, rng.poisson(lam_max * TYDEN / 60.0)))
    lam = intenzita((t % 1440) / 60.0, p) * np.array(p["den_v_tydnu"])[(t // 1440).astype(int)]
    t = t[rng.uniform(0, lam_max, t.size) < lam]           # zřeďování
    kat = rng.choice(3, size=t.size, p=p["podil_kategorii"])
    for t0 in rng.uniform(0, TYDEN - 30, rng.poisson(p["nestesti_za_tyden"])):
        k = rng.integers(6, 13)                             # hromadné neštěstí
        t = np.concatenate([t, t0 + rng.uniform(0, 30, k)])
        kat = np.concatenate([kat, np.where(rng.uniform(size=k) < 0.4, 0, 1)])
    poradi = np.argsort(t, kind="stable")
    t, kat = t[poradi], kat[poradi]
    cv, n = p["variacni_koef"], t.size
    vys = _lognorm(rng, 1.0, cv, n) * np.array(p["vysetreni_min"])[kat]
    pec = _lognorm(rng, 1.0, cv, n) * np.array(p["pece_min"])[kat]
    ft = _lognorm(rng, p["fast_track_min"], cv, n)
    trp = rng.exponential(p["trpelivost_min"], n)
    return t, kat, vys, pec, ft, trp


def simuluj(rozpis, seed, instance="hlavni", fast_track=False):
    rozpis = np.asarray(rozpis)
    if rozpis.size != 12:
        raise ValueError(f"rozpis musí mít tvar (2, 6): lékaři a sestry v 6 blocích, má {rozpis.shape}")
    rozpis = rozpis.astype(int).reshape(2, 6)
    if np.any(rozpis < 1):
        raise ValueError("každý blok potřebuje aspoň 1 lékaře a 1 sestru, rozpis má "
                         f"lékaře {rozpis[0].tolist()} a sestry {rozpis[1].tolist()}")
    t, kat, vys, pec, ftd, trp = generuj_pacienty(seed, instance)
    n = t.size
    tl, kl, vys, pec, ftd = t.tolist(), kat.tolist(), vys.tolist(), pec.tolist(), ftd.tolist()
    odchod = (t + trp).tolist()            # kdy by zelený odešel
    tl.append(float("inf"))
    cekani = [float("nan")] * n
    odesel = [False] * n
    # halda událostí (čas, typ, x): typ 0 = konec úkonu (x = 3*pacient + fáze),
    # typ 1 = nový blok (x = číslo bloku); příchody jdou z utříděného seznamu
    ud = [(float(b * BLOK), 1, b % 6) for b in range(1, 7 * 6 * 3)]
    lekari, sestry = int(rozpis[0, 0]), int(rozpis[1, 0])
    vol_l, vol_s = lekari, sestry          # volná kapacita (může být záporná)
    fronta = [deque(), deque(), deque()]   # čekají na vyšetření
    pece = []                              # halda (kategorie, příchod, pacient)
    ft_bezi = False
    dalsi, zbyva = 0, n

    def start(ted):
        nonlocal vol_l, vol_s, ft_bezi, zbyva
        z = fronta[2]
        while True:
            while z and odchod[z[0]] <= ted:            # netrpěliví už odešli
                odesel[z.popleft()] = True
                zbyva -= 1
            lek = vol_l > 0 and vol_s > 0
            if fronta[0] and lek:
                i, faze = fronta[0].popleft(), 0
            elif pece and vol_s > 0:
                i, faze = heapq.heappop(pece)[2], 1
            elif fronta[1] and lek:
                i, faze = fronta[1].popleft(), 0
            elif z and lek:
                i, faze = z.popleft(), 0
            elif fast_track and z and vol_s > 0 and not ft_bezi:
                i, faze = z.popleft(), 2
            else:
                return
            if faze != 1:
                cekani[i] = ted - tl[i]
            if faze == 0:
                vol_l -= 1
                heapq.heappush(ud, (ted + vys[i], 0, 3 * i))
            elif faze == 1:
                heapq.heappush(ud, (ted + pec[i], 0, 3 * i + 1))
            else:
                ft_bezi = True
                heapq.heappush(ud, (ted + ftd[i], 0, 3 * i + 2))
            vol_s -= 1

    while zbyva > 0:
        if not ud:
            raise RuntimeError("simulace se zasekla: pacienti čekají a není kdo by je ošetřil")
        if tl[dalsi] < ud[0][0]:                        # příchod
            ted = tl[dalsi]
            fronta[kl[dalsi]].append(dalsi)
            dalsi += 1
        else:
            ted, typ, x = heapq.heappop(ud)
            if typ == 1:                                # nový blok
                nl, ns = int(rozpis[0, x]), int(rozpis[1, x])
                vol_l += nl - lekari; vol_s += ns - sestry
                lekari, sestry = nl, ns
            else:                                       # konec úkonu
                i, faze = divmod(x, 3)
                vol_s += 1
                if faze == 0:                           # vyšetření -> péče
                    vol_l += 1
                    heapq.heappush(pece, (kl[i], tl[i], i))
                else:
                    zbyva -= 1
                    ft_bezi = ft_bezi and faze != 2
        start(ted)

    cekani, odesel = np.array(cekani), np.array(odesel)
    pocet = np.bincount(kat, minlength=3)
    nad = np.bincount(kat, weights=cekani > LIMIT[kat], minlength=3).astype(int)
    ode = np.bincount(kat, weights=odesel, minlength=3).astype(int)
    cil = float((VAHA * (nad + ode)).sum() / (VAHA * pocet).sum())
    return {"cil": cil, "pocet": pocet, "nad_limitem": nad, "odesli": ode,
            "cekani": cekani, "kategorie": kat, "prichod": t}
