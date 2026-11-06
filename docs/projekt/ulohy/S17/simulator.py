"""Simulovaný bezdrátový přenos do implantátu (úloha S17).

Konstelace je M bodů v komplexní rovině (typicky M = 16, tedy 4 bity na
symbol). Zadává se jako pole tvaru (M, 2): řádek = jeden bod, sloupce
= složka I (reálná část) a Q (imaginární část). Smí to být i ploché pole
délky 2M v pořadí I0, Q0, I1, Q1, ...

Kanál: vysílač pošle bod -> (volitelně) zesilovač -> přičte se bílý
gaussovský šum -> přijímač rozhodne pro nejbližší bod přijímané
konstelace. Chybovost symbolů (SER) je podíl symbolů, které přijímač
určil špatně; odhaduje se Monte Carlem.

Rozhraní
--------
``chybovost(body, snr_db, zesilovac=False, n_symbolu=16000, seed=0)``
    Odhad SER (číslo 0 až 1, čím menší, tím lépe). Každý bod se vyšle
    stejně často (n_symbolu / M krát).
    * ``snr_db`` – poměr referenčního výkonu 1 ku výkonu šumu N0 v dB:
      šum má složky I a Q nezávislé, normální se směrodatnou odchylkou
      sqrt(N0 / 2), kde N0 = 10 ** (-snr_db / 10). Výkon konstelace se
      **nenormalizuje** – kolik výkonu body mají, tolik ho přijímač dostane.
    * ``zesilovac=False`` – lineární vysílač, přijímaný bod = vyslaný bod.
      ``zesilovac=True`` – vysílací zesilovač se saturací (viz
      ``zesilovac``); přijímač zná zkreslenou konstelaci (naučí se ji
      z úvodní sekvence) a rozhoduje podle ní.
    * ``seed`` – seed šumu. Stejný seed = stejné vzorky šumu pro všechna
      volání (společná náhodná čísla), výsledek je deterministický.
      ``seed=None`` = pokaždé čerstvý šum.

``zesilovac(body, p=P_RAPP)``
    Výstup zesilovače (Rappův model polovodičového zesilovače): fáze se
    nemění, amplituda A přejde na A / (1 + A**(2p)) ** (1 / (2p)).
    Pro malé A je to skoro lineární, výstup se nikdy nedostane nad
    saturační amplitudu 1 (výkon 1). Čím větší p, tím ostřejší „koleno“.
    Budič před zesilovačem neumí amplitudu větší než ``A_BUDIC`` = 2;
    větší vstup se ořízne na 2 (fáze zůstane), takže výstup je nejvýš
    2 / 17 ** 0,25 = 0,985.

    Jiný zesilovač: ``chybovost(zesilovac(body, p=4), snr_db)``, tj. body
    zkreslit sami a volat ``chybovost`` se ``zesilovac=False``.

``vykon(body)``
    Dvojice (průměrný výkon, špičkový výkon) bodů, tj. průměr a maximum
    I**2 + Q**2.

``qam16()``
    Učebnicová 16-QAM (mřížka 4 x 4) normovaná na průměrný výkon 1.
"""

import numpy as np

P_RAPP = 2.0  # hladkost kolena zesilovače (typický polovodičový: 2 až 3)
A_BUDIC = 2.0  # největší amplituda, kterou budič zesilovače umí dodat


def _body(body):
    b = np.asarray(body, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 2)
    if b.ndim != 2 or b.shape[1] != 2 or len(b) < 2:
        raise ValueError("body musí mít tvar (M, 2) nebo (2M,)")
    if not np.all(np.isfinite(b)):
        raise ValueError("body obsahují nan nebo inf")
    return b


def zesilovac(body, p=P_RAPP):
    b = _body(body)
    a = np.hypot(b[:, 0], b[:, 1])
    a_vst = np.minimum(a, A_BUDIC)  # budič větší amplitudu nedá
    a_vyst = a_vst / (1.0 + a_vst ** (2 * p)) ** (1.0 / (2 * p))
    zisk = np.divide(a_vyst, a, out=np.ones_like(a), where=a > 0)
    return b * zisk[:, None]


def vykon(body):
    b = _body(body)
    e = (b ** 2).sum(axis=1)
    return float(e.mean()), float(e.max())


def qam16():
    u = np.array([-3.0, -1.0, 1.0, 3.0])
    b = np.array([(i, q) for q in u for i in u])
    return b / np.sqrt(10.0)


def _sum(m, n_na_bod, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, m, n_na_bod))


def chybovost(body, snr_db, zesilovac=False, n_symbolu=16000, seed=0):
    b = _body(body)
    if zesilovac:
        b = globals()["zesilovac"](b)
    m = len(b)
    n_na_bod = max(1, int(n_symbolu) // m)
    sigma = np.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)
    nx, ny = sigma * _sum(m, n_na_bod, seed)  # každý bod má vlastní vzorky šumu
    # vyslán bod k, přijato r = b_k + n; chyba, pokud je některý bod j
    # blíž než b_k:  |b_k - b_j + n|^2 < |n|^2
    #  <=>  |d_kj|^2 + 2 (dx_kj nx + dy_kj ny) < 0,  d_kj = b_k - b_j
    # Shodu vzdáleností přijímač rozhodne pro bod s nižším indexem (jako
    # np.argmin), tedy chyba je i rovnost s bodem j < k. U různých bodů
    # rovnost prakticky nenastane; u totožných bodů se z nich správně
    # určí jen jeden (16 totožných bodů: SER 15/16).
    dx = b[:, None, 0] - b[None, :, 0]
    dy = b[:, None, 1] - b[None, :, 1]
    d2 = dx ** 2 + dy ** 2
    np.fill_diagonal(d2, np.inf)  # bod sám se sebou se nesrovnává
    q = d2[:, :, None] + 2.0 * (dx[:, :, None] * nx[:, None, :] + dy[:, :, None] * ny[:, None, :])
    nizsi = np.tri(m, m, -1, dtype=bool)[:, :, None]  # [k, j] = True pro j < k
    blize = (q < 0) | ((q == 0) & nizsi)
    chyby = blize.any(axis=1).sum()
    return float(chyby) / (m * n_na_bod)
