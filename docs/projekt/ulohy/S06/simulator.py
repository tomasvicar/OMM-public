"""Simulovaná enzymatická reakce na destičkách (úloha S06).

Recept jsou čtyři čísla v tomto pořadí (meze viz ``MEZE``):

    pH        5 až 9
    teplota   20 až 70 °C
    substrat  0,5 až 50 mM (počáteční koncentrace substrátu)
    doba      5 až 120 min (doba inkubace před zastavením reakce)

Výstupem je množství produktu v µM na konci inkubace (čím víc, tím lépe).

Rozhraní
--------
``lab = Laborator(instance=0, seed_sumu=0)``
    Laboratoř s danou instancí enzymu (instance = jiný enzym, 0 je hlavní)
    a seedem šumu měření. Stejné seedy a stejná posloupnost měření dají
    vždy stejná čísla.

``lab.zmer(recepty)``
    Jedna destička: pole tvaru (n, 4), n = 1 až 8 receptů (jeden recept
    smí být i 1D pole délky 4). Vrací n naměřených hodnot. Každé měření
    stojí jednu jamku z rozpočtu ``lab.rozpocet`` (40); po vyčerpání
    vyhodí ``RuntimeError``. Recept mimo meze vyhodí ``ValueError``.
    Šum: každá jamka má vlastní náhodnou chybu (asi 10 %) a celá destička
    sdílí náhodný faktor (asi 8 %; jiný den, jiná várka činidel).

``lab.zaznam``
    Seznam všech měření: (číslo destičky, recept, naměřená hodnota).

``pravy_vytezek(recepty, instance=0)``
    Skutečné (bezšumové) množství produktu. V laboratoři ho nikdo nezná,
    slouží **jen k vyhodnocení** strategie po skončení pokusu.

Model (pro zvědavé, k optimalizaci ho nepotřebujete): zvonovitá
závislost rychlosti na pH, Michaelis–Mentenová se substrátovou inhibicí,
Arrheniova teplotní závislost rychlosti a tepelná inaktivace enzymu
prvního řádu. Produkt se navíc pomalu samovolně rozkládá (rychleji
v zásaditém prostředí a za tepla).
"""

from functools import lru_cache

import numpy as np

NAZVY = ("pH", "teplota", "substrat", "doba")
MEZE = np.array([[5.0, 9.0], [20.0, 70.0], [0.5, 50.0], [5.0, 120.0]])
ROZPOCET = 40
VELIKOST_DESTICKY = 8

R = 8.314e-3  # kJ/(mol K)
T_REF = 310.15  # 37 °C


@lru_cache(maxsize=None)
def _parametry(instance):
    p = dict(
        pK1=5.9, pK2=7.5,        # zvon pH
        Km=2.0, Ki=40.0,         # mM, substrát (Haldane)
        Ea=55.0,                 # kJ/mol, aktivační energie katalýzy
        kd=0.010, Ed=170.0,      # 1/min při 37 °C a kJ/mol, tepelná inaktivace
        kh=0.010, Eh=60.0, kh_pH=0.5,  # 1/min a kJ/mol, rozklad produktu
        skala=14.0,              # µM/min, maximální rychlost při 37 °C
        sum_jamka=0.10, sum_desticka=0.08,
    )
    if instance == 0:
        return p
    rng = np.random.default_rng(1000 + int(instance))
    nasob = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    p["pK1"] += rng.uniform(-0.3, 0.3)
    p["pK2"] += rng.uniform(-0.3, 0.3)
    for k, a, b in (("kd", 0.6, 1.6), ("Km", 0.5, 2.0), ("Ki", 0.5, 2.0),
                    ("kh", 0.7, 1.4), ("Ea", 0.92, 1.08), ("Ed", 0.92, 1.08)):
        p[k] *= nasob(a, b)
    p["sum_jamka"] = rng.uniform(0.08, 0.12)
    p["sum_desticka"] = rng.uniform(0.05, 0.10)
    return p


def _arrhenius(E, T):
    return np.exp(-E / R * (1.0 / T - 1.0 / T_REF))


def pravy_vytezek(recepty, instance=0):
    """Bezšumové množství produktu (µM) pro jeden recept nebo pole receptů."""
    x = np.atleast_2d(np.asarray(recepty, dtype=float))
    p = _parametry(int(instance))
    pH, tep, S, t = x.T
    T = tep + 273.15
    v0 = p["skala"] * _arrhenius(p["Ea"], T)                    # teplota
    v0 = v0 / (1.0 + 10.0 ** (p["pK1"] - pH) + 10.0 ** (pH - p["pK2"]))  # pH
    v0 = v0 * S / (p["Km"] + S + S**2 / p["Ki"])                 # substrát
    kd = p["kd"] * _arrhenius(p["Ed"], T)                        # inaktivace
    kh = p["kh"] * _arrhenius(p["Eh"], T) * 10.0 ** (p["kh_pH"] * (pH - 7.0))
    # dP/dt = v0 exp(-kd t) - kh P,  P(0) = 0, tedy
    # P(t) = v0 (exp(-kd t) - exp(-kh t)) / (kh - kd), stabilně zapsáno:
    z = (kh - kd) * t
    zlomek = np.where(np.abs(z) > 1e-8, np.expm1(z) / np.where(z == 0, 1.0, z), 1.0)
    produkt = v0 * t * np.exp(-kh * t) * zlomek
    return produkt if np.ndim(recepty) > 1 else float(produkt[0])


class Laborator:
    """Zašuměné měření po destičkách s omezeným rozpočtem jamek."""

    def __init__(self, instance=0, seed_sumu=0, rozpocet=ROZPOCET):
        self.instance = instance
        self.rozpocet = rozpocet
        self.zaznam = []
        self._p = _parametry(int(instance))
        self._rng = np.random.default_rng([int(instance), int(seed_sumu)])
        self._desticka = 0

    @property
    def zbyva(self):
        return self.rozpocet - len(self.zaznam)

    def zmer(self, recepty):
        x = np.atleast_2d(np.asarray(recepty, dtype=float))
        if x.shape[1] != 4 or not 1 <= len(x) <= VELIKOST_DESTICKY:
            raise ValueError("destička = 1 až 8 receptů po 4 číslech")
        if np.any(x < MEZE[:, 0]) or np.any(x > MEZE[:, 1]):
            raise ValueError("recept mimo meze MEZE")
        if len(x) > self.zbyva:
            raise RuntimeError(f"rozpočet vyčerpán (zbývá {self.zbyva} jamek)")
        self._desticka += 1
        f_desticky = np.exp(self._p["sum_desticka"] * self._rng.standard_normal())
        f_jamek = np.exp(self._p["sum_jamka"] * self._rng.standard_normal(len(x)))
        y = pravy_vytezek(x, self.instance) * f_desticky * f_jamek
        for r, v in zip(x, y):
            self.zaznam.append((self._desticka, r.copy(), float(v)))
        return y
