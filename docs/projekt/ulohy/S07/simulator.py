"""Simulátor S07: jak dobře by diagnostický model fungoval s vybranými vyšetřeními.

Vstup je maska vyšetření: pole 80 nul a jedniček (1 = vyšetření se dělá).
Na vybraných sloupcích se natrénuje logistická regrese (s malou L2
regularizací, příznaky standardizované) a změří se její AUC.

Rozhraní
--------
    from simulator import Simulator
    sim = Simulator(instance=1)    # načte data/trenink_s1.csv, data/vysetreni_s1.csv
                                   # (výchozí adresář = data/ vedle simulator.py)

    sim.skore(maska, seed_cv=0)   # 5násobná stratifikovaná křížová validace
                                  # na 200 trénovacích pacientech -> průměrné AUC
                                  # seed_cv určuje rozdělení pacientů do 5 částí
    sim.skore(maska, seed_cv, pacienti=idx)
                                  # totéž jen na podmnožině trénovacích pacientů
                                  # (indexy řádků trenink_s1.csv, od 0)
    sim.auc_odlozena(maska, trenink=idx1, test=idx2)
                                  # natrénuje na pacientech idx1, AUC na idx2
                                  # (obojí z trénovací sady; počítá se do volání)
    sim.cena(maska)               # cena vybraných vyšetření na pacienta (Kč)
    sim.pocet_volani              # kolikrát bylo voláno skore() a auc_odlozena()
                                  # dohromady (auc_novi_pacienti se nepočítá)

    sim.auc_novi_pacienti(maska)  # natrénuje na všech 200 pacientech a změří AUC
                                  # na 2000 nových pacientech (novi_pacienti_s1.csv).
                                  # JEN PRO ZÁVĚREČNÉ VYHODNOCENÍ, ne pro výběr.

Prázdná maska dá AUC 0,5 (model bez vyšetření nerozliší nikoho).
Výsledek je pro danou masku a seed_cv deterministický; jedno volání skore()
trvá jednotky milisekund.
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ALFA = 1.0     # síla L2 regularizace logistické regrese
K_CV = 5       # počet částí křížové validace


def auc(y, skore):
    """AUC = pravděpodobnost, že náhodný nemocný má vyšší skóre než náhodný zdravý."""
    r = rankdata(skore)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logisticka_regrese(X, y, alfa=ALFA, iteraci=25):
    """Newtonova metoda pro L2-regularizovanou logistickou regresi.

    Vrací (stred, meritko, w, b); predikce = ((X - stred) / meritko) @ w + b.
    """
    stred = X.mean(axis=0)
    meritko = X.std(axis=0) + 1e-12
    Z = (X - stred) / meritko
    n, d = Z.shape
    A = np.column_stack([Z, np.ones(n)])
    theta = np.zeros(d + 1)
    R = alfa * np.eye(d + 1)
    R[d, d] = 0.0                     # absolutní člen se neregularizuje
    for _ in range(iteraci):
        p = 1 / (1 + np.exp(-(A @ theta)))
        grad = A.T @ (p - y) + R @ theta
        H = (A * (p * (1 - p))[:, None]).T @ A + R
        krok = np.linalg.solve(H, grad)
        theta -= krok
        if np.abs(krok).max() < 1e-8:
            break
    return stred, meritko, theta[:d], theta[d]


def predikce(model, X):
    stred, meritko, w, b = model
    return ((X - stred) / meritko) @ w + b


def _nacti_csv(soubor):
    with open(soubor) as f:
        hlavicka = f.readline().strip().split(",")
    M = np.loadtxt(soubor, delimiter=",", skiprows=1)
    return hlavicka, M


class Simulator:
    def __init__(self, adresar=None, instance=1):
        adresar = Path(__file__).resolve().parent / "data" if adresar is None else Path(adresar)
        hlavicka, M = _nacti_csv(adresar / f"trenink_s{instance}.csv")
        self.nazvy = hlavicka[:-1]
        self.X, self.y = M[:, :-1], M[:, -1].astype(int)
        self.p = self.X.shape[1]
        c = np.genfromtxt(adresar / f"vysetreni_s{instance}.csv", delimiter=",",
                          skip_header=1, usecols=1)
        self.ceny = np.atleast_1d(c)
        self._soubor_novi = adresar / f"novi_pacienti_s{instance}.csv"
        self._novi = None
        self.pocet_volani = 0

    def _maska(self, maska):
        m = np.asarray(maska).astype(bool).ravel()
        if m.size != self.p:
            raise ValueError(f"maska musí mít {self.p} prvků")
        return m

    def rozdeleni(self, seed_cv, y=None):
        """Stratifikované rozdělení pacientů do K_CV částí (číslo části 0..K_CV-1)."""
        y = self.y if y is None else y
        rng = np.random.default_rng(seed_cv)
        cast = np.empty(len(y), dtype=int)
        for trida in (0, 1):
            idx = rng.permutation(np.flatnonzero(y == trida))
            cast[idx] = np.arange(len(idx)) % K_CV
        return cast

    def skore(self, maska, seed_cv=0, pacienti=None):
        """Křížově validované AUC (průměr přes K_CV částí)."""
        self.pocet_volani += 1
        m = self._maska(maska)
        if not m.any():
            return 0.5
        X, y = self.X[:, m], self.y
        if pacienti is not None:
            X, y = X[pacienti], y[pacienti]
        cast = self.rozdeleni(seed_cv, y)
        hodnoty = []
        for k in range(K_CV):
            tr, te = cast != k, cast == k
            model = logisticka_regrese(X[tr], y[tr])
            hodnoty.append(auc(y[te], predikce(model, X[te])))
        return float(np.mean(hodnoty))

    def auc_odlozena(self, maska, trenink, test):
        """Natrénuje na pacientech `trenink`, vrátí AUC na pacientech `test`."""
        self.pocet_volani += 1
        m = self._maska(maska)
        if not m.any():
            return 0.5
        model = logisticka_regrese(self.X[np.ix_(trenink, m)], self.y[trenink])
        return float(auc(self.y[test], predikce(model, self.X[np.ix_(test, m)])))

    def cena(self, maska):
        return float(self.ceny[self._maska(maska)].sum())

    def auc_novi_pacienti(self, maska):
        """AUC na 2000 nových pacientech – jen pro závěrečné vyhodnocení."""
        m = self._maska(maska)
        if not m.any():
            return 0.5
        if self._novi is None:
            _, M = _nacti_csv(self._soubor_novi)
            self._novi = (M[:, :-1], M[:, -1].astype(int))
        Xn, yn = self._novi
        model = logisticka_regrese(self.X[:, m], self.y)
        return float(auc(yn, predikce(model, Xn[:, m])))
