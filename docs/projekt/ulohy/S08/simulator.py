"""Fylogenetický strom ze zarovnaných sekvencí DNA: parsimonie a věrohodnost.

Proměnná je **topologie nezakořeněného binárního stromu** na n taxonech
(listech). Simulátor umí strom vyhodnotit dvěma způsoby a nic víc:

* ``parsimonie(strom)`` – nejmenší počet mutací, kterými se dají vysvětlit
  sekvence na listech (Fitchův algoritmus). Menší = lepší. Rychlé.
* ``log_verohodnost(strom)`` – log-věrohodnost zarovnání pod modelem
  Jukes–Cantor (JC69) při **nejlepších délkách větví** pro danou topologii.
  Délky větví si funkce najde sama (souřadnicové hledání po větvích)
  a uloží je do ``strom.delky``. Větší = lepší. Řádově dražší (desítky
  až stovky parsimonií; z uložených délek rychlejší než z náhodných –
  to šetří čas, ne rozpočet: cena je vždy CENA_VEROHODNOSTI).

Model JC69: všechny čtyři báze stejně časté, každá záměna stejně
pravděpodobná. Pro větev délky t (očekávaný počet záměn na pozici) je

    P(stejna baze) = 1/4 + 3/4 exp(-4 t / 3),
    P(konkretni jina baze) = 1/4 - 1/4 exp(-4 t / 3).

Věrohodnost stromu je součin přes pozice zarovnání; pro jednu pozici se
sčítá přes všechny možné báze ve vnitřních uzlech (Felsensteinův algoritmus).

Reprezentace stromu
-------------------
Listy jsou uzly 0..n-1 (index = pořadí taxonu v datech), vnitřní uzly
n..2n-3, každý má právě tři sousedy. ``strom.sousede[u]`` je seznam
sousedů uzlu u. Strom má 2n-3 hran. Pomocné operace:

    strom = nahodny_strom(n, rng)            # rovnoměrně náhodná topologie
    strom = nacti_newick(text, taxony)       # z formátu Newick
    text  = strom.newick(taxony)             # do formátu Newick
    hrany = strom.vnitrni_hrany()            # n-3 vnitřních hran (u, v)
    s2    = nni(strom, (u, v), varianta)     # záměna podstromů přes hranu, varianta 0/1
    s2    = spr(strom, (u, x), (a, b))       # odříznout podstrom x (visí na u), přiroubovat na hranu (a, b)
    d     = rf_vzdalenost(s1, s2)            # Robinsonova–Fouldsova vzdálenost
    s     = strom.rozdeleni()                # množina rozdělení listů (bitové masky)

``nni`` a ``spr`` vrací nový strom, vstup nemění. Data:

    data = nacti_data("data/hlavni.fasta")   # Zarovnani: .taxony, .n, .delka
    data.parsimonie(strom), data.log_verohodnost(strom)
    data.pocet_vyhodnoceni                   # slovník: kolikrát se volala která funkce
    data.spotreba                            # spotřeba rozpočtu v jednotkách (viz níže)

Relativní cesta v ``nacti_data`` se hledá nejdřív vůči složce, ve které
leží ``simulator.py`` (``data/hlavni.fasta`` tedy funguje z libovolného
adresáře), pak vůči aktuálnímu adresáři (``simulacni-uloha/data/hlavni.fasta``
z kořene repozitáře).

Rozpočet se počítá v jednotkách: jedna parsimonie stojí 1 jednotku,
jedna věrohodnost ``CENA_VEROHODNOSTI`` = 50 jednotek (zhruba poměr
jejich výpočetního času). ``data.spotreba`` je součet za všechna volání.

Vše je deterministické (žádný šum).
"""

from pathlib import Path

import numpy as np

BAZE = "ACGT"
CENA_VEROHODNOSTI = 50          # jednotek rozpočtu za jedno volání log_verohodnost
ZDE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- strom

class Strom:
    def __init__(self, n, sousede, delky=None):
        self.n = n
        self.sousede = sousede          # list of lists
        self.delky = delky              # {(min(u,v), max(u,v)): t} nebo None

    def kopie(self):
        return Strom(self.n, [list(s) for s in self.sousede],
                     dict(self.delky) if self.delky else None)

    def hrany(self):
        return [(u, v) for u in range(len(self.sousede)) for v in self.sousede[u] if u < v]

    def vnitrni_hrany(self):
        return [(u, v) for (u, v) in self.hrany() if u >= self.n and v >= self.n]

    def poradi(self, koren=0):
        """Uzly v pořadí od kořene (DFS) a rodiče; kořen je list 0."""
        rodic = [-1] * len(self.sousede)
        rodic[koren] = koren
        poradi, zasobnik = [], [koren]
        while zasobnik:
            u = zasobnik.pop()
            poradi.append(u)
            for v in self.sousede[u]:
                if rodic[v] == -1:
                    rodic[v] = u
                    zasobnik.append(v)
        return poradi, rodic

    def rozdeleni(self):
        """Netriviální rozdělení listů: bitová maska strany, kde NENÍ list 0."""
        poradi, rodic = self.poradi(0)
        maska = [0] * len(self.sousede)
        for u in reversed(poradi):
            if u < self.n:
                maska[u] = 1 << u
            for v in self.sousede[u]:
                if v != rodic[u]:
                    maska[u] |= maska[v]
        return {maska[u] for u in poradi[1:] if u >= self.n and rodic[u] != 0}

    def newick(self, taxony=None):
        jm = taxony or [f"t{i}" for i in range(self.n)]

        def text(u, p):
            d = ""
            if self.delky:
                d = f":{self.delky[(min(u, p), max(u, p))]:.5f}"
            if u < self.n:
                return jm[u] + d
            deti = [text(v, u) for v in self.sousede[u] if v != p]
            return "(" + ",".join(deti) + ")" + d

        c = self.sousede[0][0]
        deti = [text(v, c) for v in self.sousede[c]]
        return "(" + ",".join(deti) + ");"


def nahodny_strom(n, rng):
    """Náhodná topologie: listy se přidávají na rovnoměrně vybranou hranu."""
    sousede = [[] for _ in range(2 * n - 2)]
    poradi = rng.permutation(n)
    a, b, c = poradi[:3]
    m = n                                   # první vnitřní uzel
    for x in (a, b, c):
        sousede[m].append(x); sousede[x].append(m)
    hrany = [(a, m), (b, m), (c, m)]
    dalsi = n + 1
    for x in poradi[3:]:
        u, v = hrany[rng.integers(len(hrany))]
        w = dalsi; dalsi += 1
        _rozdel_hranu(sousede, u, v, w)
        sousede[w].append(x); sousede[x].append(w)
        hrany.remove((u, v))
        hrany += [(u, w), (w, v), (x, w)]
    return Strom(n, sousede)


def _rozdel_hranu(sousede, u, v, w):
    sousede[u][sousede[u].index(v)] = w
    sousede[v][sousede[v].index(u)] = w
    sousede[w] += [u, v]


def nni(strom, hrana, varianta=0):
    """Nearest neighbour interchange přes vnitřní hranu (u, v).

    u má kromě v sousedy a, b; v má kromě u sousedy c, d. Varianta 0
    prohodí b <-> c, varianta 1 prohodí b <-> d.
    """
    u, v = hrana
    s = strom.kopie(); s.delky = None
    a, b = [w for w in s.sousede[u] if w != v]
    c, d = [w for w in s.sousede[v] if w != u]
    x = c if varianta == 0 else d
    s.sousede[u][s.sousede[u].index(b)] = x
    s.sousede[x][s.sousede[x].index(v)] = u
    s.sousede[v][s.sousede[v].index(x)] = b
    s.sousede[b][s.sousede[b].index(u)] = v
    return s


def spr(strom, podstrom, cil):
    """Subtree prune and regraft.

    ``podstrom = (u, x)``: hrana, x je kořen odřezávaného podstromu a u je
    vnitřní uzel, na kterém visí. u se ze stromu vyjme (jeho dva další
    sousedé se spojí) a vloží se doprostřed hrany ``cil = (a, b)``, která
    nesmí ležet v odřezávaném podstromu ani se dotýkat u.
    """
    u, x = podstrom
    s = strom.kopie(); s.delky = None
    p, q = [w for w in s.sousede[u] if w != x]
    if u in cil:
        raise ValueError("cílová hrana se dotýká odřezávaného uzlu")
    s.sousede[p][s.sousede[p].index(u)] = q
    s.sousede[q][s.sousede[q].index(u)] = p
    s.sousede[u] = [x]
    a, b = cil
    if b not in s.sousede[a]:
        raise ValueError("cílová hrana ve stromu po odříznutí neexistuje")
    _rozdel_hranu(s.sousede, a, b, u)
    # kontrola: cílová hrana nesměla být v podstromu (pak by vznikl cyklus)
    poradi, _ = s.poradi(0)
    if len(poradi) != len(s.sousede):
        raise ValueError("cílová hrana leží v odřezávaném podstromu")
    return s


def rf_vzdalenost(s1, s2):
    """Počet rozdělení, která má jen jeden ze stromů (0 až 2(n-3))."""
    return len(s1.rozdeleni() ^ s2.rozdeleni())


def nacti_newick(text, taxony):
    """Načte nezakořeněný strom z Newicku (délky větví se uloží, jsou-li)."""
    idx = {t: i for i, t in enumerate(taxony)}
    n = len(taxony)
    sousede = [[] for _ in range(2 * n - 1)]
    delky = {}
    text = text.strip().rstrip(";")
    pos = 0
    dalsi = [n]

    def uzel():
        nonlocal pos
        if text[pos] == "(":
            pos += 1
            u = dalsi[0]; dalsi[0] += 1
            while True:
                v, t = uzel()
                sousede[u].append(v); sousede[v].append(u)
                if t is not None:
                    delky[(min(u, v), max(u, v))] = t
                if text[pos] == ",":
                    pos += 1
                    continue
                pos += 1          # ')'
                break
        else:
            z = pos
            while text[pos] not in ",():;":
                pos += 1
            u = idx[text[z:pos].strip()]
        t = None
        if pos < len(text) and text[pos] == ":":
            z = pos = pos + 1
            while pos < len(text) and text[pos] not in ",()":
                pos += 1
            t = float(text[z:pos])
        return u, t

    koren, _ = uzel()
    if len(sousede[koren]) == 2:           # zakořeněný zápis: kořen odstranit
        p, q = sousede[koren]
        sousede[p][sousede[p].index(koren)] = q
        sousede[q][sousede[q].index(koren)] = p
        k1, k2 = (min(p, koren), max(p, koren)), (min(q, koren), max(q, koren))
        if k1 in delky and k2 in delky:
            delky[(min(p, q), max(p, q))] = delky.pop(k1) + delky.pop(k2)
        sousede[koren] = []
    # přečíslovat vnitřní uzly na n..2n-3
    vnitrni = [u for u in range(n, len(sousede)) if sousede[u]]
    mapa = {u: u for u in range(n)} | {u: n + i for i, u in enumerate(vnitrni)}
    nove = [[] for _ in range(2 * n - 2)]
    for u in mapa:
        nove[mapa[u]] = [mapa[v] for v in sousede[u]]
    d2 = {(min(mapa[u], mapa[v]), max(mapa[u], mapa[v])): t for (u, v), t in delky.items()}
    return Strom(n, nove, d2 or None)


# ---------------------------------------------------------------- data

class Zarovnani:
    def __init__(self, taxony, sekvence):
        self.taxony = list(taxony)
        self.n = len(taxony)
        self.delka = len(sekvence[0])
        M = np.array([[BAZE.index(c) for c in s] for s in sekvence], dtype=np.int8)
        # stejné sloupce (vzory) se počítají jen jednou, s vahou
        vzory, vahy = np.unique(M.T, axis=0, return_counts=True)
        self.vzory = vzory.T                                 # (n, pocet_vzoru)
        self.vahy = vahy.astype(float)
        self.bity = (1 << self.vzory).astype(np.uint8)       # Fitch: množina bází
        self.tip = np.eye(4)[self.vzory].transpose(0, 2, 1).copy()    # (n, 4, vzory)
        self.pocet_vyhodnoceni = {"parsimonie": 0, "log_verohodnost": 0}

    @property
    def spotreba(self):
        """Spotřeba rozpočtu: parsimonie po 1, věrohodnost po CENA_VEROHODNOSTI."""
        p = self.pocet_vyhodnoceni
        return p["parsimonie"] + CENA_VEROHODNOSTI * p["log_verohodnost"]

    # -------------------------------------------- parsimonie (Fitch)
    def parsimonie(self, strom):
        self.pocet_vyhodnoceni["parsimonie"] += 1
        poradi, rodic = strom.poradi(0)
        mnoz = [None] * len(strom.sousede)
        skore = 0.0
        for u in reversed(poradi[1:]):
            if u < self.n:
                mnoz[u] = self.bity[u]
                continue
            a, b = [mnoz[v] for v in strom.sousede[u] if v != rodic[u]]
            prunik = a & b
            prazdny = prunik == 0
            skore += self.vahy[prazdny].sum()
            mnoz[u] = np.where(prazdny, a | b, prunik)
        c = poradi[1]                            # soused listu 0
        skore += self.vahy[(mnoz[c] & self.bity[0]) == 0].sum()
        return int(skore)

    # -------------------------------------------- věrohodnost (JC69)
    def log_verohodnost(self, strom, max_pruchodu=20, tol=1e-2):
        """Log-věrohodnost při délkách větví optimalizovaných pro tuto topologii.

        Délky se hledají souřadnicově: průchod stromem od kořene nastaví
        každou větev na maximum věrohodnosti při ostatních pevných (Newtonova
        metoda) a průchody se opakují, dokud se log-věrohodnost zlepšuje
        o víc než ``tol``. Má-li strom uložené délky (``strom.delky``), začne
        z nich (teplý start), jinak ze všech 0,1. Výsledné délky uloží do
        ``strom.delky``. S ``max_pruchodu=0`` délky nemění a jen přepočítá
        věrohodnost při uložených délkách (kontrola).
        """
        self.pocet_vyhodnoceni["log_verohodnost"] += 1
        poradi, rodic = strom.poradi(0)
        deti = {u: [v for v in strom.sousede[u] if v != rodic[u]] for u in poradi}
        uloz = strom.delky or {}
        # e[u] = exp(-4t/3) pro větev mezi u a jeho rodičem
        e = {u: np.exp(-4 / 3 * uloz.get((min(u, rodic[u]), max(u, rodic[u])), 0.1))
             for u in poradi[1:]}
        dolu = {}                                # věrohodnost podstromu pod uzlem
        for u in reversed(poradi[1:]):
            dolu[u] = self.tip[u] if u < self.n else _soucin(e, dolu, deti[u])
        nahoru = {poradi[1]: self.tip[0]}        # věrohodnost zbytku stromu nad větví

        def pruchod(u):
            e[u] = _nejlepsi_e(nahoru[u], dolu[u], self.vahy, e[u])
            if u < self.n:
                return
            horni = _pres(e[u], nahoru[u])
            for c in deti[u]:
                nahoru[c] = horni * _soucin(e, dolu, [v for v in deti[u] if v != c])
                pruchod(c)
            dolu[u] = _soucin(e, dolu, deti[u])

        logl = self._logl(e, dolu, poradi[1])
        for _ in range(max_pruchodu):
            pruchod(poradi[1])
            nova = self._logl(e, dolu, poradi[1])
            if nova - logl < tol:
                logl = max(logl, nova)
                break
            logl = nova
        strom.delky = {(min(u, rodic[u]), max(u, rodic[u])): -0.75 * np.log(e[u])
                       for u in poradi[1:]}
        return float(logl)

    def _logl(self, e, dolu, c):
        """Log-věrohodnost přes větev mezi listem 0 a jeho sousedem c."""
        return self.vahy @ np.log((self.tip[0] * _pres(e[c], dolu[c])).sum(0) / 4)


def _pres(e, x):
    """Přenos přes větev: P(t) @ x, kde P(t) = e I + (1 - e)/4 (samé jedničky)."""
    return e * x + (1 - e) / 4 * x.sum(0)


def _soucin(e, dolu, uzly):
    x = 1.0
    for v in uzly:
        x = x * _pres(e[v], dolu[v])
    return x


def _nejlepsi_e(x, y, w, e0, t_min=1e-6, t_max=5.0):
    """Nejlepší e = exp(-4t/3) jedné větve (x nad ní, y pod ní).

    Věrohodnost pozice je (c + e d)/4 s c = sum(x) sum(y)/4 a
    d = sum(x y) - c; log-věrohodnost je v e konkávní, takže stačí
    Newtonova metoda s ohraničením na [exp(-4 t_max/3), exp(-4 t_min/3)].
    """
    c = x.sum(0) * y.sum(0) / 4
    d = (x * y).sum(0) - c
    lo, hi = np.exp(-4 * t_max / 3), np.exp(-4 * t_min / 3)
    e = min(max(e0, lo), hi)
    for _ in range(10):
        q = d / (c + e * d)
        e2 = min(max(e + (w @ q) / (w @ (q * q)), lo), hi)
        if abs(e2 - e) < 1e-6 * e:
            return e2
        e = e2
    return e


def nacti_data(cesta="data/hlavni.fasta"):
    """Načte zarovnání z FASTA. Relativní cesta se hledá nejdřív vůči
    složce simulátoru (``data/hlavni.fasta``), pak vůči aktuálnímu
    adresáři (``simulacni-uloha/data/hlavni.fasta`` z kořene repozitáře)."""
    cesta = Path(cesta)
    if not cesta.is_absolute():
        kandidati = [ZDE / cesta, Path.cwd() / cesta]
        cesta = next((c for c in kandidati if c.exists()), None)
        if cesta is None:
            raise FileNotFoundError("soubor nenalezen ani vůči složce simulátoru, ani vůči "
                                    "aktuálnímu adresáři: " + ", ".join(map(str, kandidati)))
    taxony, sekvence = [], []
    for radek in open(cesta, encoding="utf-8"):
        radek = radek.strip()
        if radek.startswith(">"):
            taxony.append(radek[1:].strip()); sekvence.append("")
        elif radek:
            sekvence[-1] += radek.upper()
    return Zarovnani(taxony, sekvence)
