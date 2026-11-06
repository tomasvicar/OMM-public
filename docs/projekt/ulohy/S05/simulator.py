"""Tetris pro ladění vah počítačového hráče (simulační úloha S05).

Hráč u každého dílku vyzkouší všechna umístění (otočení × sloupec; dílek
padá svisle dolů, bez posouvání pod převisy), u každého spočítá osm rysů
desky po dopadu a smazání plných řad a vybere umístění s největším skóre
w · rysy. Dopředu se nedívá – zná jen aktuální dílek.

Rysy (pořadí ve vektoru vah):
    0  vyska_dopadu     výška středu dílku při dopadu (řádky od 0)
    1  erodovane        smazané řady × počet buněk dílku ve smazaných řadách
    2  prechody_radku   přechody plná/prázdná v řádcích (stěny jsou plné)
    3  prechody_sloupcu přechody plná/prázdná ve sloupcích (dno je plné)
    4  diry             prázdné buňky, nad kterými je ve sloupci plná buňka
    5  studny           pro každou studnu hloubky d přičte 1 + 2 + … + d
                        (studna = prázdná buňka nad stavbou, oba sousedé plní)
    6  hloubka_der      součet plných buněk nad každou dírou
    7  radky_s_dirami   počet řádků s aspoň jednou dírou

Hra končí, když se dílek nikam nevejde, nebo po `max_dilku` dílcích (pak
je to „výhra“ s nejvýš 333 řadami). Dílky I O T S Z J L chodí náhodně, každý
s pravděpodobností 1/7; sekvence je dána seedem hry, takže výsledek je pro
dané váhy a seed deterministický. Šum vzniká jen tím, jaké seedy (hry)
zvolíte.

Rozhraní
--------
hraj(vahy, seedy, sirka=6, vyska=12, max_dilku=500, zaznam=False)
    vahy  – pole 8 čísel (stejné pro všechny hry), nebo pole (len(seedy), 8),
            kde každá hra má vlastní váhy;
    seedy – celá čísla, jedna hra na seed;
    vrací dict polí délky len(seedy):
        'rady'   smazané řady (to, co chceme maximalizovat),
        'dilky'  počet položených dílků;
    zaznam=True navíc vrátí 'desky' (řádky první hry po každém dílku jako
    celá čísla, bit c = sloupec c) a 'rady_prubeh' (smazané řady po dílku).
    Hry se simulují najednou (vektorově): jedno volání s 50 hrami je mnohem
    rychlejší než 50 volání po jedné.

Příklad
-------
>>> import numpy as np
>>> from simulator import hraj
>>> w = np.array([-1, 1, -1, -1, -4, -1, 0, 0])   # Dellacherie (6 rysů)
>>> hraj(w, seedy=range(5))["rady"]
"""
from functools import lru_cache
import numpy as np

NAZVY_RYSU = ["vyska_dopadu", "erodovane", "prechody_radku", "prechody_sloupcu",
              "diry", "studny", "hloubka_der", "radky_s_dirami"]
DILKY = "IOTSZJL"
_TVARY = {  # buňky (řádek, sloupec), řádek 0 = spodní
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(1, 0), (1, 1), (1, 2), (0, 1)],
    "S": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "Z": [(1, 0), (1, 1), (0, 1), (0, 2)],
    "J": [(1, 0), (0, 0), (0, 1), (0, 2)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],
}
_POP = np.array([bin(i).count("1") for i in range(1 << 12)])  # počet jedniček


@lru_cache
def _tabulky(sirka):
    """Pro každý dílek všechna umístění: sloupce a řádky 4 buněk, masky řádků."""
    umisteni = []
    for d in DILKY:
        u, b = [], _TVARY[d]
        for _ in range(4):  # otočení o 90°, jen různá
            r0 = min(r for r, c in b); c0 = min(c for r, c in b)
            tvar = sorted((r - r0, c - c0) for r, c in b)
            sir = max(c for r, c in tvar) + 1
            if all(tvar != t for t, _ in u):
                u += [(tvar, x) for x in range(sirka - sir + 1)]
            b = [(c, -r) for r, c in b]
        umisteni.append(u)
    P = max(len(u) for u in umisteni)
    sl = np.zeros((7, P, 4), int); dy = np.zeros((7, P, 4), int)
    masky = np.zeros((7, P, 4), int); platne = np.zeros((7, P), bool)
    for k, u in enumerate(umisteni):
        for p in range(P):
            tvar, x = u[min(p, len(u) - 1)]
            platne[k, p] = p < len(u)
            for i, (r, c) in enumerate(tvar):
                sl[k, p, i], dy[k, p, i] = x + c, r
                masky[k, p, r] |= 1 << (x + c)
    return sl, dy, masky, platne


def _rysy(B, y, dy, W):
    """B (..., H) řádky jako bitové masky po položení dílku.
    Vrací rysy (..., 8), desky po smazání řad a počet smazaných řad."""
    H = B.shape[-1]; PLNY = (1 << W) - 1
    plne = B == PLNY
    nr = plne.sum(-1)
    erod = nr * np.take_along_axis(plne, np.minimum(y[..., None] + dy, H - 1), -1).sum(-1)
    if nr.any():  # plné řady pryč, zbytek sjede dolů
        poradi = np.argsort(plne, axis=-1, kind="stable")
        B = np.where(np.arange(H) < (H - nr)[..., None],
                     np.take_along_axis(B, poradi, -1), 0)
    kryt = np.flip(np.bitwise_or.accumulate(np.flip(B, -1), -1), -1)  # na a nad
    nad = np.zeros_like(kryt); nad[..., :-1] = kryt[..., 1:]           # jen nad
    diry_m = nad & ~B & PLNY
    diry = _POP[diry_m].sum(-1)
    radky_d = (diry_m != 0).sum(-1)
    horni = np.triu(np.ones((H, H), bool), 1)                          # r' > r
    hloubka = (_POP[diry_m[..., :, None] & B[..., None, :]] * horni).sum((-2, -1))
    x = (B << 1) | 1 | (1 << (W + 1))
    pr_r = _POP[(x ^ (x >> 1)) & ((1 << (W + 1)) - 1)].sum(-1)
    pr_s = _POP[B[..., 1:] ^ B[..., :-1]].sum(-1) + _POP[PLNY & ~B[..., 0]]
    studna = ~kryt & ((B << 1) | 1) & ((B >> 1) | (1 << (W - 1))) & PLNY
    studny = _POP[studna].sum(-1); a = studna
    for j in range(1, H):  # buňka přičte délku studny od sebe nahoru
        a = a[..., :-1] & studna[..., j:]
        if not a.any():
            break
        studny = studny + _POP[a].sum(-1)
    vd = y + dy.max(-1) / 2.0
    F = np.stack([vd, erod, pr_r, pr_s, diry, studny, hloubka, radky_d], -1)
    return F.astype(float), B, nr


def hraj(vahy, seedy, sirka=6, vyska=12, max_dilku=500, zaznam=False):
    """Jedna hra pro každý seed (viz docstring modulu)."""
    seedy = [int(s) for s in np.atleast_1d(seedy)]
    G, H, W = len(seedy), vyska, sirka
    assert W <= 10, "nejvýš 10 sloupců"
    w = np.broadcast_to(np.asarray(vahy, float), (G, 8))
    sl, dy, masky, platne = _tabulky(W)
    sekv = np.array([np.random.default_rng(s).integers(7, size=max_dilku) for s in seedy])
    desky = np.zeros((G, H + 4), int)       # 4 řádky rezervy nad deskou
    vys = np.zeros((G, W), int)
    rady = np.zeros(G, int); dilky = np.zeros(G, int)
    zive = np.ones(G, bool); film = []
    sloupce = np.arange(W)
    for t in range(max_dilku):
        g = np.flatnonzero(zive)
        if g.size == 0:
            break
        k = sekv[g, t]
        C, D, M, V = sl[k], dy[k], masky[k], platne[k]
        n, P = C.shape[:2]
        ii = np.arange(n)[:, None, None]
        y = (vys[g][ii, C] - D).max(-1)                       # řádek dopadu
        V = V & (y + D.max(-1) < H)
        B = np.repeat(desky[g][:, None], P, 1)
        B[ii, np.arange(P)[None, :, None], y[..., None] + np.arange(4)] |= M
        F, B, nr = _rysy(B[..., :H], y, D, W)
        skore = np.where(V, np.einsum("gpf,gf->gp", F, w[g]), -np.inf)
        hraje = V.any(-1)
        j = np.arange(n)[hraje]; best = skore.argmax(-1)[hraje]; gh = g[hraje]
        desky[gh, :H] = B[j, best]
        kryt = np.bitwise_or.accumulate(desky[gh, H - 1::-1], -1)
        vys[gh] = ((kryt[..., None] >> sloupce) & 1).sum(-2)
        rady[gh] += nr[j, best]
        dilky[gh] += 1
        zive[g[~hraje]] = False
        if zaznam and zive[0]:
            film.append((desky[0, :H].copy(), int(rady[0])))
    out = {"rady": rady, "dilky": dilky}
    if zaznam:
        out["desky"] = [d for d, _ in film]
        out["rady_prubeh"] = [r for _, r in film]
    return out


if __name__ == "__main__":
    import time
    w = np.array([-1, 1, -1, -1, -4, -1, 0, 0], float)
    t = time.time()
    r = hraj(w, seedy=range(10))
    print(r["rady"], r["dilky"], f"{time.time() - t:.2f} s")
