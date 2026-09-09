"""Ukazka pro prvni cviceni: mekke pristani rakety jako konvexni uloha.

Nemedicinska "ochutnavka" toho, co optimalizace umi. Uloha je skutecna:
algoritmus G-FOLD (Acikmese & Ploen), ktery resi presne tohle, bezi na palube
pristavajicich raket. Cely vypocet je konvexni, takze doba reseni je
predvidatelna - to je duvod, proc se smi pustit na palube v realnem case.

    promenne     ... poloha p_k, rychlost v_k a vektor tahu T_k v case
    ucelova f.   ... minimalizuj spotrebu paliva, tedy soucet velikosti tahu
    omezeni      ... Newtonova pohybova rovnice (diskretizovana),
                     zadany start, mekke pristani (nulova poloha i rychlost),
                     raketa nesmi pod zem a musi zustat v pristavacim kuzelu,
                     velikost tahu mezi Tmin a Tmax

Trik: omezeni ||T|| >= Tmin je NEKONVEXNI (motor nejde skrtit pod minimum ani
vypnout). Zavedenim pomocne promenne sigma s ||T|| <= sigma, Tmin <= sigma <= Tmax
vznikne konvexni uloha - a da se dokazat, ze optimum je stejne
(tzv. lossless convexification). Presne tenhle druh preformulovani je to,
co se v predmetu ucime.

Skript pocita a porovnava TRI zpusoby, jak stejny sestup odletet:

    1. optimalni rizeni ze solveru  ... vyjde bang-bang, nikdo to tak nezadal
    2. konstantni tah motoru        ... jedno cislo skrtici klapky na cely let,
                                        smeruje se jen gimbalem
    3. jedno sepnuti na konci       ... motor na minimu a brzda az na konec
                                        (hoverslam, "suicide burn")

Prvni dve varianty se lisi jen spotrebou, treti vubec neexistuje - a to je
stejne poucne jako cislo: solver to nezkousi uhodnout, on to dokaze.
"""

import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

N, dt = 60, 1.0                                   # pocet kroku, delka kroku [s]
g, m = 9.81, 2000.0                               # tihove zrychleni [m/s2], hmotnost [kg]
gv = np.array([0.0, g])                           # vektor tihoveho zrychleni
Tmax, Tmin = 30000.0, 5000.0                      # mezni tah motoru [N]
p0 = np.array([1200.0, 1500.0])                   # pocatecni poloha [m]
v0 = np.array([-60.0, -80.0])                     # pocatecni rychlost [m/s]


def vyres(navic=None):
    """Sestavi zakladni ulohu, volitelne s dalsim omezenim na profil tahu.

    Argument `navic` je funkce, ktera dostane promennou sigma (velikost tahu
    v jednotlivych krocich) a vrati seznam dalsich omezeni. Tim se ze stejne
    ulohy udela varianta "konstantni tah" nebo "pozdni zazeh".
    """
    p = cp.Variable((N + 1, 2))                   # poloha
    v = cp.Variable((N + 1, 2))                   # rychlost
    T = cp.Variable((N, 2))                       # vektor tahu
    sigma = cp.Variable(N)                        # horni odhad velikosti tahu

    omezeni = [
        p[0] == p0, v[0] == v0,                   # odkud startujeme
        p[N] == 0, v[N] == 0,                     # mekke pristani presne na plosine
        p[:, 1] >= 0,                             # nesmi pod zem
        p[:, 1] >= 0.5 * cp.abs(p[:, 0]),         # pristavaci kuzel (glide slope)
        cp.norm(T, axis=1) <= sigma,              # lossless convexification
        sigma >= Tmin, sigma <= Tmax,
    ]
    for k in range(N):                            # diskretizovana Newtonova rovnice
        omezeni += [
            v[k + 1] == v[k] + dt * (T[k] / m - gv),
            p[k + 1] == p[k] + dt * (v[k] + v[k + 1]) / 2,
        ]
    if navic is not None:
        omezeni += navic(sigma)

    uloha = cp.Problem(cp.Minimize(cp.sum(sigma) * dt), omezeni)
    t0 = time.time()
    uloha.solve()
    return uloha, time.time() - t0, p.value, v.value, T.value, sigma.value


def simuluj(x):
    """Odleta sestup s konstantni velikosti tahu x[0] a uhly gimbalu x[1:].

    Tytez pohybove rovnice jako v modelu vyse, jen dopredu a bez solveru;
    scitani pres kroky je vektorove, protoze tuhle funkci vola NLP solver
    nekolik tisickrat.
    """
    c, uhly = x[0], x[1:]
    a = c * np.stack([np.cos(uhly), np.sin(uhly)], axis=1) / m - gv
    v = np.vstack([v0, v0 + dt * np.cumsum(a, axis=0)])
    p = np.vstack([p0, p0 + dt * np.cumsum((v[:-1] + v[1:]) / 2, axis=0)])
    return p, v


# ---------------------------------------------------------------- 1. optimum
uloha, cas, p_opt, v_opt, T_opt, _ = vyres()
tah_opt = np.linalg.norm(T_opt, axis=1)
impuls_opt = uloha.value

print(f"status: {uloha.status}, vyreseno za {cas:.2f} s")
print(f"spotrebovany impuls : {impuls_opt / 1e6:.2f} MN*s")
print(f"dopadova rychlost   : {np.linalg.norm(v_opt[-1]):.4f} m/s")
print(f"tah                 : {tah_opt.min():.0f} az {tah_opt.max():.0f} N (meze {Tmin:.0f}-{Tmax:.0f})")

# kontrola dosazenim zpet - kontrola, ktera umi selhat
assert np.linalg.norm(p_opt[-1]) < 1e-3, "raketa nepristala v pocatku"
assert np.linalg.norm(v_opt[-1]) < 1e-3, "pristani neni mekke"
assert (p_opt[:, 1] >= -1e-6).all(), "trajektorie prochazi pod zemi"
assert tah_opt.max() <= Tmax + 1e-3, "prekrocen maximalni tah"
print("kontrola omezeni prosla")


# --------------------------------------------------------- 2. konstantni tah
# "Necham motor na jednom cisle a jen s nim natacim." Konvexni relaxace (sigma
# stejna ve vsech krocich) da spodni odhad spotreby, ale jeji reseni jeste neni
# fyzikalni: v nekterych krocich vyjde ||T|| < sigma, coz klapka drzena na
# jednom cisle neumi. Skutecne omezeni ||T_k|| = c je NEKONVEXNI (povrch koule,
# ne cela koule), takze se dopocita obecnym NLP solverem - rozjezd z relaxace.
uloha_k, cas_k, _, _, T_rel, sigma_rel = vyres(lambda s: [s[1:] == s[:-1]])
odhad_k = uloha_k.value

def dosednuti(x):                                 # ma byt nula: p_N = 0, v_N = 0
    p, v = simuluj(x)
    return np.concatenate([p[N] / 100, v[N]])     # skalovani, at maji podobnou vahu


def kuzel(x):                                     # ma byt >= 0 ve vsech krocich
    p, _ = simuluj(x)
    return (p[:, 1] - 0.5 * np.abs(p[:, 0])) / 100


start = np.concatenate([[sigma_rel[0]], np.arctan2(T_rel[:, 1], T_rel[:, 0])])
t0 = time.time()
nlp = minimize(                                   # promenne: velikost tahu a N uhlu
    lambda x: x[0] * N * dt / 1e6, start, method="SLSQP",
    bounds=[(Tmin, Tmax)] + [(-np.pi, np.pi)] * N,
    constraints=[{"type": "eq", "fun": dosednuti}, {"type": "ineq", "fun": kuzel}],
    options={"maxiter": 600, "ftol": 1e-10},
)
cas_nlp = time.time() - t0
tah_konst = nlp.x[0]
p_konst, v_konst = simuluj(nlp.x)
impuls_konst = tah_konst * N * dt

assert nlp.success, "nekonvexni dopocet konstantniho tahu nedobehl"
assert np.linalg.norm(p_konst[-1]) < 1e-3, "konstantni tah minul plosinu"
assert np.linalg.norm(v_konst[-1]) < 1e-3, "konstantni tah nedosedl mekce"
assert (p_konst[:, 1] >= -1e-6).all(), "konstantni tah vede pod zem"

print()
print(f"konstantni tah      : {tah_konst:.0f} N po celych {N * dt:.0f} s")
print(f"  spotreba          : {impuls_konst / 1e6:.2f} MN*s "
      f"({100 * (impuls_konst / impuls_opt - 1):+.1f} % oproti optimu)")
print(f"  konvexni relaxace : {odhad_k / 1e6:.3f} MN*s (spodni mez, dopocet dal "
      f"{impuls_konst / 1e6:.3f})")
print(f"  doba vypoctu      : {cas_k:.2f} s relaxace + {cas_nlp:.2f} s nekonvexni dopocet")


# ---------------------------------------------------- 3. jedno sepnuti na konci
# "Necham to padat a zabrzdim az na konci." Motor nejde vypnout, takze cekani
# znamena tah na minimu; hleda se nejpozdejsi okamzik zazehu, pri kterem uloha
# jeste ma reseni.
posledni_zazeh, impuls_zazeh, status_pote = None, None, None
for zpozdeni in range(N):                         # zpozdeni = kolik sekund se ceka
    uloha_z, *_ = vyres(lambda s, z=zpozdeni: [s[:z] == Tmin])
    if uloha_z.status != cp.OPTIMAL:
        status_pote = uloha_z.status
        break
    posledni_zazeh, impuls_zazeh = zpozdeni, uloha_z.value

assert status_pote is not None, "cekani na brzdu tady vubec nevadi - zmenilo se zadani?"

print()
print(f"jedno sepnuti na konci: nejpozdejsi zazeh v case {posledni_zazeh} s")
print(f"  spotreba          : {impuls_zazeh / 1e6:.2f} MN*s "
      f"({100 * (impuls_zazeh / impuls_opt - 1):+.2f} % oproti optimu)")
print(f"  delsi cekani      : {status_pote} - uloha nema reseni, raketa uz nema "
      f"cim zabrzdit")


# --------------------------------------------------------------------- obrazek
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax[0].plot(p_opt[:, 0], p_opt[:, 1], "-o", ms=3.5, color="#007f86", label="optimum")
ax[0].plot(p_konst[:, 0], p_konst[:, 1], "-", lw=1.8, color="#e08214",
           label="konstantní tah")
xs = np.linspace(-1400, 1400, 100)
ax[0].plot(xs, 0.5 * np.abs(xs), "k--", lw=0.9, label="přistávací kužel")
for k in range(0, N, 3):                          # sipky ukazuji, kam motor tlaci
    ax[0].arrow(*p_opt[k], *(-T_opt[k] / Tmax * 260), color="#c73e1d",
                head_width=28, length_includes_head=True)
ax[0].set(xlabel="vodorovná vzdálenost [m]", ylabel="výška [m]",
          title="Trajektorie a vektory tahu", ylim=(-60, 1750))
ax[0].set_aspect("equal")
ax[0].legend(loc="upper left", fontsize=9)

t = np.arange(N) * dt
ax[1].plot(t, tah_opt / 1000, lw=2, color="#007f86", label="optimum")
ax[1].plot(t, np.full(N, tah_konst / 1000), lw=1.8, color="#e08214",
           label="konstantní tah")
ax[1].axvline(posledni_zazeh, color="#c73e1d", ls=":", lw=1.4)
ax[1].annotate(f"pozdější zážeh než\nv {posledni_zazeh}. s už nemá řešení",
               xy=(posledni_zazeh, 16.5), xytext=(8, 12.5), fontsize=8.5,
               color="#c73e1d",
               arrowprops=dict(arrowstyle="->", color="#c73e1d", lw=0.9))
ax[1].axhline(Tmax / 1000, ls="--", c="k", lw=0.9)
ax[1].axhline(Tmin / 1000, ls="--", c="k", lw=0.9)
ax[1].set(xlabel="čas [s]", ylabel="tah [kN]",
          title="Optimální řízení motoru vyšlo „na doraz“", ylim=(0, 34))
ax[1].text(42, Tmax / 1000 - 2.5, "maximální tah", ha="center", fontsize=9)
ax[1].text(42, Tmin / 1000 + 1.2, "minimální tah", ha="center", fontsize=9)
ax[1].legend(loc="center right", fontsize=9)

popisky = ["optimum\n(bang-bang)", "konstantní\ntah", "jedno sepnutí\nna konci"]
hodnoty = [impuls_opt / 1e6, impuls_konst / 1e6]
strop = 1.9                                       # vyska tretiho, "nemozneho" sloupce
ax[2].bar(popisky[:2], hodnoty, width=0.55, color=["#007f86", "#e08214"])
ax[2].bar(popisky[2], strop, width=0.55, facecolor="none",   # neexistuje -> mimo graf
          edgecolor="#c73e1d", hatch="//", lw=1.2)
for x, y in zip(popisky, hodnoty):                 # desetinna carka, ne tecka
    ax[2].text(x, y + 0.04, f"{y:.2f} MN·s".replace(".", ","), ha="center", fontsize=10)
ax[2].text(1, hodnoty[1] + 0.19,
           f"{100 * (impuls_konst / impuls_opt - 1):+.1f} %".replace(".", ","),
           ha="center", fontsize=10, color="#e08214")
bily = dict(facecolor="white", edgecolor="none", pad=2.5)   # at je text pres srafu videt
ax[2].text(2, 0.78, "nemá\nřešení", ha="center", va="center", bbox=bily,
           fontsize=12, color="#c73e1d", weight="bold")
ax[2].text(2, 0.45, f"po {posledni_zazeh}. sekundě\nuž raketa nemá\nčím zabrzdit",
           ha="center", va="center", bbox=bily, fontsize=8.5, color="#c73e1d")
ax[2].set(ylabel="spotřebovaný impuls [MN·s]", ylim=(0, strop),
          title="Cena tří způsobů, jak to odletět")

plt.tight_layout()
plt.savefig("cviceni/C1/obrazky/raketa.svg")
print("\nobrazek ulozen do cviceni/C1/obrazky/raketa.svg")
