"""Ukazka pro prvni cviceni: brachistochrona, tedy nejrychlejsi skluzavka.

Mezi dvema body (levy nahore, pravy nize a stranou) hledame tvar drahy, po
ktere kulicka sjede za nejkratsi cas. Zadne treni, jen tihove zrychleni.
Nikdo optimalizaci nerekne, jak ma tvar vypadat - promenne jsou jen vysky
v pevnych x-ovych uzlech. Presto vyjde presne krivka, kterou v roce 1696
vyresili Bernoulli, Newton a Leibniz: cykloida.

    promenne     ... vysky y_1 ... y_{n-2} v pevnych x-ovych uzlech
                     (krajni body jsou pevne a mezi promenne nepatri)
    ucelova f.   ... doba sjezdu kulicky po lomene care danymi uzly,
                     spocitana ze zakona zachovani energie:
                     v(y) = sqrt(2 g (y_start - y)),
                     doba segmentu = delka / prumerna rychlost na segmentu
                     (pro usecku je to presne, protoze zrychleni podel
                     usecky je konstantni a rychlost roste linearne s casem)
    omezeni      ... prosta ohranicenost promennych (box):
                     kazdy vnitrni uzel musi lezet aspon nepatrne pod startem
                     - jinak by rychlost byla nulova a cas nekonecny -
                     a nesmi spadnout hloubeji nez y_dolni_mez

Kontrola: k teto uloze existuje ZNAME ANALYTICKE RESENI. Cykloidu
x = R (theta - sin theta), y = -R (1 - cos theta) prolozime obema body
(R a koncovy uhel theta_1 dopocitame numericky) a jeji dobu sjezdu
porovname s numerickym optimem. Je to ukazka dvou pravidel predmetu
najednou: "mala instance se znamym resenim" a "kontrola, ktera umi selhat".
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq, minimize, minimize_scalar

G = 9.81                                          # tihove zrychleni [m/s2]
N = 51                                            # pocet uzlu vcetne krajnich
XB, YB = 2.0, -0.6                                # cilovy bod [m]; start je (0, 0)
Y_DOLNI_MEZ = -2.0                                # jak hluboko smi draha klesnout [m]
Y_HORNI_MEZ = -1e-4                               # osetreni singularity na startu

x = np.linspace(0.0, XB, N)                       # x-ove uzly jsou pevne


def doba_sjezdu(x, y):
    """Doba sjezdu kulicky po lomene care (x, y) ze startu v klidu.

    Rychlost plyne ze zachovani energie: v = sqrt(2 g (y[0] - y)).
    Na kazde usecce je zrychleni konstantni, takze rychlost roste linearne
    s casem a plati presne  t = delka / ((v_i + v_{i+1}) / 2).
    Tim se obejde singularita na startu, kde je okamzita rychlost nulova -
    integrovat 1/v by tam neslo, prumerna rychlost je konecna.
    """
    v = np.sqrt(2.0 * G * np.maximum(y[0] - y, 0.0) + 1e-16)
    delka = np.hypot(np.diff(x), np.diff(y))
    return float(np.sum(2.0 * delka / (v[:-1] + v[1:])))


def ucelova(y_vnitrni):
    """Blackbox ucelova funkce: dostane vysky vnitrnich uzlu, vrati cas."""
    return doba_sjezdu(x, np.concatenate(([0.0], y_vnitrni, [YB])))


# ---------------------------------------------------------------- analyticke
# Cykloida x = R (th - sin th), y = -R (1 - cos th) ma projit bodem (XB, YB).
# Podil obou rovnic vyradi R, zbyde jedna rovnice pro koncovy uhel th_1.
def _rovnice_uhlu(th):
    return (th - np.sin(th)) / (1.0 - np.cos(th)) - XB / abs(YB)


theta_1 = brentq(_rovnice_uhlu, 1e-3, 2.0 * np.pi - 1e-3)
R = abs(YB) / (1.0 - np.cos(theta_1))
cas_cykloida = np.sqrt(R / G) * theta_1           # presna doba sjezdu po cykloide

th = np.linspace(0.0, theta_1, 600)
x_cyk = R * (th - np.sin(th))
y_cyk = -R * (1.0 - np.cos(th))

# ------------------------------------------------------------------ srovnani
y_primka = np.linspace(0.0, YB, N)                # startovni bod optimalizace
cas_primka = doba_sjezdu(x, y_primka)


def _cas_oblouku(prohnuti):
    """Kruhovy oblouk prochazejici obema body, prohnuty pod tetivu."""
    return doba_sjezdu(*_oblouk(prohnuti))


def _oblouk(prohnuti, pocet=400):
    """Body kruhoveho oblouku danych 'prohnuti' (hloubka pod stredem tetivy)."""
    if prohnuti < 1e-6:                           # degeneruje na primku
        t = np.linspace(0.0, 1.0, pocet)
        return t * XB, t * YB
    tetiva = np.hypot(XB, YB)
    polomer = (tetiva ** 2 / 4.0 + prohnuti ** 2) / (2.0 * prohnuti)
    # normala k tetive smerem dolu
    nx, ny = -YB / tetiva, XB / tetiva
    if ny > 0:
        nx, ny = -nx, -ny
    sx, sy = XB / 2.0 + nx * prohnuti, YB / 2.0 + ny * prohnuti   # nejnizsi bod
    cx, cy = sx - nx * polomer, sy - ny * polomer                 # stred kruznice
    a0 = np.arctan2(0.0 - cy, 0.0 - cx)               # uhel startu
    a1 = np.arctan2(YB - cy, XB - cx)                 # uhel cile
    am = np.arctan2(sy - cy, sx - cx)                 # uhel stredu oblouku
    sweep = (a1 - a0) % (2.0 * np.pi)                 # vyber tu pulku kruznice,
    if (am - a0) % (2.0 * np.pi) > sweep:             # ktera opravdu vede dolu
        sweep -= 2.0 * np.pi
    a = a0 + np.linspace(0.0, sweep, pocet)
    return cx + polomer * np.cos(a), cy + polomer * np.sin(a)


nejlepsi_oblouk = minimize_scalar(_cas_oblouku, bounds=(0.01, 1.2), method="bounded")
cas_oblouk = float(nejlepsi_oblouk.fun)

# -------------------------------------------------------------- optimalizace
meze = [(Y_DOLNI_MEZ, Y_HORNI_MEZ)] * (N - 2)     # box omezeni na vnitrni uzly
start = np.clip(y_primka[1:-1], Y_DOLNI_MEZ, Y_HORNI_MEZ)

t0 = time.perf_counter()
vysledek = minimize(ucelova, start, method="L-BFGS-B", bounds=meze,
                    options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-12})
doba_behu = time.perf_counter() - t0

y_opt = np.concatenate(([0.0], vysledek.x, [YB]))
cas_opt = doba_sjezdu(x, y_opt)

# ------------------------------------------------------------------ kontrola
rozdil = 100.0 * (cas_opt - cas_cykloida) / cas_cykloida
zrychleni = 100.0 * (cas_primka - cas_opt) / cas_primka

print(f"konec optimalizace : {vysledek.message} ({vysledek.nit} iteraci)")
print(f"doba behu          : {doba_behu:.2f} s")
print()
print(f"primka             : {cas_primka:.4f} s")
print(f"kruhovy oblouk     : {cas_oblouk:.4f} s  (nejlepsi z jednoparametricke rodiny)")
print(f"numericke optimum  : {cas_opt:.4f} s  ({N} uzlu, {N - 2} promennych)")
print(f"analyticka cykloida: {cas_cykloida:.4f} s  (R = {R:.4f} m, theta_1 = {theta_1:.4f} rad)")
print()
print(f"optimum je o {zrychleni:.1f} % rychlejsi nez primka")
print(f"odchylka od znameho reseni: {rozdil:+.3f} %")

# kontrola, ktera umi selhat - mala instance se znamym resenim
assert abs(rozdil) < 1.0, f"numericke optimum se lisi od cykloidy o {rozdil:.2f} %"
assert cas_opt < cas_primka, "optimalizace nenasla nic lepsiho nez primku"
print("kontrola proti analytickemu reseni prosla")

# -------------------------------------------------------------------- obrazek
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(10.5, 5.6))

ax.plot(x_cyk, y_cyk, lw=11, color="#ffcc66", solid_capstyle="round",
        label=f"analytická cykloida  {cas_cykloida:.3f} s")
ax.plot(x, y_opt, lw=2.4, color="#007f86", marker="o", ms=3.2,
        label=f"numerické optimum  {cas_opt:.3f} s")
ax.plot([0, XB], [0, YB], lw=2.4, ls="--", color="#c73e1d",
        label=f"přímka  {cas_primka:.3f} s")

ax.plot([0], [0], "o", ms=13, color="#222222", zorder=5)
ax.plot([XB], [YB], "o", ms=13, color="#222222", zorder=5)
ax.annotate("start", (0, 0), textcoords="offset points", xytext=(6, 12), fontsize=15)
ax.annotate("cíl", (XB, YB), textcoords="offset points", xytext=(-6, 12),
            ha="right", fontsize=15)

ax.set_xlabel("vodorovná vzdálenost [m]")
ax.set_ylabel("výška [m]")
ax.set_title(f"Nejrychlejší skluzavka není přímka — je o {zrychleni:.0f} % rychlejší",
             fontsize=17)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=14, framealpha=0.95)
plt.tight_layout()
plt.savefig("cviceni/C1/obrazky/brachistochrona.svg")
print("obrazek ulozen do cviceni/C1/obrazky/brachistochrona.svg")

# -------------------------------------------------------------------- animace
def priprav_drahu(x, y):
    """Vrati (x, y, kumulativni casy uzlu, rychlosti v uzlech)."""
    v = np.sqrt(2.0 * G * np.maximum(y[0] - y, 0.0) + 1e-16)
    delka = np.hypot(np.diff(x), np.diff(y))
    dt = 2.0 * delka / (v[:-1] + v[1:])
    return x, y, np.concatenate(([0.0], np.cumsum(dt))), v


def poloha(draha, t):
    """Poloha kulicky v case t - podle skutecne rychlosti, ne podle parametru."""
    xs, ys, casy, v = draha
    if t >= casy[-1]:
        return xs[-1], ys[-1]
    i = int(np.searchsorted(casy, t, side="right") - 1)
    tau = t - casy[i]
    dt = casy[i + 1] - casy[i]
    zrych = (v[i + 1] - v[i]) / dt                # podel usecky je konstantni
    s = v[i] * tau + 0.5 * zrych * tau ** 2       # ujeta draha po usecce
    delka = float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
    p = min(max(s / delka, 0.0), 1.0)
    return xs[i] + p * (xs[i + 1] - xs[i]), ys[i] + p * (ys[i + 1] - ys[i])


draha_opt = priprav_drahu(x, y_opt)
draha_pri = priprav_drahu(np.array([0.0, XB]), np.array([0.0, YB]))

POCET = 70                                        # snimku vlastniho sjezdu
VYDRZ = 20                                        # snimku dlouhe vydrze na konci
casy_snimku = np.linspace(0.0, cas_primka, POCET)
# prvni snimek je vysledny stav, aby obrazek daval smysl i v PDF bez animace
poradi = [POCET - 1] + list(range(POCET)) + [POCET - 1] * VYDRZ

fig_a, ax_a = plt.subplots(figsize=(9.5, 5.2))
ax_a.plot(x, y_opt, lw=3, color="#007f86")
ax_a.plot([0, XB], [0, YB], lw=3, ls="--", color="#c73e1d")
ax_a.plot([0, XB], [0, YB], "o", ms=11, color="#222222", zorder=5)
kul_pri, = ax_a.plot([], [], "o", ms=21, color="#c73e1d", zorder=6)
kul_opt, = ax_a.plot([], [], "o", ms=17, color="#007f86", zorder=7,
                     mec="white", mew=2)
popis_cas = ax_a.text(0.985, 0.96, "", transform=ax_a.transAxes, ha="right", va="top",
                      fontsize=19, family="monospace")
popis_opt = ax_a.text(0.985, 0.80, "", transform=ax_a.transAxes, ha="right", va="top",
                      fontsize=17, family="monospace", color="#007f86")
popis_pri = ax_a.text(0.985, 0.67, "", transform=ax_a.transAxes, ha="right", va="top",
                      fontsize=17, family="monospace", color="#c73e1d")
ax_a.set_xlabel("vodorovná vzdálenost [m]", fontsize=15)
ax_a.set_ylabel("výška [m]", fontsize=15)
ax_a.set_title("Obě kuličky vypuštěny naráz, obě jedou jen tíhou", fontsize=17)
ax_a.set_aspect("equal")
ax_a.set_xlim(-0.15, XB + 0.15)
ax_a.set_ylim(-0.95, 0.28)
ax_a.grid(alpha=0.25)


def snimek(k):
    t = casy_snimku[poradi[k]]
    kul_opt.set_data(*[[c] for c in poloha(draha_opt, t)])
    kul_pri.set_data(*[[c] for c in poloha(draha_pri, t)])
    popis_cas.set_text(f"čas {t:5.2f} s")
    popis_opt.set_text(f"optimum {min(t, cas_opt):5.2f} s"
                       + ("  v cíli" if t >= cas_opt else ""))
    popis_pri.set_text(f"přímka  {min(t, cas_primka):5.2f} s"
                       + ("  v cíli" if t >= cas_primka else ""))
    return kul_opt, kul_pri, popis_cas, popis_opt, popis_pri


anim = FuncAnimation(fig_a, snimek, frames=len(poradi), interval=60, blit=False)
anim.save("cviceni/C1/obrazky/brachistochrona.gif", writer=PillowWriter(fps=16), dpi=76)
print("animace ulozena do cviceni/C1/obrazky/brachistochrona.gif")
