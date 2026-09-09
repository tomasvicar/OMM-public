"""Ukazka pro prvni cviceni: topologicka optimalizace lehke vyztuhy.

Medicinska "ochutnavka" toho, co optimalizace umi. Zadani: mame obdelnikovou
desku, vlevo je pevne prichycena ke kosti (vetknuti), vpravo na ni pusobi sila.
Smime pouzit jen 35 % materialu, ktery by se do desky vesel. Kam ho dat, aby
byla vyztuha co nejtuzsi, tedy aby se pod zatizenim co nejmin prohnula?

    promenne     ... hustota materialu x_e v kazdem elementu site,
                     0 = dira, 1 = plny titan, mezihodnoty jsou docasne
    ucelova f.   ... poddajnost (compliance) c = f^T u, tedy prace, kterou
                     sila vykona na posunuti; mala poddajnost = velka tuhost
    omezeni      ... prumerna hustota nejvyse 35 % (rozpocet materialu),
                     0 <= x_e <= 1, a rovnovaha K(x) u = f z metody
                     konecnych prvku

Nikdo tvar nekreslil - vyleze z vypoctu. Presne takhle dnes vznikaji 3D
tistene titanove implantaty, fixatory a nosne dily v letectvi.

Metoda: SIMP (tuhost elementu ~ x_e^3, cimz se sede mezihodnoty nevyplati)
+ filtr citlivosti (bez nej vznika sachovnice) + optimality criteria, coz je
jednoducha aktualizacni formule odvozena z KKT podminek. Jde o Pythonovy port
klasickeho 88-radkoveho kodu (Andreassen a kol. 2011, navazuje na Sigmund 2001).

Uloha je NEKONVEXNI - vysledek zavisi na startu a je to lokalni optimum.
Ze je dobre, poznam az porovnanim s referenci, ne z toho, ze algoritmus dobehl.
"""

import time

import matplotlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- parametry ulohy ---------------------------------------------------------
NELX, NELY = 120, 60          # sit konecnych prvku (sirka x vyska desky)
VOLFRAC = 0.35                # rozpocet materialu: nejvyse 35 % objemu
PENAL = 3.0                   # exponent SIMP - penalizuje sede mezihodnoty
RMIN = 2.0                    # polomer filtru citlivosti [element]
ITERACI = 50                  # pevny pocet iteraci, at je beh predvidatelny
KROK = 0.2                    # maximalni zmena hustoty za jednu iteraci
E0, EMIN, NU = 1.0, 1e-9, 0.3  # modul plneho materialu, diry a Poissonovo cislo


def matice_tuhosti_elementu(nu):
    """Tuhost jednoho ctvercoveho elementu (rovinna napjatost, 4 uzly)."""
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
                  -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    return 1 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


KE = matice_tuhosti_elementu(NU)
POCET = NELX * NELY

# --- ocislovani stupnu volnosti ----------------------------------------------
# uzly se cisluji po sloupcich, kazdy ma dva posuny (vodorovny a svisly)
edof = np.zeros((POCET, 8), dtype=int)
for ex in range(NELX):
    for ey in range(NELY):
        el = ey + ex * NELY
        n1 = (NELY + 1) * ex + ey            # levy horni uzel elementu
        n2 = (NELY + 1) * (ex + 1) + ey      # pravy horni uzel elementu
        edof[el] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                    2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3]
iK = np.kron(edof, np.ones((8, 1))).flatten().astype(int)
jK = np.kron(edof, np.ones((1, 8))).flatten().astype(int)

# --- filtr citlivosti --------------------------------------------------------
# Bez nej vyjde "sachovnice": stridani plnych a prazdnych elementu, ktera je v
# diskretizaci umele tuha. Filtr rozmaze citlivost po okoli o polomeru RMIN.
r = int(np.ceil(RMIN))
iH, jH, sH = [], [], []
for i in range(NELX):
    for j in range(NELY):
        e1 = i * NELY + j
        for k in range(max(i - r + 1, 0), min(i + r, NELX)):
            for l in range(max(j - r + 1, 0), min(j + r, NELY)):
                vaha = RMIN - np.hypot(i - k, j - l)
                if vaha > 0:
                    iH.append(e1)
                    jH.append(k * NELY + l)
                    sH.append(vaha)
H = coo_matrix((sH, (iH, jH)), shape=(POCET, POCET)).tocsr()
Hs = np.array(H.sum(1)).flatten()

# --- okrajove podminky: vlevo vetknuti, vpravo uprostred sila dolu ------------
ndof = 2 * (NELX + 1) * (NELY + 1)
uzel_zatizeni = (NELX + 1) * (NELY + 1) - (NELY + 1) + NELY // 2
f = np.zeros(ndof)
f[2 * uzel_zatizeni + 1] = -1.0                    # svisla sila o velikosti 1
pevne = np.arange(0, 2 * (NELY + 1))               # cela leva hrana, oba smery
volne = np.setdiff1d(np.arange(ndof), pevne)


def poddajnost(x):
    """Vyresi rovnovahu K(x) u = f a vrati poddajnost a citlivost po elementech."""
    E = EMIN + x**PENAL * (E0 - EMIN)              # SIMP: tuhost roste s x^3
    sK = ((KE.flatten()[np.newaxis]).T * E).flatten(order="F")
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    u = np.zeros(ndof)
    u[volne] = spsolve(K[volne, :][:, volne], f[volne])
    ue = u[edof]
    ce = (ue @ KE * ue).sum(1)                     # energie napjatosti elementu
    return (E * ce).sum(), ce


def krok_optimality_criteria(x, dc):
    """Aktualizace hustot pri presne dodrzenem rozpoctu materialu (pulenim intervalu)."""
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.clip(x * np.sqrt(-dc / lmid), np.maximum(0.0, x - KROK),
                       np.minimum(1.0, x + KROK))
        if xnew.sum() > VOLFRAC * POCET:           # moc materialu -> zdrazit
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# --- vlastni optimalizace ----------------------------------------------------
print(f"sit {NELX}x{NELY} = {POCET} elementu, rozpocet materialu {VOLFRAC:.0%}")
x = np.full(POCET, VOLFRAC)                        # start: material rovnomerne
historie = [x.copy()]
t0 = time.time()
for it in range(ITERACI):
    c, ce = poddajnost(x)
    if it == 0:
        c_rovnomerne = c                           # reference: rovnomerna deska
    dc = -PENAL * x**(PENAL - 1) * (E0 - EMIN) * ce
    dc = np.asarray(H @ (x * dc)) / Hs / np.maximum(1e-3, x)   # filtr citlivosti
    xnova = krok_optimality_criteria(x, dc)
    zmena = np.abs(xnova - x).max()
    x = xnova
    historie.append(x.copy())
    if it % 10 == 0 or it == ITERACI - 1:
        print(f"iterace {it:3d}   poddajnost {c:9.2f}   objem {x.mean():.3f}"
              f"   zmena {zmena:.3f}")
c_opt, _ = poddajnost(x)
cas = time.time() - t0

# --- kontrola a srovnani s referencemi ---------------------------------------
c_plna, _ = poddajnost(np.ones(POCET))             # cela deska z plneho materialu
assert x.mean() <= VOLFRAC + 1e-6, "rozpocet materialu je prekrocen"
assert (x >= -1e-9).all() and (x <= 1 + 1e-9).all(), "hustota mimo interval [0,1]"

sede = np.mean((x > 0.1) & (x < 0.9))
print(f"\nhotovo za {cas:.1f} s")
print(f"rovnomerna deska (35 % materialu) : poddajnost {c_rovnomerne:8.2f}")
print(f"optimalizovany tvar (35 % mater.) : poddajnost {c_opt:8.2f}"
      f"  -> {c_rovnomerne / c_opt:.1f}x tuzsi pri stejne hmotnosti")
print(f"plna deska (100 % materialu)      : poddajnost {c_plna:8.2f}"
      f"  -> optimalizovany tvar je jen {c_opt / c_plna:.1f}x poddajnejsi"
      f" pri {VOLFRAC:.0%} hmotnosti")
print(f"podil sedych elementu (0,1 az 0,9): {sede:.1%} - tvar je temer 0/1")


# --- obrazky -----------------------------------------------------------------
def vykresli(ax, hustota, nadpis):
    """Jeden panel: rozlozeni materialu, vetknuti vlevo a sila vpravo."""
    ax.imshow(hustota.reshape(NELX, NELY).T, cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.plot([-1.5, -1.5], [-0.5, NELY - 0.5], color="#007f86", lw=6,
            solid_capstyle="butt", clip_on=False)
    ax.annotate("", xy=(NELX - 0.5, NELY // 2 + 13), xytext=(NELX - 0.5, NELY // 2),
                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.9",
                                color="#c73e1d", lw=3.5), annotation_clip=False)
    ax.text(NELX + 2, NELY // 2 + 8, "síla", color="#c73e1d", fontsize=15,
            fontweight="bold", va="center")
    ax.text(-9, NELY / 2, "vetknutí", color="#007f86", fontsize=15,
            fontweight="bold", rotation=90, va="center", ha="center")
    ax.set_title(nadpis, fontsize=17, pad=12)
    ax.set_xlim(-12, NELX + 16)
    ax.set_ylim(NELY + 16, -6)
    ax.axis("off")


fig, axy = plt.subplots(1, 2, figsize=(14, 4.4))
vykresli(axy[0], historie[0],
         f"Start: materiál rovnoměrně\npoddajnost {c_rovnomerne:.0f}")
vykresli(axy[1], x,
         f"Výsledek: stejná hmotnost, {c_rovnomerne / c_opt:.0f}× tužší\n"
         f"poddajnost {c_opt:.0f}")
fig.suptitle("Odlehčená výztuha: „z 35 % materiálu udělej co nejtužší nosník“",
             fontsize=19, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig("cviceni/C1/obrazky/topologie.png", dpi=110)
plt.close(fig)
print("\nobrazek ulozen do cviceni/C1/obrazky/topologie.png")

# Animace: PRVNI snimek je vysledek, aby obrazek daval smysl i v PDF, kde se
# animace neprehraje; posledni snimek se opakuje, takze ma dlouhou vydrz.
posledni = len(historie) - 1
snimky = [posledni] + list(range(len(historie))) + [posledni] * 18
fig, ax = plt.subplots(figsize=(7.2, 3.7))
vykresli(ax, x, "")
obraz = ax.images[0]
nadpis = ax.set_title("", fontsize=16, pad=10)


def snimek(k):
    i = snimky[k]
    obraz.set_data(historie[i].reshape(NELX, NELY).T)
    if i == 0:
        nadpis.set_text("start: materiál rozprostřený rovnoměrně")
    elif i == posledni:
        nadpis.set_text(f"výsledek po {ITERACI} iteracích")
    else:
        nadpis.set_text(f"iterace {i} z {ITERACI}")
    return obraz, nadpis


anim = FuncAnimation(fig, snimek, frames=len(snimky), interval=110, blit=False)
anim.save("cviceni/C1/obrazky/topologie.gif", writer=PillowWriter(fps=9), dpi=76)
plt.close(fig)
print("animace ulozena do cviceni/C1/obrazky/topologie.gif")
