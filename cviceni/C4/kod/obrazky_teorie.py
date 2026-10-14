"""Obrazky a cisla reportu teorie.html a tabule.html ctvrteho cviceni.

Tri cesty ke gradientu prubezne funkce f(x1, x2) = (x1 - 1)^2 + (x1 x2 - 2)^2:
numericky (konecne diference), spocitany (analyticky, SymPy) a automaticky
(vypocetni graf, zpetny pruchod). Zavazna cisla jsou v data/cisla.json
(kod/cisla.py); tady se jen prepocitaji a porovnaji.

    uv run python cviceni/C4/kod/obrazky_teorie.py

Vystup: cviceni/C4/obrazky/teorie-*.svg a tabule-kroky.svg. Obrazky vypocetniho
grafu jsou psane primo jako SVG (presne rozmisteni uzlu a popisku), grafy
funkci kresli matplotlib.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spolecne import f, grad_analyticky, grad_numericky  # noqa: E402

KOREN = Path(__file__).resolve().parent.parent
OBRAZKY = KOREN / "obrazky"
CISLA = json.loads((KOREN / "data" / "cisla.json").read_text(encoding="utf-8"))["funkce"]

CERV, MODR, ZEL, AKC, SEDA = "#c73e1d", "#1f4e79", "#1a7f37", "#007f86", "#9ca3af"
TEXT, TLUM = "#1c1c1e", "#5c5c66"
PISMO = "Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif"


def cz(v, d=2):
    """Cislo s desetinnou carkou a typografickym minusem."""
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def g(v):
    """Kratky zapis cisla (bez zbytecnych nul) s carkou a minusem."""
    return f"{v:g}".replace(".", ",").replace("-", "−")


# ====================================================================== cisla
BOD = np.array(CISLA["bod"])
H = 0.1


def dopredu(x1, x2):
    """Dopredny pruchod vypocetnim grafem: hodnoty uzlu."""
    q = x1 - 1
    a = x1 * x2
    r = a - 2
    return {"q": q, "a": a, "r": r, "f": q ** 2 + r ** 2}


def zpet(x1, x2, u):
    """Zpetny pruchod: dF/d(uzel) od vystupu ke vstupum."""
    df_df = 1.0
    df_dq = df_df * 2 * u["q"]
    df_dr = df_df * 2 * u["r"]
    df_da = df_dr * 1
    cesta_q = df_dq * 1                 # x1 -> q -> f
    cesta_a = df_da * x2                # x1 -> a -> r -> f
    return {"df/df": df_df, "df/dq": df_dq, "df/dr": df_dr, "df/da": df_da,
            "cesta_q": cesta_q, "cesta_a": cesta_a,
            "df/dx1": cesta_q + cesta_a, "df/dx2": df_da * x1}


U = dopredu(*BOD)
Z = zpet(*BOD, U)
for k, v in CISLA["uzly_dopredu"].items():
    assert abs(U[k] - v) < 1e-12, k
for k, v in CISLA["uzly_zpet"].items():
    assert abs(Z[k] - v) < 1e-12, k
assert np.allclose(grad_analyticky(*BOD), CISLA["analyticky"])
assert np.allclose([Z["df/dx1"], Z["df/dx2"]], CISLA["autograd"])

# kontrola cisel teorie: diference s h = 0,1 a jeden krok GD
F0 = f(*BOD)
FX1P, FX1M = f(BOD[0] + H, BOD[1]), f(BOD[0] - H, BOD[1])
FX2P, FX2M = f(BOD[0], BOD[1] + H), f(BOD[0], BOD[1] - H)
dopredna = grad_numericky(f, BOD, h=H, centralni=False)
centralni = grad_numericky(f, BOD, h=H, centralni=True)
assert np.allclose(dopredna, CISLA["dopredna_h01"]) and np.allclose(centralni, CISLA["centralni_h01"])
KROK = BOD - 0.1 * grad_analyticky(*BOD)
assert np.allclose(KROK, CISLA["krok_alfa01"]["x"]) and abs(f(*KROK) - CISLA["krok_alfa01"]["f"]) < 1e-9

print("prubezna funkce v bode (1; 1):  f =", g(F0))
print(f"  f(1,1; 1) = {FX1P:.4f}, f(0,9; 1) = {FX1M:.4f}, f(1; 1,1) = {FX2P:.4f}, f(1; 0,9) = {FX2M:.4f}")
print(f"  dopredna h = 0,1: ({(FX1P - F0) / H:.4f}; {(FX2P - F0) / H:.4f})")
print(f"  centralni h = 0,1: ({(FX1P - FX1M) / (2 * H):.4f}; {(FX2P - FX2M) / (2 * H):.4f})")
print("  graf dopredu:", {k: g(v) for k, v in U.items()})
print("  graf zpet:   ", {k: g(v) for k, v in Z.items()})
print(f"  krok GD alfa = 0,1: x = ({KROK[0]:.4f}; {KROK[1]:.4f}), f = {f(*KROK):.4f}")

# 1D rez pro secny a chybu derivace: phi(s) = f(s, s), phi'(1) = -4
def phi(s):
    return f(s, s)


PHI_D1 = -4.0                       # phi'(s) = 2(s-1) + 4s(s^2-2), v s = 1
PHI_D2, PHI_D3 = 6.0, 24.0          # phi'' = 12 s^2 - 6, phi''' = 24 s
assert abs((phi(1 + 1e-6) - phi(1 - 1e-6)) / 2e-6 - PHI_D1) < 1e-8

hs = 10.0 ** np.arange(-16, 0.001, 0.1)
chyba_dop = np.abs((phi(1 + hs) - phi(1)) / hs - PHI_D1)
chyba_cen = np.abs((phi(1 + hs) - phi(1 - hs)) / (2 * hs) - PHI_D1)
chyba_dop[chyba_dop == 0] = np.nan          # nahodne presne trefy by na log ose zmizely
chyba_cen[chyba_cen == 0] = np.nan


def vyhlazene_minimum(chyby):
    """Minimum chyby po klouzavem medianu (aby nerozhodla nahodna trefa)."""
    c = np.nan_to_num(chyby, nan=1e-20)
    m = np.array([np.median(c[max(0, i - 5):i + 6]) for i in range(len(c))])
    i = int(np.argmin(m))
    return hs[i], m[i]


MIN_DOP, MIN_CEN = vyhlazene_minimum(chyba_dop), vyhlazene_minimum(chyba_cen)
print(f"chyba phi'(1): dopredna nejlepe kolem h = {MIN_DOP[0]:.0e} (chyba ~ {MIN_DOP[1]:.0e}), "
      f"centralni kolem h = {MIN_CEN[0]:.0e} (chyba ~ {MIN_CEN[1]:.0e})")
for h in (1e-1, 1e-4, 1e-8, 1e-12):
    i = int(np.argmin(np.abs(hs - h)))
    print(f"  h = {h:.0e}: dopredna {chyba_dop[i]:.1e}, centralni {chyba_cen[i]:.1e}")

# bobtnani vyrazu: logisticke zobrazeni l_{n+1} = 4 l_n (1 - l_n), derivace podle x = l_0; po n krocich je to l_n
x = sp.Symbol("x")
HLOUBKY = list(range(1, 8))
ops_vyraz, ops_derivace, ops_program = [], [], []
lx = x
for n in HLOUBKY:
    lx = 4 * lx * (1 - lx)
    ops_vyraz.append(sp.count_ops(lx))
    ops_derivace.append(sp.count_ops(sp.diff(lx, x)))
    ops_program.append(3 * n)        # jeden krok = 3 operace (1 - l, krat, krat 4)
print("bobtnani (pocet operaci): n, program, vyraz SymPy, derivace SymPy")
for row in zip(HLOUBKY, ops_program, ops_vyraz, ops_derivace):
    print("  ", *row)


# ====================================================================== matplotlib
plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "svg.hashsalt": "mpc-omm-c4-teorie",
                     "axes.spines.top": False, "axes.spines.right": False})
HORNI = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MOCNINA = plt.FuncFormatter(lambda v, _: "1" if v == 1 else "10" + str(round(np.log10(v))).translate(HORNI))
CARKA = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def uloz(fig, jmeno):
    fig.tight_layout()
    fig.savefig(OBRAZKY / jmeno, metadata={"Date": None})
    plt.close(fig)
    print("ulozeno", OBRAZKY / jmeno)


def vrstevnice(ax, rozsah, urovne, popisky=True):
    a1, b1, a2, b2 = rozsah
    X1, X2 = np.meshgrid(np.linspace(a1, b1, 400), np.linspace(a2, b2, 400))
    cs = ax.contour(X1, X2, f(X1, X2), levels=urovne, colors=SEDA, linewidths=0.9)
    if popisky:
        ax.clabel(cs, fmt=lambda v: g(v), fontsize=9, inline_spacing=2)
    ax.xaxis.set_major_formatter(CARKA)
    ax.yaxis.set_major_formatter(CARKA)
    ax.set_xlim(a1, b1)
    ax.set_ylim(a2, b2)
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂", rotation=0, labelpad=10)


def sipka(ax, z, do, barva, lw=2.4):
    ax.annotate("", xy=do, xytext=z,
                arrowprops=dict(arrowstyle="-|>", color=barva, lw=lw, mutation_scale=18,
                                shrinkA=0, shrinkB=0))


BILE = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
OBRAZKY.mkdir(exist_ok=True)

# --- 1) prubezna funkce: vrstevnice, minimum, bod (1; 1), krok -alfa grad f
fig, ax = plt.subplots(figsize=(6.0, 7.0))
vrstevnice(ax, (0.2, 2.6, 0.2, 3.4), [0.05, 0.2, 0.5, 1, 2, 4, 8])
s = np.linspace(0.62, 2.6, 200)
ax.plot(s, 2 / s, ls="--", color=AKC, lw=1.3)
ax.text(0.7, 3.2, "údolí x₁x₂ = 2", color=AKC, fontsize=11, bbox=BILE)
ax.plot(1, 2, "*", color=ZEL, ms=17, zorder=5)
ax.text(1.07, 2.1, "minimum (1; 2), f = 0", color=ZEL, fontsize=12, bbox=BILE)
sipka(ax, BOD, KROK, CERV)
ax.plot(*BOD, "o", color=TEXT, ms=8, zorder=5)
ax.text(0.98, 0.83, "(1; 1), f = 1", ha="right", fontsize=12, bbox=BILE)
ax.text(1.24, 1.02, "−0,1·∇f = (0,2; 0,2)", color=CERV, fontsize=12, bbox=BILE)
uloz(fig, "teorie-funkce.svg")

# --- 2) tecna a secny na 1D rezu phi(s) = f(s, s)
HS = 0.5
s = np.linspace(0.3, 1.75, 300)
sk_dop = (phi(1 + HS) - phi(1)) / HS
sk_cen = (phi(1 + HS) - phi(1 - HS)) / (2 * HS)
print(f"secny h = {HS}: dopredna {sk_dop:.4f}, centralni {sk_cen:.4f}, tecna {PHI_D1}")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s, phi(s), color=TEXT, lw=2.2, label="φ(s) = f(s, s)")
t = np.linspace(0.3, 1.75, 2)
ax.plot(t, 1 + PHI_D1 * (t - 1), color=ZEL, lw=2, label=f"tečna: sklon φ′(1) = {g(PHI_D1)}")
ax.plot(t, 1 + sk_dop * (t - 1), color=MODR, lw=1.8, ls="--",
        label=f"dopředná sečna: sklon {cz(sk_dop, 3)}")
ax.plot(t, phi(1 - HS) + sk_cen * (t - 1 + HS), color=CERV, lw=1.8, ls="-.",
        label=f"centrální sečna: sklon {cz(sk_cen, 0)}")
for sx, barva in [(1 - HS, CERV), (1, TEXT), (1 + HS, MODR)]:
    ax.plot(sx, phi(sx), "o", color=barva, ms=8, zorder=5)
ax.plot(1 + HS, phi(1 + HS), "o", color=CERV, ms=4, zorder=6)
ax.annotate("", xy=(1 + HS, -0.55), xytext=(1, -0.55),
            arrowprops=dict(arrowstyle="<->", color=TLUM, lw=1.2))
ax.text(1 + HS / 2, -0.45, "h", ha="center", color=TLUM)
ax.annotate("", xy=(1 - HS, -0.55), xytext=(1, -0.55),
            arrowprops=dict(arrowstyle="<->", color=TLUM, lw=1.2))
ax.text(1 - HS / 2, -0.45, "h", ha="center", color=TLUM)
ax.set_xticks([0.5, 1, 1.5], ["1 − h", "1", "1 + h"])
ax.set_ylim(-0.8, 4.2)
ax.set_xlim(0.3, 1.75)
ax.set_xlabel("s")
ax.legend(loc="upper right", frameon=False, fontsize=11)
ax.grid(alpha=0.25)
uloz(fig, "teorie-secny.svg")

# --- 3) chyba derivace vs h (log-log), float64
EPS = np.finfo(float).eps
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(hs, chyba_dop, ".-", color=MODR, lw=1, ms=4, label="dopředná diference")
ax.loglog(hs, chyba_cen, ".-", color=CERV, lw=1, ms=4, label="centrální diference")
hh = np.array([1e-16, 1])
ax.loglog(hh, PHI_D2 / 2 * hh, ":", color=MODR, lw=1.3)
ax.loglog(hh, PHI_D3 / 6 * hh ** 2, ":", color=CERV, lw=1.3)
ax.loglog(hh, EPS / hh, ":", color=TLUM, lw=1.3)
ax.text(2e-3, 3 * 2e-3 * 0.12, "~ h", color=MODR, fontsize=12, bbox=BILE)
ax.text(3e-2, 4 * 3e-2 ** 2 * 0.1, "~ h²", color=CERV, fontsize=12, bbox=BILE)
ax.text(2e-14, EPS / 2e-14 * 4, "~ ε / h", color=TLUM, fontsize=12, bbox=BILE)
ax.text(3e-16, 3e-14, "zaokrouhlovací\nchyba převažuje", color=TLUM, fontsize=11, va="bottom")
ax.text(3e-4, 3e-14, "chyba useknutí\npřevažuje", color=TLUM, fontsize=11, va="bottom")
ax.axvline(MIN_DOP[0], color=MODR, lw=0.8, alpha=0.5)
ax.axvline(MIN_CEN[0], color=CERV, lw=0.8, alpha=0.5)
ax.set_ylim(1e-14, 1e2)
ax.set_xlim(1e-16, 1)
ax.xaxis.set_major_formatter(MOCNINA)
ax.yaxis.set_major_formatter(MOCNINA)
ax.set_xticks(10.0 ** np.arange(-16, 1, 4))
ax.set_xticks([], minor=True)
ax.set_yticks(10.0 ** np.arange(-12, 1, 4))
ax.set_yticks([], minor=True)
ax.set_xlabel("krok h")
ax.set_ylabel("|odhad − φ′(1)|")
ax.legend(loc="upper center", frameon=False, fontsize=11)
ax.grid(alpha=0.25, which="major")
uloz(fig, "teorie-chyba-h.svg")

# --- 4) bobtnani vyrazu
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.semilogy(HLOUBKY, ops_derivace, "o-", color=CERV, lw=2, label="derivace v SymPy (sp.diff)")
ax.semilogy(HLOUBKY, ops_vyraz, "s-", color=SEDA, lw=2, label="samotný výraz lₙ v SymPy")
ax.semilogy(HLOUBKY, ops_program, "^-", color=MODR, lw=2, label="program (smyčka): 3 operace na krok")
for n, v in zip(HLOUBKY, ops_derivace):
    ax.text(n, v * 1.35, str(v), ha="center", color=CERV, fontsize=10)
ax.yaxis.set_major_formatter(MOCNINA)
ax.set_xlabel("počet kroků složení n  (lₙ₊₁ = 4 lₙ (1 − lₙ), l₀ = x)")
ax.set_ylabel("počet operací")
ax.set_xticks(HLOUBKY)
ax.set_ylim(1, max(ops_derivace) * 4)
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.grid(alpha=0.25)
uloz(fig, "teorie-bobtnani.svg")

# --- 5) tabule: tri kroky GD s alfa = 0,1 a analytickym gradientem
KROKY = [np.array(k["x"]) for k in CISLA["kroky_alfa01"]]
x = BOD.copy()
for k in CISLA["kroky_alfa01"]:
    assert np.allclose(x, k["x"], atol=1e-6) and abs(f(*x) - k["f"]) < 1e-6
    assert np.allclose(grad_analyticky(*x), k["grad"], atol=1e-6)
    x = x - 0.1 * grad_analyticky(*x)
fig, ax = plt.subplots(figsize=(5.6, 6.6))
vrstevnice(ax, (0.8, 1.75, 0.8, 2.2), [0.02, 0.05, 0.1, 0.2, 0.5, 1])
s = np.linspace(0.9, 1.7, 100)
ax.plot(s, 2 / s, ls="--", color=AKC, lw=1.2)
ax.text(1.42, 1.32, "údolí\nx₁x₂ = 2", color=AKC, fontsize=10, bbox=BILE)
ax.plot(1, 2, "*", color=ZEL, ms=16, zorder=5)
ax.text(0.96, 2.0, "minimum\n(1; 2)", color=ZEL, fontsize=11, ha="right", va="center", bbox=BILE)
for z, do in zip(KROKY, KROKY[1:]):
    sipka(ax, z, do - 0.012 * (do - z) / np.hypot(*(do - z)), CERV, lw=2)
popisky = [(1.03, 0.98, "left", "top"), (1.23, 1.17, "left", "top"),
           (1.275, 1.345, "right", "center"), (1.34, 1.43, "left", "bottom")]
for k, (xk, (tx, ty, ha, va)) in enumerate(zip(KROKY, popisky)):
    ax.plot(*xk, "o", color=TEXT if k == 0 else CERV, ms=7, zorder=5)
    ax.text(tx, ty, f"x{'₀₁₂₃'[k]}, f = {cz(f(*xk), 4 if k else 0)}",
            color=TEXT if k == 0 else CERV, fontsize=11, fontweight="bold", ha=ha, va=va, bbox=BILE)
uloz(fig, "tabule-kroky.svg")


# ====================================================================== SVG grafu
W, VYSKA = 1000, 470
UZEL_W, UZEL_H = 176, 60
# stredy uzlu
POZ = {"x1": (110, 160), "x2": (110, 360), "q": (390, 160), "a": (390, 360),
       "r": (640, 360), "f": (880, 260)}
POPIS = {"x1": "x₁", "x2": "x₂", "q": "q = x₁ − 1", "a": "a = x₁ · x₂",
         "r": "r = a − 2", "f": "f = q² + r²"}
HRANY = [("x1", "q"), ("x1", "a"), ("x2", "a"), ("a", "r"), ("q", "f"), ("r", "f")]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def okraj(stred, smer):
    """Bod na okraji obdelniku uzlu ve smeru (dx, dy) od stredu."""
    cx, cy = stred
    dx, dy = smer
    sx = (UZEL_W / 2 + 4) / abs(dx) if dx else np.inf
    sy = (UZEL_H / 2 + 4) / abs(dy) if dy else np.inf
    t = min(sx, sy)
    return cx + dx * t, cy + dy * t


def usecka(od, do):
    (x0, y0), (x1, y1) = POZ[od], POZ[do]
    d = np.array([x1 - x0, y1 - y0], float)
    d /= np.hypot(*d)
    return okraj((x0, y0), d), okraj((x1, y1), -d)


def text(x, y, s, size=20, barva=TEXT, anchor="middle", weight="normal", halo=False, italic=False):
    h = ' paint-order="stroke" stroke="#fff" stroke-width="5" stroke-linejoin="round"' if halo else ""
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{barva}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}{h}>{esc(s)}</text>')


def sipka_svg(p0, p1, barva, sirka=2.6, carkovane=False):
    (x0, y0), (x1, y1) = p0, p1
    d = np.array([x1 - x0, y1 - y0], float)
    d /= np.hypot(*d)
    hl = 14
    bx, by = x1 - d[0] * hl, y1 - d[1] * hl
    n = np.array([-d[1], d[0]]) * hl * 0.45
    ca = ' stroke-dasharray="7 5"' if carkovane else ""
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{barva}" '
            f'stroke-width="{sirka}"{ca}/>'
            f'<polygon points="{x1:.1f},{y1:.1f} {bx + n[0]:.1f},{by + n[1]:.1f} '
            f'{bx - n[0]:.1f},{by - n[1]:.1f}" fill="{barva}"/>')


def uzel(jm, okraj_barva=TEXT, vypln="#fff"):
    cx, cy = POZ[jm]
    vstup = jm in ("x1", "x2")
    if jm == "f":
        vypln = "#e6f4f5"
        okraj_barva = AKC
    if vstup:
        vypln = "#f1f2f4"
    return (f'<rect x="{cx - UZEL_W / 2:.1f}" y="{cy - UZEL_H / 2:.1f}" width="{UZEL_W}" '
            f'height="{UZEL_H}" rx="12" fill="{vypln}" stroke="{okraj_barva}" stroke-width="2"/>'
            + text(cx, cy + 8, POPIS[jm], size=23, weight="600"))


def stitek(x, y, s, barva, size=19, anchor="middle", ramecek=None):
    """Hodnota u uzlu: barevny text, volitelne v ramecku."""
    out = ""
    if ramecek:
        sirka = 0.53 * size * len(s) + 22
        x0 = x - sirka / 2 if anchor == "middle" else (x if anchor == "start" else x - sirka)
        out += (f'<rect x="{x0:.1f}" y="{y - size - 5:.1f}" width="{sirka:.1f}" height="{size + 16}" '
                f'rx="6" fill="{ramecek}" stroke="{barva}" stroke-width="1.5"/>')
        x = x0 + sirka / 2
        anchor = "middle"
    return out + text(x, y, s, size=size, barva=barva, anchor=anchor, weight="600")


def svg(obsah, vyska=VYSKA, sirka=W):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sirka} {vyska}" '
            f'width="{sirka}" height="{vyska}" font-family="{PISMO}">'
            f'<rect width="{sirka}" height="{vyska}" fill="#fff"/>{obsah}</svg>\n')


def zapis(jmeno, obsah):
    (OBRAZKY / jmeno).write_text(obsah, encoding="utf-8")
    print("ulozeno", OBRAZKY / jmeno)


# --- 6) samotny graf
c = []
for od, do in HRANY:
    c.append(sipka_svg(*usecka(od, do), "#6b7280", 2.2))
c += [uzel(j) for j in POZ]
c.append(text(110, 60, "vstupy", 18, TLUM))
c.append(text(515, 60, "elementární operace (mezivýsledky)", 18, TLUM))
c.append(text(880, 190, "výstup", 18, TLUM))
c.append(text(W / 2, 445, "Každý uzel je jedna jednoduchá operace. Šipka = „tahle hodnota jde do výpočtu“.",
              18, TLUM))
zapis("teorie-graf.svg", svg("".join(c)))

# --- 7) dopredny pruchod
HODN = {"x1": BOD[0], "x2": BOD[1], **U}
c = []
for od, do in HRANY:
    c.append(sipka_svg(*usecka(od, do), MODR, 2.8))
c += [uzel(j, okraj_barva=MODR) if j != "f" else uzel(j) for j in POZ]
for j, (cx, cy) in POZ.items():
    pod = j in ("x2", "a", "r", "f")
    yy = cy + UZEL_H / 2 + 34 if pod else cy - UZEL_H / 2 - 16
    jmeno = {"x1": "x₁", "x2": "x₂"}.get(j, j)
    c.append(stitek(cx, yy, f"{jmeno} = {g(HODN[j])}", MODR, 21, ramecek="#eaf1f8"))
c.append(text(W / 2, 488, "Dopředu: dosadím x₁ = 1, x₂ = 1 a počítám uzel po uzlu zleva doprava. "
              "Mezivýsledky si pamatuju.", 18, TLUM))
zapis("teorie-dopredu.svg", svg("".join(c), vyska=505))

# --- 8) zpetny pruchod
LOKALNI = {("q", "f"): "2q = 0", ("r", "f"): "2r = −2", ("a", "r"): "1",
           ("x1", "q"): "1", ("x1", "a"): "x₂ = 1", ("x2", "a"): "x₁ = 1"}
POSUN_LOK = {("q", "f"): (0, -16), ("r", "f"): (-22, -14), ("a", "r"): (0, -14),
             ("x1", "q"): (0, -14), ("x1", "a"): (40, -4), ("x2", "a"): (0, -14)}
assert LOKALNI[("q", "f")].endswith(g(2 * U["q"])) and LOKALNI[("r", "f")].endswith(g(2 * U["r"]))
c = []
for od, do in HRANY:
    p0, p1 = usecka(od, do)
    dvojita = od == "x1"
    c.append(sipka_svg(p1, p0, CERV, 3.2 if dvojita else 2.8))
    mx, my = (p0[0] + p1[0]) / 2 + POSUN_LOK[(od, do)][0], (p0[1] + p1[1]) / 2 + POSUN_LOK[(od, do)][1]
    c.append(text(mx, my, LOKALNI[(od, do)], 18, "#7d2a14", halo=True, italic=True))
c += [uzel(j, okraj_barva=CERV) if j != "f" else uzel(j) for j in POZ]
ZP = {"f": f"∂f/∂f = {g(Z['df/df'])}", "q": f"∂f/∂q = {g(Z['df/dq'])}",
      "r": f"∂f/∂r = {g(Z['df/dr'])}", "a": f"∂f/∂a = {g(Z['df/da'])}",
      "x2": f"∂f/∂x₂ = {g(Z['df/da'])}·{g(BOD[0])} = {g(Z['df/dx2'])}"}
for j, s in ZP.items():
    cx, cy = POZ[j]
    if j == "x2":
        cx = 20 + (0.53 * 20 * len(s) + 22) / 2     # zarovnat k levemu okraji jako x1
    pod = j in ("x2", "a", "r", "f")
    yy = cy + UZEL_H / 2 + 34 if pod else cy - UZEL_H / 2 - 16
    c.append(stitek(cx, yy, s, CERV, 20, ramecek="#fbecE8".lower()))
# x1: soucet dvou cest, zvyrazneny
soucet = (f"∂f/∂x₁ = {g(Z['df/dq'])}·1 + ({g(Z['df/da'])})·{g(BOD[1])} = {g(Z['df/dx1'])}")
c.append(stitek(20, POZ["x1"][1] - UZEL_H / 2 - 18, soucet, CERV, 20, anchor="start", ramecek="#fff3c4"))
c.append(text(30, POZ["x1"][1] - UZEL_H / 2 - 60, "x₁ jde do q i do a → dvě cesty se sečtou", 18,
              "#7d2a14", anchor="start", weight="600"))
c.append(text(W / 2, 488, "Zpět: na hraně je lokální derivace (kurzívou), v uzlu ∂f/∂uzel "
              "= (∂f/∂následník) × (lokální derivace).", 18, TLUM))
zapis("teorie-zpet.svg", svg("".join(c), vyska=505))

# --- 9) forward vs reverse mode
def rezim(x0, reverse):
    out = []
    ys = [70, 125, 180, None, 290]
    jm = ["x₁", "x₂", "x₃", "⋮", "xₙ"]
    box = (x0 + 175, 100, 140, 170)
    fx, fy = x0 + 400, 185
    for y, n in zip(ys, jm):
        if y is None:
            out.append(text(x0 + 60, 245, "⋮", 26, TLUM))
            continue
        hl = (not reverse and n == "x₁") or reverse
        out.append(f'<circle cx="{x0 + 60}" cy="{y}" r="24" fill="{"#fff" if hl else "#f1f2f4"}" '
                   f'stroke="{(CERV if reverse else MODR) if hl else SEDA}" stroke-width="2"/>')
        out.append(text(x0 + 60, y + 7, n, 20, TEXT if hl else TLUM, weight="600"))
        if reverse:
            out.append(sipka_svg((box[0] - 4, box[1] + box[3] / 2 + (y - 185) * 0.45),
                                 (x0 + 88, y), CERV, 2.4))
        else:
            out.append(sipka_svg((x0 + 88, y), (box[0] - 4, box[1] + box[3] / 2 + (y - 185) * 0.45),
                                 MODR if hl else SEDA, 2.8 if hl else 1.4))
    out.append(f'<rect x="{box[0]}" y="{box[1]}" width="{box[2]}" height="{box[3]}" rx="12" '
               f'fill="#f4f7f8" stroke="{TEXT}" stroke-width="2"/>')
    out.append(text(box[0] + box[2] / 2, box[1] + 80, "výpočet f", 21, weight="600"))
    out.append(text(box[0] + box[2] / 2, box[1] + 106, "(graf)", 17, TLUM))
    out.append(f'<circle cx="{fx}" cy="{fy}" r="28" fill="#e6f4f5" stroke="{AKC}" stroke-width="2"/>')
    out.append(text(fx, fy + 8, "f", 23, weight="600", italic=True))
    if reverse:
        out.append(sipka_svg((fx - 32, fy), (box[0] + box[2] + 4, fy), CERV, 3))
        out.append(text(x0 + 230, 30, "reverse mode (zpětný průchod)", 22, CERV, weight="600"))
        out.append(text(x0 + 230, 345, "1 průchod zpět → všech n složek ∇f", 20, TEXT, weight="600"))
        out.append(text(x0 + 230, 372, "cena ≈ pár vyhodnocení f, nezávisle na n", 18, TLUM))
    else:
        out.append(sipka_svg((box[0] + box[2] + 4, fy), (fx - 32, fy), MODR, 3))
        out.append(text(x0 + 230, 30, "forward mode (dopředný průchod)", 22, MODR, weight="600"))
        out.append(text(x0 + 230, 345, "1 průchod → jen ∂f/∂x₁", 20, TEXT, weight="600"))
        out.append(text(x0 + 230, 372, "celý ∇f = n průchodů (jako diference)", 18, TLUM))
    return "".join(out)


c = [rezim(20, False), f'<line x1="500" y1="20" x2="500" y2="380" stroke="{SEDA}" stroke-dasharray="4 5"/>',
     rezim(520, True)]
zapis("teorie-rezimy.svg", svg("".join(c), vyska=395))
