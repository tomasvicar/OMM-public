"""Animace bloku „Omezení jako pokuta a zeď“ třetí přednášky: penalizace a bariéra.

Obě metody nahradí omezení pekárny členem v účelové funkci a řeší posloupnost
neomezených úloh s rostoucím parametrem (ρ u pokuty, t u bariéry):

    pokuta    min f(x) + ρ Σ max(0, gᵢ(x))²        → k hranici ZVENKU
    bariéra   min f(x) − (1/t) Σ log(−gᵢ(x))        → k hranici ZEVNITŘ

Čísla, která se objevují na slidech (spočítá je ``vypis_cisla()``):

    pokuta:   překročení mouky ≈ μ₁/(2ρ) = 8/ρ kg, odhad μ₁ = 2ρ·překročení → 16,
              číslo podmíněnosti Hessiánu ≈ 12ρ;
    bariéra:  rezerva mouky ≈ 1/(μ₁ t) = 1/(16 t) kg, odhad μ₁ = 1/(t·rezerva) → 16,
              ztráta proti optimu ≤ m/t = 4/t Kč (m = 4 nerovnosti).

Výstupy v ``prednasky/L3/obrazky/``:

    penalizace.gif      mapa s cestou penalizovaných řešení + překročení a odhad μ₁
    bariera-cesta.gif   mapa s vrstevnicemi bariérové funkce a centrální cestou
                        + rezerva mouky a záruka m/t

U obou animací je **první snímek výsledný stav** (záloha pro PDF a tisk)
a poslední snímek má dlouhou výdrž.

    cd prednasky/L3/kod && uv run python obrazky_blok7.py
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from spolecne import (
    A_PEKARNA,
    B_PEKARNA,
    C_PEKARNA,
    CERVENA,
    MODRA,
    ORANZOVA,
    Q_PEKARNA,
    SVETLE_MODRA,
    TYRKYSOVA,
    X_NEOMEZENE,
    X_QP,
    ZISK_QP,
    MU_QP,
    uloz_gif,
)

Q = np.diag(Q_PEKARNA)
C = np.array(C_PEKARNA)
A = np.array(A_PEKARNA)
B = np.array(B_PEKARNA)
X_STAR = np.array(X_QP)
MU1 = MU_QP[0]

# Bariéra potřebuje všechny čtyři nerovnosti, tedy i nezápornost.
A4 = np.vstack([A, [[-1.0, 0.0], [0.0, -1.0]]])
B4 = np.concatenate([B, [0.0, 0.0]])
M = len(B4)

VRCHOLY = [(0.0, 0.0), (160.0, 0.0), (100.0, 150.0), (0.0, 300.0)]
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8}
TMAVE_TYRKYSOVA = "#006a70"

RHO = np.geomspace(1e-3, 1e3, 25)   # pokuta: posloupnost parametrů
T = np.geomspace(2e-5, 1e2, 27)     # bariéra: posloupnost parametrů


def f(x: np.ndarray) -> float:
    """Účelová funkce pekárny ½xᵀQx + cᵀx (záporný zisk v Kč)."""
    return 0.5 * x @ Q @ x + C @ x


def penalizovane_reseni(rho: float) -> np.ndarray:
    """Minimum ½xᵀQx + cᵀx + ρ Σ max(0, aᵢᵀx − bᵢ)² (mouka a pec)."""
    x = np.array(X_NEOMEZENE)
    for _ in range(60):
        aktivni = A @ x - B > 0.0
        matice = Q + 2 * rho * A[aktivni].T @ A[aktivni]
        prava = -C + 2 * rho * A[aktivni].T @ B[aktivni]
        novy = np.linalg.solve(matice, prava)
        if np.allclose(novy, x, atol=1e-12):
            return novy
        x = novy
    return x


def _newton_bariera(t: float, start: np.ndarray, vaha_ucelu: float = 1.0) -> np.ndarray:
    """Minimum vaha·f(x) − (1/t) Σ log(−gᵢ(x)) tlumeným Newtonem."""
    x = start.copy()
    for _ in range(200):
        rezerva = B4 - A4 @ x
        grad = vaha_ucelu * (Q @ x + C) + (A4.T @ (1.0 / rezerva)) / t
        hess = vaha_ucelu * Q + (A4.T * (1.0 / rezerva**2)) @ A4 / t
        krok = -np.linalg.solve(hess, grad)
        if np.linalg.norm(krok) < 1e-12:
            break
        delka = 1.0
        while np.any(B4 - A4 @ (x + delka * krok) <= 1e-12):
            delka *= 0.5
        x = x + delka * krok
    return x


def analyticky_stred() -> np.ndarray:
    """Bod, kde je součet logaritmů rezerv největší (bariéra bez účelu)."""
    return _newton_bariera(1.0, np.array([50.0, 80.0]), vaha_ucelu=0.0)


def centralni_cesta(t_hodnoty) -> np.ndarray:
    """Body centrální cesty pro rostoucí t (teplý start z předchozího bodu)."""
    body, x = [], analyticky_stred()
    for t in t_hodnoty:
        x = _newton_bariera(t, x)
        body.append(x.copy())
    return np.array(body)


def carka(text: str) -> str:
    """Desetinná čárka v popiscích."""
    return text.replace(".", ",")


def cislo_rho(r: float) -> str:
    """Hezký zápis parametru: 0,001 · 0,1 · 3,2 · 100 · 1000."""
    if r < 1e-3:
        exponent = int(np.floor(np.log10(r)))
        return f"{r / 10**exponent:.0f}\\cdot 10^{{{exponent}}}"
    if r >= 10:
        return f"{r:.0f}"
    if r >= 1:
        return f"{r:.1f}".replace(".", "{,}")
    cifry = max(0, -int(np.floor(np.log10(r)))) + 1
    return f"{r:.{cifry}f}".replace(".", "{,}")


def cislo_kc(v: float) -> str:
    """Částka v Kč s rozumným počtem cifer a desetinnou čárkou."""
    if v >= 100:
        return f"{v:,.0f}".replace(",", " ")
    if v >= 1:
        return carka(f"{v:.1f}")
    return carka(f"{v:.2f}")


def _mapa(osa, xlim=(-6, 300), ylim=(-8, 312)) -> None:
    """Přípustná oblast pekárny, optimum a popisky omezení."""
    osa.add_patch(Polygon(VRCHOLY, closed=True, facecolor=SVETLE_MODRA,
                          edgecolor=TYRKYSOVA, linewidth=2.0, zorder=2))
    osa.scatter(*X_STAR, s=230, marker="*", color=CERVENA, edgecolor="white",
                linewidth=1.0, zorder=11)
    osa.text(154, 42, "mouka", color=TMAVE_TYRKYSOVA, fontsize=12, weight="bold",
             zorder=8, bbox=STITEK)
    osa.text(56, 236, "pec", color=TMAVE_TYRKYSOVA, fontsize=12, weight="bold",
             zorder=8, bbox=STITEK)
    osa.text(40, 120, "přípustná\noblast", color=TMAVE_TYRKYSOVA, fontsize=12, ha="center",
             zorder=4)
    osa.set_xlabel("chleby $x_1$ [ks/den]", fontsize=12)
    osa.set_ylabel("bagety $x_2$ [ks/den]", fontsize=12)
    osa.set(xlim=xlim, ylim=ylim)
    osa.tick_params(labelsize=11)
    osa.grid(alpha=0.18)


def _snimky(fig, vykresli, pocet: int):
    """Pořadí snímků: výsledný stav, pak celá posloupnost od začátku."""
    def zachyt(i):
        vykresli(i)
        fig.canvas.draw()
        return copy.deepcopy(fig)

    poradi = [pocet - 1] + list(range(pocet))
    return [zachyt(i) for i in poradi]


def penalizace_gif() -> None:
    """Pokuta: řešení se s rostoucím ρ stahuje k hranici mouky zvenku."""
    rho_vse = np.concatenate([[0.0], RHO])
    body = np.array([penalizovane_reseni(r) for r in rho_vse])
    prekroceni = A[0] @ body.T - B[0]
    odhad_mu = 2 * rho_vse * prekroceni

    fig = plt.figure(figsize=(7.8, 4.7))
    mrizka = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], hspace=0.72, wspace=0.3,
                              left=0.095, right=0.985, top=0.88, bottom=0.12)
    osa = fig.add_subplot(mrizka[:, 0])
    osa_v = fig.add_subplot(mrizka[0, 1])
    osa_mu = fig.add_subplot(mrizka[1, 1])

    # Mapa: vrstevnice penalizované funkce se s rostoucím ρ mačkají ke hraně.
    _mapa(osa)
    g1, g2 = np.meshgrid(np.linspace(-6, 300, 260), np.linspace(-8, 312, 260))
    f_mrizka = 0.5 * (Q[0, 0] * g1**2 + Q[1, 1] * g2**2) + C[0] * g1 + C[1] * g2
    porus = [np.maximum(0.0, A[i, 0] * g1 + A[i, 1] * g2 - B[i]) for i in range(2)]
    osa.text(X_NEOMEZENE[0] + 8, X_NEOMEZENE[1] + 12, "neomezené\noptimum", color="#52616b",
             fontsize=12, ha="right", va="bottom", zorder=7, bbox=STITEK)
    osa.scatter(*X_NEOMEZENE, s=40, color="#52616b", zorder=6)
    stopa, = osa.plot([], [], color=ORANZOVA, linewidth=2.6, zorder=7)
    bod = osa.scatter([], [], s=150, color=ORANZOVA, edgecolor=MODRA, linewidth=1.6, zorder=10)
    vrstevnice = []

    # Vpravo nahoře: překročení mouky klesá jako 1/ρ.
    kladne = rho_vse > 0
    osa_v.loglog(RHO, MU1 / (2 * RHO), color="#9aa7ae", linewidth=1.4, linestyle="--", zorder=2)
    osa_v.loglog(RHO, prekroceni[kladne], color=ORANZOVA, linewidth=2.4, alpha=0.35, zorder=3)
    osa_v.text(12.0, 1.2, "$\\approx \\mu_1/(2\\rho)$", color="#52616b", fontsize=12, zorder=5)
    osa_v.set(xlim=(8e-4, 1.3e3), ylim=(4e-3, 300))
    osa_v.set_title("překročení mouky [kg]", fontsize=13.5, color=MODRA)
    osa_v.set_xlabel("$\\rho$", fontsize=12.5, labelpad=0)
    bod_v = osa_v.scatter([], [], s=70, color=ORANZOVA, edgecolor=MODRA, zorder=6)

    # Vpravo dole: 2ρ·překročení je odhad multiplikátoru a míří k 16.
    osa_mu.axhline(MU1, color=CERVENA, linewidth=1.8, linestyle="--", zorder=2)
    osa_mu.text(1.2e-3, MU1 - 1.2, "stínová cena $\\mu_1=16$ Kč/kg", color=CERVENA, fontsize=12,
                va="top", zorder=5)
    osa_mu.semilogx(RHO, odhad_mu[kladne], color=ORANZOVA, linewidth=2.4, alpha=0.35, zorder=3)
    osa_mu.set(xlim=(8e-4, 1.3e3), ylim=(-1, 19.5))
    osa_mu.set_title("odhad $\\mu_1 = 2\\rho\\cdot$překročení", fontsize=13.5, color=MODRA)
    osa_mu.set_xlabel("$\\rho$", fontsize=12.5, labelpad=0)
    bod_mu = osa_mu.scatter([], [], s=70, color=ORANZOVA, edgecolor=MODRA, zorder=6)
    for o in (osa_v, osa_mu):
        o.tick_params(labelsize=10.5)
        o.grid(alpha=0.2, which="major")

    nadpis = fig.suptitle("", fontsize=15, color=MODRA, x=0.095, ha="left")

    def vykresli(i: int):
        for kolekce in vrstevnice:
            kolekce.remove()
        vrstevnice.clear()
        rho = rho_vse[i]
        hodnoty = f_mrizka + rho * (porus[0] ** 2 + porus[1] ** 2)
        minimum = f(body[i]) + rho * np.sum(np.maximum(0, A @ body[i] - B) ** 2)
        urovne = minimum + np.array([40.0, 160.0, 400.0, 800.0, 1400.0])
        vrstevnice.append(osa.contour(g1, g2, hodnoty, levels=urovne, colors=["#8a949b"],
                                      linewidths=1.0, linestyles=":", zorder=3))
        stopa.set_data(body[: i + 1, 0], body[: i + 1, 1])
        bod.set_offsets(body[i])
        if rho > 0:
            bod_v.set_offsets([rho, prekroceni[i]])
            bod_mu.set_offsets([rho, odhad_mu[i]])
            text_rho = f"$\\rho={cislo_rho(rho)}$"
        else:
            bod_v.set_offsets(np.empty((0, 2)))
            bod_mu.set_offsets(np.empty((0, 2)))
            text_rho = "$\\rho=0$ (žádná pokuta)"
        nadpis.set_text(carka(f"Pokuta   {text_rho}:   mouka {80 + prekroceni[i]:.2f} kg"
                              f"   (limit 80 kg)"))

    snimky = _snimky(fig, vykresli, len(rho_vse))
    doby = [2600] + [900] + [400] * (len(rho_vse) - 2) + [3600]
    uloz_gif(snimky, doby, "penalizace", dpi=100)
    plt.close(fig)


def bariera_gif() -> None:
    """Bariéra: řešení jde z analytického středu po centrální cestě do optima."""
    cesta = centralni_cesta(T)
    stred = analyticky_stred()
    rezerva = B[0] - A[0] @ cesta.T
    ztrata = np.array([f(x) for x in cesta]) + ZISK_QP

    fig = plt.figure(figsize=(7.8, 4.7))
    mrizka = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], hspace=0.72, wspace=0.3,
                              left=0.095, right=0.985, top=0.88, bottom=0.12)
    osa = fig.add_subplot(mrizka[:, 0])
    osa_r = fig.add_subplot(mrizka[0, 1])
    osa_g = fig.add_subplot(mrizka[1, 1])

    _mapa(osa, xlim=(-6, 215), ylim=(-8, 312))
    osa.texts[-1].set_visible(False)  # popisek oblasti by překážel vrstevnicím
    g1, g2 = np.meshgrid(np.linspace(0.05, 159.95, 320), np.linspace(0.05, 299.95, 320))
    rezervy = np.stack([B4[i] - A4[i, 0] * g1 - A4[i, 1] * g2 for i in range(M)])
    uvnitr = np.all(rezervy > 1e-9, axis=0)
    logy = -np.sum(np.log(np.where(uvnitr, rezervy, 1.0)), axis=0)
    f_mrizka = 0.5 * (Q[0, 0] * g1**2 + Q[1, 1] * g2**2) + C[0] * g1 + C[1] * g2
    osa.scatter(*stred, s=55, color=MODRA, zorder=8)
    osa.annotate("analytický střed\n(začátek, $t\\to0$)", xy=stred, xytext=(6, 8),
                 fontsize=12, color=MODRA, zorder=9, bbox=STITEK,
                 arrowprops={"arrowstyle": "->", "color": MODRA})
    osa.annotate("$\\mathbf{x}^\\star=(120,100)$", xy=(122, 101), xytext=(212, 150),
                 fontsize=12, color=CERVENA, weight="bold", ha="right", zorder=9, bbox=STITEK,
                 arrowprops={"arrowstyle": "->", "color": CERVENA})
    stopa, = osa.plot([], [], color=TYRKYSOVA, linewidth=2.8, zorder=7)
    bod = osa.scatter([], [], s=150, color="#5fc4c9", edgecolor=MODRA, linewidth=1.6, zorder=10)
    vrstevnice = []

    # Vpravo nahoře: rezerva mouky klesá jako 1/t — ale nikdy není záporná.
    osa_r.loglog(T, 1 / (MU1 * T), color="#9aa7ae", linewidth=1.4, linestyle="--", zorder=2)
    osa_r.loglog(T, rezerva, color=TYRKYSOVA, linewidth=2.4, alpha=0.35, zorder=3)
    osa_r.text(3e-2, 2.2, "$\\approx 1/(\\mu_1 t)$", color="#52616b", fontsize=12, zorder=5)
    osa_r.set(xlim=(1e-5, 2e2), ylim=(2e-4, 150))
    osa_r.set_title("rezerva mouky [kg]", fontsize=13.5, color=MODRA)
    osa_r.set_xlabel("$t$", fontsize=12.5, labelpad=0)
    bod_r = osa_r.scatter([], [], s=70, color="#5fc4c9", edgecolor=MODRA, zorder=6)

    # Vpravo dole: záruka m/t (známe hned) proti skutečné ztrátě (známe až zpětně).
    osa_g.loglog(T, M / T, color=CERVENA, linewidth=2.0, zorder=3)
    osa_g.loglog(T, ztrata, color=TYRKYSOVA, linewidth=2.4, alpha=0.35, zorder=3)
    osa_g.text(1e-2, 1.5e3, "záruka $m/t=4/t$", color=CERVENA, fontsize=12, zorder=5, va="bottom",
               bbox=STITEK)
    osa_g.text(1.5e-5, 1e-2, "skutečná ztráta", color=TMAVE_TYRKYSOVA, fontsize=12, zorder=5,
               bbox=STITEK)
    osa_g.set(xlim=(1e-5, 2e2), ylim=(3e-3, 5e5))
    osa_g.set_title("vzdálenost od optima [Kč]", fontsize=13.5, color=MODRA)
    osa_g.set_xlabel("$t$", fontsize=12.5, labelpad=0)
    bod_g = osa_g.scatter([], [], s=70, color="#5fc4c9", edgecolor=MODRA, zorder=6)
    for o in (osa_r, osa_g):
        o.tick_params(labelsize=10.5)
        o.grid(alpha=0.2, which="major")

    nadpis = fig.suptitle("", fontsize=15, color=MODRA, x=0.095, ha="left")

    def vykresli(i: int):
        for kolekce in vrstevnice:
            kolekce.remove()
        vrstevnice.clear()
        t = T[i]
        hodnoty = np.where(uvnitr, f_mrizka + logy / t, np.nan)
        urovne = np.nanquantile(hodnoty, [0.015, 0.05, 0.12, 0.24, 0.42, 0.65])
        vrstevnice.append(osa.contour(g1, g2, hodnoty, levels=urovne, colors=[TYRKYSOVA],
                                      linewidths=1.0, alpha=0.55, zorder=3))
        stopa.set_data(cesta[: i + 1, 0], cesta[: i + 1, 1])
        bod.set_offsets(cesta[i])
        bod_r.set_offsets([t, rezerva[i]])
        bod_g.set_offsets([t, ztrata[i]])
        nadpis.set_text(f"Bariéra   $t={cislo_rho(t)}$:   "
                        + carka(f"mouka {80 - rezerva[i]:.3f} kg,   ztráta ")
                        + f"$\\leq$ {cislo_kc(M / t)} Kč")

    snimky = _snimky(fig, vykresli, len(T))
    doby = [2600] + [900] + [400] * (len(T) - 2) + [3600]
    uloz_gif(snimky, doby, "bariera-cesta", dpi=100)
    plt.close(fig)


def vypis_cisla() -> None:
    """Čísla použitá na slidech a ve widgetu (kontrola, nic se neukládá)."""
    print("pokuta:  rho  x  překročení  2rho*v  cond")
    for rho in (0.1, 1.0, 10.0, 100.0, 1000.0):
        x = penalizovane_reseni(rho)
        v = A[0] @ x - B[0]
        hess = Q + 2 * rho * np.outer(A[0], A[0])
        print(f"  {rho:7g}  {x.round(3)}  {v:.4f}  {2 * rho * v:.3f}  {np.linalg.cond(hess):.0f}")
    print("bariéra: t  x  rezerva  1/(t*rez)  f-p*  m/t")
    cesta = centralni_cesta(np.geomspace(2e-5, 1000.0, 400))
    for t in (0.01, 1.0, 10.0, 100.0, 1000.0):
        x = cesta[np.argmin(np.abs(np.geomspace(2e-5, 1000.0, 400) - t))]
        x = _newton_bariera(t, x)
        r = B[0] - A[0] @ x
        print(f"  {t:7g}  {x.round(4)}  {r:.6f}  {1 / (t * r):.3f}  {f(x) + ZISK_QP:.4f}  {M / t:.4f}")
    print("analytický střed", analyticky_stred().round(2))


if __name__ == "__main__":
    vypis_cisla()
    penalizace_gif()
    bariera_gif()
