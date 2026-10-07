"""Obrázky pro blok „KKT: pravidla pro správnou cenu“ třetí přednášky.

Spuštění:

    cd prednasky/L3/kod && uv run python obrazky_blok5.py

Volitelně lze do proměnné prostředí ``OMM_NAHLEDY`` dát adresář, kam se
kromě SVG uloží i rastrový náhled pro rychlou vizuální kontrolu.
"""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

from spolecne import (
    OUT,
    SVG_METADATA,
    uloz_gif,
    A_PEKARNA,
    B_PEKARNA,
    Q_PEKARNA,
    C_PEKARNA,
    MODRA,
    TYRKYSOVA,
    CERVENA,
    ORANZOVA,
    SVETLE_MODRA,
    SEDIVA,
)

plt.rcParams["figure.max_open_warning"] = 0   # snímky GIFu se zavírají až po uložení
_NAHLEDY = os.environ.get("OMM_NAHLEDY")
STITEK = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8}
ZELENA = "#3a8a52"


def uloz(fig: plt.Figure, jmeno: str) -> None:
    """Uloží obrázek jako SVG a případně i jako PNG náhled."""
    fig.savefig(OUT / f"{jmeno}.svg", metadata=SVG_METADATA, bbox_inches="tight")
    if _NAHLEDY:
        Path(_NAHLEDY).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(_NAHLEDY) / f"{jmeno}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)




def znamenko_mu() -> None:
    """Proč musí být μ nezáporné: směr zlepšení musí mířit ven z oblasti."""
    fig, osy = plt.subplots(1, 2, figsize=(6.4, 2.8))

    t = np.linspace(0.0, 4.0, 50)
    normala = np.array([0.6, 0.8])          # jednotková vnější normála, tedy směr ∇g
    tecna = np.array([0.8, -0.6])
    bod = np.array([1.55, 2.85])            # bod na hranici 0,6·x₁ + 0,8·x₂ = 3,2
    bod_g = bod + 1.15 * tecna              # ∇g kreslíme opodál, ať se šipky nekryjí

    varianty = (
        ("$\\mu>0$  —  tady se opravdu zastavíme", TYRKYSOVA, 1.0),
        ("$\\mu<0$  —  to nemůže být optimum", CERVENA, -1.0),
    )

    for osa, (nadpis, barva, znamenko) in zip(osy, varianty):
        hranice = (3.2 - 0.6 * t) / 0.8
        osa.fill_between(t, 0.0, hranice, color=SVETLE_MODRA, zorder=1)
        osa.plot(t, hranice, color=TYRKYSOVA, linewidth=2.4, zorder=3)
        osa.text(0.12, 1.28, "přípustná oblast\n$g(\\mathbf{x})\\leq0$", color="#0d6a70",
                 fontsize=10.5, va="top", zorder=4)

        # vnější normála ∇g je v obou panelech stejná
        osa.add_patch(FancyArrowPatch(bod_g, bod_g + 0.85 * normala, arrowstyle="-|>",
                                      mutation_scale=15, linewidth=2.2,
                                      color=ORANZOVA, zorder=6))
        osa.text(*(bod_g + 0.95 * normala), "$\\nabla g$  ven", color="#9a6200",
                 fontsize=11.5, weight="bold", ha="center", va="bottom", zorder=7)

        # ∇f = −μ∇g, směr zlepšení je −∇f
        grad_f = -znamenko * 0.95 * normala
        osa.add_patch(FancyArrowPatch(bod, bod + grad_f, arrowstyle="-|>",
                                      mutation_scale=15, linewidth=2.6,
                                      color=MODRA, zorder=6))
        osa.add_patch(FancyArrowPatch(bod, bod - 0.95 * grad_f, arrowstyle="-|>",
                                      mutation_scale=13, linewidth=2.0,
                                      color=barva, linestyle=(0, (4, 2)), zorder=5))
        osa.scatter(*bod, s=110, color=CERVENA, edgecolor="white", linewidth=1.2, zorder=8)

        if znamenko > 0:
            osa.text(0.70, 1.92, "$\\nabla f$", color=MODRA, fontsize=12.5, weight="bold",
                     ha="right", va="center", zorder=7)
            osa.text(2.28, 3.70, "$-\\nabla f$ míří ven", color=barva, fontsize=11,
                     weight="bold", ha="center", va="center", zorder=7, bbox=STITEK)
            zprava = "zlepšit by šlo jen ven\nz oblasti — tam se ale nesmí"
        else:
            osa.text(2.30, 3.72, "$\\nabla f$", color=MODRA, fontsize=12.5, weight="bold",
                     ha="center", va="center", zorder=7)
            osa.text(0.70, 1.92, "$-\\nabla f$", color=barva, fontsize=11.5,
                     weight="bold", ha="right", va="center", zorder=7)
            zprava = "zlepšení míří dovnitř —\nještě se dá sejít níž"
        osa.text(2.05, 0.34, zprava, color=barva, fontsize=10.5, weight="bold",
                 ha="center", va="center", zorder=7,
                 bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                       "edgecolor": barva, "linewidth": 1.2})

        osa.set(xlim=(0.0, 4.0), ylim=(0.0, 4.0), aspect="equal", xticks=[], yticks=[])
        osa.set_title(nadpis, fontsize=12, color=barva, pad=6)
        for strana in osa.spines.values():
            strana.set_color("#cbd5da")

    fig.tight_layout()
    uloz(fig, "znamenko-mu")


# --- pekárna s proměnnou kapacitou: přesné KKT řešení přes aktivní množiny ---
_A4 = np.vstack([np.array(A_PEKARNA), [[-1.0, 0.0], [0.0, -1.0]]])
_Q_INV = np.diag(1.0 / np.array(Q_PEKARNA))
_C = np.array(C_PEKARNA)


def kkt_pekarna(b1: float, b2: float) -> tuple[np.ndarray, np.ndarray]:
    """Vrátí optimum a multiplikátory (mouka, pec, x1>=0, x2>=0) pekárny.

    Ruční postup z přednášky naprogramovaný doslova: zkouší aktivní množiny
    (nejvýš dvě omezení, víc jich ve 2D nemá smysl), z rovnic dopočítá x a μ
    a vezme tu, kde je x přípustné a všechna μ nezáporná. Úloha je striktně
    konvexní, takže takový bod existuje právě jeden.
    """
    b4 = np.array([b1, b2, 0.0, 0.0])
    mnoziny = [()] + [(i,) for i in range(4)] + [(i, j) for i in range(4)
                                                 for j in range(i + 1, 4)]
    for aktivni in mnoziny:
        mu = np.zeros(4)
        if aktivni:
            a_s = _A4[list(aktivni)]
            matice = a_s @ _Q_INV @ a_s.T
            if abs(np.linalg.det(matice)) < 1e-12:
                continue
            mu[list(aktivni)] = np.linalg.solve(matice, a_s @ _Q_INV @ (-_C) - b4[list(aktivni)])
        x = _Q_INV @ (-_C - _A4.T @ mu)
        if mu.min() >= -1e-9 and np.all(_A4 @ x <= b4 + 1e-7):
            return x, mu
    raise RuntimeError("KKT bod nenalezen")


def oblast_pekarny(b1: float, b2: float) -> np.ndarray:
    """Vrcholy přípustné oblasti pekárny (ořez čtverce polorovinami)."""
    body = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
    for a, b in zip(_A4, (b1, b2, 0.0, 0.0)):
        novy = []
        for p, q in zip(body, np.roll(body, -1, axis=0)):
            gp, gq = a @ p - b, a @ q - b
            if gp <= 0:
                novy.append(p)
            if gp * gq < 0:
                novy.append(p + gp / (gp - gq) * (q - p))
        body = np.array(novy)
    return body


def _kontrola_kkt() -> None:
    """Porovná ruční aktivní množiny s CVXPY na několika kapacitách pece."""
    for b2 in (600.0, 545.0, 480.0):
        x_var = cp.Variable(2)
        zdroje = np.array(A_PEKARNA) @ x_var <= np.array([80.0, b2])
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x_var, np.diag(Q_PEKARNA)) + _C @ x_var),
                   [zdroje, x_var >= 0]).solve()
        x, mu = kkt_pekarna(80.0, b2)
        assert np.allclose(x, x_var.value, atol=1e-3), (b2, x, x_var.value)
        assert np.allclose(mu[:2], zdroje.dual_value, atol=1e-3), (b2, mu, zdroje.dual_value)


def _cislo(hodnota: float, mist: int = 1) -> str:
    """Číslo s desetinnou čárkou; záporná nula se nezobrazuje."""
    if abs(hodnota) < 0.5 * 10**-mist:
        hodnota = 0.0
    return f"{hodnota:.{mist}f}".replace(".", "{,}")


def komplementarita_pec_gif() -> None:
    """Pec se zmenšuje: napne se, cena se přesune z mouky na pec.

    První snímek je skutečná pekárna (600 min), aby obrázek dával smysl i v PDF.
    """
    b2_osa = np.linspace(600.0, 440.0, 321)
    mu_osa = np.array([kkt_pekarna(80.0, b)[1][:2] for b in b2_osa])
    x_osa = np.array([kkt_pekarna(80.0, b)[0] for b in b2_osa])
    b2_obe_od, b2_obe_do = 560.0, 532.0 + 4.0 / 23.0   # obě omezení napnutá

    mrizka1, mrizka2 = np.meshgrid(np.linspace(0, 230, 240), np.linspace(0, 230, 240))
    zisk_mrizka = (14 * mrizka1 - 0.025 * mrizka1**2 + 8 * mrizka2 - 0.024 * mrizka2**2)
    t = np.linspace(0, 240, 50)

    def snimek(b2: float) -> plt.Figure:
        x, mu = kkt_pekarna(80.0, b2)
        g = np.array(A_PEKARNA) @ x - np.array([80.0, b2])
        zisk_opt = float(14 * x[0] - 0.025 * x[0]**2 + 8 * x[1] - 0.024 * x[1]**2)

        fig = plt.figure(figsize=(8.6, 4.1))
        mrizka = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.3), hspace=0.42,
                                  wspace=0.26, left=0.085, right=0.985, top=0.915, bottom=0.13)
        osa = fig.add_subplot(mrizka[:, 0])
        osa_m = fig.add_subplot(mrizka[0, 1])
        osa_p = fig.add_subplot(mrizka[1, 1], sharex=osa_m)

        # --- rovina výroby ---
        vrcholy = oblast_pekarny(80.0, b2)
        osa.add_patch(Polygon(vrcholy, closed=True, facecolor=SVETLE_MODRA,
                              edgecolor="none", zorder=1))
        osa.contour(mrizka1, mrizka2, zisk_mrizka, levels=[1500, 1700, 2000, 2200],
                    colors=[MODRA], linewidths=0.9, linestyles=":", zorder=2)
        osa.contour(mrizka1, mrizka2, zisk_mrizka, levels=[zisk_opt], colors=[ORANZOVA],
                    linewidths=2.2, zorder=3)
        osa.plot(t, (80 - 0.5 * t) / 0.2, color=CERVENA, linewidth=2.6, zorder=4)
        osa.plot(t, (b2 - 3 * t) / 2, color=ZELENA, linewidth=2.6, zorder=4)
        osa.plot(t, (600 - 3 * t) / 2, color=ZELENA, linewidth=1.0, linestyle="--",
                 alpha=0.45, zorder=3)
        osa.plot(x_osa[:, 0], x_osa[:, 1], color="#8a97a0", linewidth=1.2,
                 linestyle=(0, (2, 2)), zorder=5)
        osa.scatter(*x, s=260, marker="*", color=CERVENA, edgecolor="white",
                    linewidth=1.1, zorder=8)
        osa.text(150, 214, "mouka", color=CERVENA, fontsize=11, weight="bold",
                 ha="left", va="top", zorder=9, bbox=STITEK)
        osa.text(8, 8, f"pec: {b2:.0f} min", color=ZELENA, fontsize=11,
                 weight="bold", va="bottom", zorder=9, bbox=STITEK)
        osa.set_title(f"optimum: zisk {zisk_opt:.0f} Kč", fontsize=11.5, color="#9a6200",
                      weight="bold", loc="left", pad=4)
        osa.set(xlim=(0, 230), ylim=(0, 230),
                xlabel="chleby $x_1$", ylabel="bagety $x_2$")
        osa.tick_params(labelsize=10)
        osa.grid(alpha=0.2)

        # --- ceny obou omezení ---
        for osa_mu, i, barva, nazev, jednotka, zasoba in (
                (osa_m, 0, CERVENA, "mouka", "Kč/kg", "kg"),
                (osa_p, 1, ZELENA, "pec", "Kč/min", "min")):
            volna = np.abs(mu_osa[:, i]) < 1e-9
            rezerva = -g[i]
            osa_mu.fill_between(b2_osa, 0, 1, where=volna, transform=osa_mu.get_xaxis_transform(),
                                color=SEDIVA, zorder=0)
            osa_mu.plot(b2_osa, mu_osa[:, i], color=barva, linewidth=2.4, zorder=3)
            osa_mu.scatter([b2], [mu[i]], s=70, color=barva, edgecolor="white",
                           linewidth=1.2, zorder=5)
            osa_mu.axvline(b2, color="#5c6672", linewidth=1.0, zorder=2)
            stav = "napnutá" if abs(g[i]) < 1e-6 else f"volná (rezerva {_cislo(rezerva, 0)} {zasoba})"
            text = f"{nazev}: {stav},  $\\mu_{i + 1}={_cislo(mu[i], 1)}$ {jednotka}"
            osa_mu.set_title(text, fontsize=10.5, color=barva, weight="bold", loc="left", pad=4)
            osa_mu.set_ylim(-0.08 * mu_osa[:, i].max(), 1.18 * mu_osa[:, i].max())
            osa_mu.tick_params(labelsize=10)
            osa_mu.grid(alpha=0.2)
        osa_m.text(487, 0.5 * mu_osa[:, 0].max(), "šedě: omezení volné\n$\\Rightarrow$ cena nula",
                   color="#5c6672", fontsize=10, va="center", ha="center", zorder=6)
        for hranice in (b2_obe_od, b2_obe_do):
            for osa_mu in (osa_m, osa_p):
                osa_mu.axvline(hranice, color="#8a97a0", linewidth=0.8, linestyle=":", zorder=1)
        osa_p.set_xlim(605, 435)
        osa_p.set_xlabel("kapacita pece $b_2$ [min]  (pec se zmenšuje →)", fontsize=10)
        plt.setp(osa_m.get_xticklabels(), visible=False)
        return fig

    doba = 85
    cesta = list(np.linspace(600.0, 450.0, 38)) + list(np.linspace(450.0, 600.0, 16))[1:]
    snimky, doby = [], []
    for i, b2 in enumerate(cesta):
        snimky.append(snimek(b2))
        doby.append(3000 if i == 0 else (1600 if abs(b2 - 450.0) < 1e-9 else doba))
    doby[-1] = 2500
    uloz_gif(snimky, doby, "komplementarita-pec", dpi=100)


# --- KKT ručně: tipni aktivní množinu, dopočítej, ověř ---
def kkt_davka() -> None:
    """Nejmenší KKT příklad: min (x - 3)^2 za x <= b, pro limit b = 2 a b = 5."""
    fig, osy = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
    x = np.linspace(0.0, 6.0, 300)
    varianty = (
        (2.0, "limit 2 mg: napnutý, $\\mu=2$", CERVENA),
        (5.0, "limit 5 mg: volný, $\\mu=0$", TYRKYSOVA),
    )
    for osa, (b, nadpis, barva) in zip(osy, varianty):
        osa.axvspan(0.0, b, color=SVETLE_MODRA, zorder=0)
        osa.axvline(b, color=CERVENA, linewidth=2.0, zorder=2)
        osa.text(b - 0.1, -1.2, f"limit {b:.0f}", color=CERVENA, fontsize=10.5,
                 weight="bold", ha="right", va="bottom", zorder=5, bbox=STITEK)
        osa.text(b / 2, 8.3, "smí se", color="#0d6a70", fontsize=10.5, ha="center",
                 va="top", zorder=5, bbox=STITEK)
        osa.plot(x, (x - 3) ** 2, color=MODRA, linewidth=2.4, zorder=3)
        osa.scatter([3], [0], s=70, color="#8a97a0", edgecolor="white", zorder=4)
        osa.text(3, -0.45, "ideál 3", color="#5c6672", fontsize=10, ha="center",
                 va="top", zorder=5)
        xo = min(b, 3.0)
        if b < 3:
            # tečna v optimu: sklon f'(2) = -2, tedy mu = 2
            t = np.linspace(xo - 1.0, xo + 0.6, 2)
            osa.plot(t, (xo - 3) ** 2 + 2 * (xo - 3) * (t - xo), color=ORANZOVA,
                     linewidth=2.0, linestyle=(0, (4, 2)), zorder=4)
            osa.text(xo - 1.05, 3.55, "sklon $-2$", color="#9a6200", fontsize=10.5,
                     weight="bold", ha="left", va="bottom", zorder=6, bbox=STITEK)
        osa.scatter([xo], [(xo - 3) ** 2], s=200, marker="*", color=barva,
                    edgecolor="white", linewidth=1.0, zorder=6)
        if b < 3:
            osa.text(xo + 0.25, (xo - 3) ** 2 + 0.25, "odchylka 1", color=barva, fontsize=10,
                     weight="bold", ha="left", va="bottom", zorder=6, bbox=STITEK)
        osa.set_title(nadpis, fontsize=11.5, color=barva, pad=5)
        osa.set(xlim=(0, 6), ylim=(-1.4, 8.6), xlabel="dávka $x$ [mg]")
        osa.tick_params(labelsize=9.5)
        osa.grid(alpha=0.2)
    osy[0].set_ylabel("odchylka $(x-3)^2$")
    fig.tight_layout()
    uloz(fig, "kkt-davka")


def _aktivni_mnozina(aktivni: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Vyřeší KKT soustavu pekárny, když jsou napnutá právě omezení `aktivni`."""
    A = np.array(A_PEKARNA)[list(aktivni)]
    b = np.array(B_PEKARNA)[list(aktivni)]
    Q = np.diag(Q_PEKARNA)
    n = len(aktivni)
    K = np.block([[Q, A.T], [A, np.zeros((n, n))]])
    reseni = np.linalg.solve(K, np.concatenate([-_C, b]))
    return reseni[:2], reseni[2:]


def kkt_pekarna_tipy() -> None:
    """Čtyři tipy aktivní množiny pekárny; projde jen jeden."""
    fig, osa = plt.subplots(figsize=(5.4, 3.7))
    osa.add_patch(Polygon([[0, 0], [160, 0], [100, 150], [0, 300]], closed=True,
                          facecolor=SVETLE_MODRA, edgecolor="none", zorder=1))
    t = np.linspace(0, 320, 40)
    osa.plot(t, (80 - 0.5 * t) / 0.2, color=CERVENA, linewidth=2.2, zorder=3)
    osa.plot(t, (600 - 3 * t) / 2, color=ZELENA, linewidth=2.2, zorder=3)
    osa.text(84, 205, "mouka", color=CERVENA, fontsize=10.5, weight="bold", zorder=6,
             bbox=STITEK)
    osa.text(212, 22, "pec", color=ZELENA, fontsize=10.5, weight="bold", zorder=6,
             bbox=STITEK)
    # vrstevnice zisku kolem neomezeného maxima
    g1, g2 = np.meshgrid(np.linspace(0, 320, 300), np.linspace(0, 240, 300))
    z = 14 * g1 - 0.025 * g1**2 + 8 * g2 - 0.024 * g2**2
    osa.contour(g1, g2, z, levels=[1400, 1880, 2300], colors=MODRA, linewidths=0.9,
                linestyles=":", zorder=2)

    tipy = (
        ((), "A", "nic napnuté\nmouky chybí 93 kg ✗", (-8, 10), "right"),
        ((0,), "B", "jen mouka\n$\\mu_1=16$ ✓", (-12, -16), "right"),
        ((1,), "C", "jen pec\nmouky chybí 10 kg ✗", (10, -12), "left"),
        ((0, 1), "D", "obě\n$\\mu_2=-3{,}5$ ✗", (-10, 4), "right"),
    )
    for aktivni, pismeno, popis, posun, zarovnani in tipy:
        x, mu = _aktivni_mnozina(aktivni)
        ok = bool(np.all(mu >= -1e-9) and np.all(np.array(A_PEKARNA) @ x <= np.array(B_PEKARNA) + 1e-7))
        barva = TYRKYSOVA if ok else "#8a97a0"
        osa.scatter(*x, s=230 if ok else 90, marker="*" if ok else "o", color=barva,
                    edgecolor="white", linewidth=1.2, zorder=7)
        osa.annotate(f"{pismeno}: {popis}", xy=x, xytext=posun, textcoords="offset points",
                     ha=zarovnani, va="center", fontsize=9.5, weight="bold",
                     color=TYRKYSOVA if ok else "#4a5560", zorder=8, bbox=STITEK)
    osa.set(xlim=(0, 300), ylim=(0, 220), xlabel="chleby $x_1$", ylabel="bagety $x_2$")
    osa.tick_params(labelsize=9.5)
    osa.grid(alpha=0.2)
    fig.tight_layout()
    uloz(fig, "kkt-pekarna-tipy")


def _kontrola_tipu() -> None:
    """Čísla ze slidu: projde jen tip B, D je vrchol LP z druhé přednášky."""
    x, mu = _aktivni_mnozina(())
    assert np.allclose(x, (280, 500 / 3)) and abs(0.5 * x[0] + 0.2 * x[1] - 80 - 93.33) < 0.01
    x, mu = _aktivni_mnozina((0,))
    assert np.allclose(x, (120, 100)) and np.allclose(mu, 16)
    x, mu = _aktivni_mnozina((1,))
    assert np.allclose(x, (149.367, 75.949), atol=1e-3) and abs(0.5 * x[0] + 0.2 * x[1] - 89.87) < 0.01
    x, mu = _aktivni_mnozina((0, 1))
    assert np.allclose(x, (100, 150)) and np.allclose(mu, (39, -3.5))


if __name__ == "__main__":
    _kontrola_kkt()
    znamenko_mu()
    komplementarita_pec_gif()
    _kontrola_tipu()
    kkt_davka()
    kkt_pekarna_tipy()
