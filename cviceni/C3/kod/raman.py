"""Samostatna uloha tretiho cviceni: odecteni pozadi Ramanova spektra (QP).

    uv run python cviceni/C3/kod/raman.py

Spektrum ma 300 hodnot na vlnoctech 400-1800 cm^-1 (seed 1). Pod uzkymi piky
lezi siroke fluorescencni pozadi. Hledame pozadi z, ktere
  - nikde nelezi nad spektrem          -> omezeni  z <= y
  - je spektru co nejbliz              -> plocha mezi krivkami 1^T (y - z)
                                          (diky omezeni je y - z >= 0, abs. hodnota netreba)
  - je hladke                          -> male druhe diference, alfa * ||D2 z||^2

    minimize   1^T (y - z) + alfa * ||D2 z||_2^2
    subject to z <= y                                    (alfa = 1e4)

KKT: stacionarita -1 + 2 alfa D2^T D2 z + mu = 0. D2 nuluje konstantu
i primku, takze po vynasobeni 1^T a i^T vyjde sum(mu) = n = 300 a
sum(i * mu_i) = sum(i) = 44 850 (indexy 0..299) -- pro kazde alfa.
Multiplikatory jsou nenulove jen v bodech dotyku (komplementarita).

Skript je jediny zdroj cisel pro report raman.html, notebook raman.ipynb
a obrazky raman-*.svg.

Vaha hladkosti se jmenuje alfa (jako u ridge v prednasce), ne lambda: lambda je
ve cviceni multiplikator rovnosti. Pozadi je z, protoze b je v praporku vektor dat.
"""

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

OBRAZKY = Path("cviceni/C3/obrazky")
SVG = {"Date": None}
plt.rcParams.update({"svg.hashsalt": "mpc-omm-c3-raman", "svg.fonttype": "path",
                     "font.size": 10})

# ------------------------------------------------------------------ data ----
N = 300
ALFA = 1e4
PIKY = [(1003, 4.0, 8), (1250, 1.5, 25), (1450, 2.5, 15), (1660, 2.0, 20)]  # stred, vyska, sigma
PRAH_MU = 0.1        # dotyk = multiplikator nad prahem (mezera mezi 0,02 a 1,2 pri alfa = 1e4)


def data():
    """Presne poradi generovani ze spolecneho zadani (seed 1)."""
    v = np.linspace(400, 1800, N)
    rng = np.random.default_rng(1)
    pozadi = 3 * np.exp(-(v - 400) / 900) + 0.8
    piky = sum(h * np.exp(-(v - c) ** 2 / (2 * s ** 2)) for c, h, s in PIKY)
    y = pozadi + piky + rng.normal(0, 0.05, N)
    return v, y, pozadi


V, Y, POZADI_PRAVDA = data()
D2 = np.diff(np.eye(N), 2, axis=0)
INDEXY_PIKU = [int(np.argmin(np.abs(V - c))) for c, _, _ in PIKY]


def pod_signalem(alfa=ALFA):
    """Spravny model: pozadi pod spektrem. Vraci (z, mu)."""
    z = cp.Variable(N)
    pod = z <= Y
    cil = cp.sum(Y - z) + alfa * cp.sum_squares(D2 @ z)
    cp.Problem(cp.Minimize(cil), [pod]).solve(solver=cp.CLARABEL)
    return z.value, pod.dual_value


def bez_omezeni(alfa=ALFA):
    """Obycejne vyhlazeni nejmensimi ctverci: pozadi jde prostredkem."""
    z = cp.Variable(N)
    cp.Problem(cp.Minimize(cp.sum_squares(Y - z) + alfa * cp.sum_squares(D2 @ z))).solve(
        solver=cp.CLARABEL)
    return z.value


def rmse(z):
    return float(np.sqrt(np.mean((z - POZADI_PRAVDA) ** 2)))


def vysky(z):
    """Vyska piku = hodnota korigovaneho spektra v nejblizsim bode ke stredu."""
    return (Y - z)[INDEXY_PIKU]


def c(x, d=2):
    """Desetinna carka pro popisky."""
    return f"{x:.{d}f}".replace(".", ",")


# ------------------------------------------------------------- obrazky ----
BARVA_Y, BARVA_Z, BARVA_PRAVDA, BARVA_BEZ = "#374151", "#1d4ed8", "#9ca3af", "#b42318"


def obrazek_pozadi(z, mu, dotyky):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1]})
    ax.plot(V, Y, color=BARVA_Y, lw=1.1, label="naměřené spektrum $\\mathbf{y}$")
    ax.plot(V, POZADI_PRAVDA, color=BARVA_PRAVDA, lw=1.5, ls="--",
            label="skutečné pozadí (známe jen díky simulaci)")
    ax.plot(V, z, color=BARVA_Z, lw=2.2, label=f"odhad pozadí $\\mathbf{{z}}$, RMSE {c(rmse(z), 3)}")
    ax.plot(V[dotyky], Y[dotyky], "o", ms=5, color=BARVA_Z, mec="white", mew=0.8, zorder=5,
            label=f"body dotyku ($\\mu_i > {c(PRAH_MU, 1).replace(",", "{,}")}$): {len(dotyky)}")
    ax.set_ylabel("intenzita [a. u.]")
    ax.set_title("Pozadí pod signálem, $\\alpha = 10^4$: dotýká se jen mezi píky", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax2.vlines(V, 0, mu, color=BARVA_Z, lw=1.4)
    ax2.plot(V[dotyky], mu[dotyky], "o", ms=3.5, color=BARVA_Z)
    ax2.set_ylabel("$\\mu_i$")
    ax2.set_xlabel("Ramanův posun [cm$^{-1}$]")
    ax2.set_title(f"multiplikátory omezení $z_i \\leq y_i$: nenulové jen v bodech dotyku, "
                  f"součet {c(mu.sum(), 3)}", fontsize=10)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OBRAZKY / "raman-pozadi.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)


def obrazek_korigovane(z):
    k = Y - z
    pravda = Y - POZADI_PRAVDA
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.plot(V, pravda, color=BARVA_PRAVDA, lw=1.4, ls="--",
            label="spektrum minus skutečné pozadí")
    ax.plot(V, k, color=BARVA_Z, lw=1.6, label="spektrum minus odhad $\\mathbf{y} - \\mathbf{z}$")
    for (stred, h, _), i, hv in zip(PIKY, INDEXY_PIKU, vysky(z)):
        ax.plot(V[i], hv, "o", ms=5, color=BARVA_Z, zorder=5)
        ax.annotate(f"{c(hv)} (pravda {c(h, 1)})", (V[i], hv), xytext=(0, 9),
                    textcoords="offset points", fontsize=9, color="#1e3a8a", ha="center",
                    va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_xlabel("Ramanův posun [cm$^{-1}$]")
    ax.set_ylabel("intenzita [a. u.]")
    ax.set_title("Korigované spektrum a výšky píků", fontsize=11)
    ax.set_ylim(-0.4, 5.0)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OBRAZKY / "raman-korigovane.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)


def obrazek_alfa():
    fig, osy = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, e, popis in zip(osy, (2, 4, 6), ("příliš ohebné", "zvolené", "tužší")):
        z, mu = pod_signalem(10.0 ** e)
        d = np.flatnonzero(mu > PRAH_MU)
        ax.plot(V, Y, color=BARVA_Y, lw=0.9)
        ax.plot(V, POZADI_PRAVDA, color=BARVA_PRAVDA, lw=1.3, ls="--")
        ax.plot(V, z, color=BARVA_Z, lw=2)
        ax.plot(V[d], Y[d], "o", ms=3.5, color=BARVA_Z, mec="white", mew=0.6, zorder=5)
        ax.set_title(f"$\\alpha = 10^{e}$ ({popis})\nRMSE {c(rmse(z), 3)}, dotyků {len(d)}",
                     fontsize=10)
        ax.set_xlabel("Ramanův posun [cm$^{-1}$]")
        ax.grid(alpha=0.25)
    osy[0].set_ylabel("intenzita [a. u.]")
    fig.tight_layout()
    fig.savefig(OBRAZKY / "raman-alfa.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    z, mu = pod_signalem()
    dotyky = np.flatnonzero(mu > PRAH_MU)
    i = np.arange(N)
    print(f"alfa = 1e4: RMSE pozadi {rmse(z):.3f}, max odchylka {np.max(np.abs(z - POZADI_PRAVDA)):.2f}")
    print(f"  vysky piku {np.round(vysky(z), 2)} (pravda {[p[1] for p in PIKY]})")
    print(f"  sum mu = {mu.sum():.3f} (n = {N}), sum i*mu = {np.sum(i * mu):.1f}, sum i = {i.sum()}")
    print(f"  min(y - z) = {np.min(Y - z):.2e} (omezeni splneno), "
          f"max |mu_i (y_i - z_i)| = {np.max(np.abs(mu * (Y - z))):.1e} (komplementarita)")
    serazene = np.sort(mu)[::-1]
    k = len(dotyky)
    print(f"  dotyku s mu > {PRAH_MU}: {k}; mezera: {k}. nejvetsi mu {serazene[k - 1]:.3f}, "
          f"{k + 1}. {serazene[k]:.4f}")
    for prah in (1e-2, 1e-3, 1e-6):
        print(f"    prah {prah:g}: {np.sum(mu > prah)} bodu (sum je solver)")
    print(f"  primarne y - z < 1e-6: {np.sum(Y - z < 1e-6)} bodu")
    print(f"  dotyky na vlnoctech {np.round(V[dotyky]).astype(int).tolist()}")
    print(f"  v okne +-2 sigma kolem stredu piku: "
          f"{sum(np.any(np.abs(V[dotyky] - s) < 2 * sg) for s, _, sg in PIKY)} piku ma dotyk")

    q = bez_omezeni()
    print("\njine vahy alfa (do reportu: Navic k reseni, Otazky do salu):")
    for e in (2, 3, 4, 4.5, 5, 5.5, 6):
        ze, mue = pod_signalem(10.0 ** e)
        print(f"  10^{e:g}: RMSE {rmse(ze):.3f}, dotyku {np.sum(mue > PRAH_MU)}, "
              f"vysky {np.round(vysky(ze), 2)}, sum mu {mue.sum():.1f}")

    print(f"bez omezeni (LS vyhlazeni), alfa = 1e4: RMSE {rmse(q):.3f}, "
          f"zapornych {np.mean(Y - q < 0):.0%} bodu, min(y - z) {np.min(Y - q):.2f}, "
          f"vysky {np.round(vysky(q), 2)}")

    OBRAZKY.mkdir(parents=True, exist_ok=True)
    obrazek_pozadi(z, mu, dotyky)
    obrazek_korigovane(z)
    obrazek_alfa()
    print("\nobrazky: raman-pozadi.svg, raman-korigovane.svg, raman-alfa.svg")
