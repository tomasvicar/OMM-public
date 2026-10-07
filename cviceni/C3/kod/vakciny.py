"""Samostatna uloha tretiho cviceni: rozdeleni vakcin mezi ctyri okresy.

    uv run python cviceni/C3/kod/vakciny.py

Kraj ma 30 tisic davek pro rizikove skupiny ve ctyrech okresech (Mesto,
Prumysl, Venkov, Hory). x_i tisic davek v okrese i odvrati a_i ln(1 + x_i)
hospitalizaci, a = (60; 40; 30; 3). Rizikove populace n = (10; 12; 18; 6)
tisic osob, vic davek nez lidi okresu nedame.

    maximize   sum_i a_i ln(1 + x_i)
    subject to sum_i x_i <= 30            (dodavka, multiplikator mu_D)
               x_i <= n_i                 (strop, mu_i^strop)
               x_i >= 0                   (nezapornost, mu_i^nula)

-> x* = (10; 11,57; 8,43; 0), odvraceno 312,44, mu_D = 70/22 = 3,18.
Stacionarita po slozkach: a_i/(1 + x_i) = mu_D + mu_i^strop - mu_i^nula.
Okresy se v textu cisluji 1-4 (Mesto ... Hory), v Pythonu jsou na indexech 0-3;
multiplikator dodavky je mu_D.

Skript je jediny zdroj cisel pro report, notebook a obrazek
cviceni/C3/obrazky/vakciny-dodavka.svg.
"""

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

A = np.array([60.0, 40.0, 30.0, 3.0])        # hospitalizace na jednotku ln(1 + x)
N = np.array([10.0, 12.0, 18.0, 6.0])        # rizikova populace [tis. osob]
DODAVKA = 30.0                               # [tis. davek]
OBRAZKY = Path("cviceni/C3/obrazky")
SVG = {"Date": None}
plt.rcParams["svg.hashsalt"] = "mpc-omm-c3-vakciny"


def c(v: float, mist: int = 2) -> str:
    """Cislo s desetinnou carkou."""
    return f"{v:.{mist}f}".replace(".", ",")


def vyres(dodavka=DODAVKA, strop=True, minimum=0.0, rovnost=False):
    """Vrati x, odvraceno, mu_D, mu_strop, mu_nula (nebo None, kdyz uloha nema reseni)."""
    x = cp.Variable(4)
    rozpocet = (cp.sum(x) == dodavka) if rovnost else (cp.sum(x) <= dodavka)
    om_strop = x <= N
    om_nula = x >= minimum
    omezeni = [rozpocet, om_nula] + ([om_strop] if strop else [])
    uloha = cp.Problem(cp.Maximize(A @ cp.log(1 + x)), omezeni)
    uloha.solve(solver=cp.CLARABEL)
    if uloha.status not in ("optimal", "optimal_inaccurate"):
        return None, uloha.status
    mu_strop = om_strop.dual_value if strop else np.zeros(4)
    return (x.value, uloha.value, float(rozpocet.dual_value), mu_strop, om_nula.dual_value), uloha.status


def rozdeleni(dodavka: float):
    """Totez analyticky ze stacionarity: x_i = clip(a_i/mu_D - 1, 0, n_i), mu_D bisekci."""
    if dodavka >= N.sum():
        return N.copy(), 0.0
    lo, hi = 1e-9, A.max()
    for _ in range(200):
        mu_D = (lo + hi) / 2
        if np.clip(A / mu_D - 1, 0, N).sum() > dodavka:
            lo = mu_D
        else:
            hi = mu_D
    mu_D = (lo + hi) / 2
    return np.clip(A / mu_D - 1, 0, N), mu_D


def odvraceno(x) -> float:
    return float(A @ np.log(1 + np.asarray(x)))


def obrazek_dodavka(mu_D_30, f30):
    s = np.linspace(0, 60, 601)
    f = np.array([odvraceno(rozdeleni(v)[0]) for v in s])
    mu_D = np.array([rozdeleni(v)[1] for v in s])
    f40 = odvraceno(rozdeleni(40)[0])
    fig, (o1, o2) = plt.subplots(1, 2, figsize=(11, 4.0))
    o1.plot(s, f, color="#007f86", lw=2.2, label="optimum pro danou dodávku")
    t = np.array([18, 48])
    o1.plot(t, f30 + mu_D_30 * (t - 30), color="#c73e1d", ls="--", lw=1.4,
            label=f"tečna se sklonem $\\mu_D$ = {c(mu_D_30)}")
    o1.plot([30], [f30], "o", color="#007f86", ms=8)
    o1.plot([40], [f40], "o", color="#007f86", ms=8)
    o1.plot([40], [f30 + 10 * mu_D_30], "o", color="#c73e1d", ms=7, mfc="white")
    o1.annotate(f"30 tis.: {c(f30)}", (30, f30), xytext=(14, 330), fontsize=9.5,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#374151"))
    o1.annotate(f"40 tis. skutečně: {c(f40)} (+{c(f40 - f30)})", (40, f40), xytext=(38, 300),
                fontsize=9.5, color="#007f86", arrowprops=dict(arrowstyle="-", lw=0.8, color="#007f86"))
    o1.annotate(f"odhad {c(f30)} + 10·$\\mu_D$: {c(f30 + 10 * mu_D_30)} (+{c(10 * mu_D_30)})", (40, f30 + 10 * mu_D_30),
                xytext=(8, 372), fontsize=9.5, color="#c73e1d",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#c73e1d"))
    o1.set_xlabel("dodávka [tis. dávek]")
    o1.set_ylabel("odvrácené hospitalizace")
    o1.set_ylim(0, 400)
    o1.set_title("Hodnota optima podle velikosti dodávky", fontsize=11)
    o1.legend(loc="lower right", fontsize=9)
    o1.grid(alpha=0.25)
    o2.plot(s, mu_D, color="#c73e1d", lw=2.2)
    o2.axvline(30, color="#9ca3af", ls=":")
    o2.axvline(N.sum(), color="#9ca3af", ls=":")
    o2.text(30.5, 9.3, "30 tis.", fontsize=9, color="#374151")
    o2.text(N.sum() + 0.8, 5.0, f"{N.sum():.0f} tis.:\nvšichni\nnaočkovaní,\n$\\mu_D$ = 0", fontsize=9,
            color="#374151", va="top", ha="left")
    o2.set_xlabel("dodávka [tis. dávek]")
    o2.set_ylabel("$\\mu_D$ [hospitalizace na tis. dávek]")
    o2.set_ylim(0, 10)
    o2.set_title("Stínová cena dodávky klesá (klesající výnos)", fontsize=11)
    o2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OBRAZKY / "vakciny-dodavka.svg", metadata=SVG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    (x, f, mu_D, mu_s, mu_n), _ = vyres()
    print("HLAVNI ULOHA (30 tis. davek)")
    print(f"  x* = {np.round(x, 4)} tis. davek, odvraceno {f:.4f}")
    print(f"  mu_D (dodavka) = {mu_D:.4f}  (70/22 = {70 / 22:.4f})")
    print(f"  mu strop = {np.round(mu_s, 4)}  (60/11 - mu_D = {60 / 11 - mu_D:.4f})")
    print(f"  mu nula  = {np.round(mu_n, 4)}  (mu_D - 3 = {mu_D - 3:.4f})")
    mezni = A / (1 + x)
    print(f"  mezni prinosy a_i/(1+x_i) = {np.round(mezni, 4)}")
    print(f"  stacionarita a/(1+x) - mu_D - mu_s + mu_n = {np.round(mezni - mu_D - mu_s + mu_n, 6)}")
    print(f"  Prumysl+Venkov: 70/mu_D - 2 = {70 / mu_D - 2:.4f} = 30 - 10; x2 = 40/mu_D - 1 = {40 / mu_D - 1:.4f}")
    xa, nua = rozdeleni(DODAVKA)
    print(f"  analyticky ze stacionarity: x = {np.round(xa, 4)}, mu_D = {nua:.4f}")

    print("\nCO PRINESE DODAVKA NAVIC")
    for d in (31, 35, 40):
        (_, fd, mu_Dd, _, _), _ = vyres(d)
        print(f"  {d} tis.: odvraceno {fd:.4f}, prirustek {fd - f:+.4f}, linearni odhad {mu_D * (d - 30):+.4f},"
              f" mu_D = {mu_Dd:.4f}")
    (_, f29, _, _, _), _ = vyres(29)
    print(f"  29 tis.: odvraceno {f29:.4f}, ztrata {f - f29:.4f}")
    print(f"  1 hospitalizace navic stoji asi {1000 / mu_D:.0f} davek")

    print("\nPOROVNANI")
    lidi = DODAVKA * N / N.sum()
    print(f"  podle populace {np.round(lidi, 2)}: odvraceno {odvraceno(lidi):.4f} (o {f - odvraceno(lidi):.2f} mene)")
    stejne = np.full(4, DODAVKA / 4)
    print(f"  rovnomerne 7,5: sum = {stejne.sum()}, Hory nad stropem? {stejne[3] > N[3]} "
          f"(7,5 > 6) -> odvraceno {odvraceno(stejne):.4f} (neplatne, nepouzivat)")
    (xb, fb, nub, _, _), _ = vyres(strop=False)
    print(f"  bez stropu: x = {np.round(xb, 4)}, slibuje {fb:.4f}, mu_D = {nub:.4f}")
    print(f"     skutecne (davky nad populaci nic nedelaji): {odvraceno(np.minimum(xb, N)):.4f}")

    print("\nTYPICKE CHYBY")
    _, st = vyres(rovnost=True)
    print(f"  rozpocet jako rovnost, 30 tis.: status {st} (vyjde totez, rozpocet je aktivni)")
    _, st50 = vyres(50, rovnost=True)
    _, st50n = vyres(50)
    print(f"  rozpocet jako rovnost, 50 tis.: status {st50}; s <= : {st50n}")
    try:
        xx = cp.Variable(4)
        cp.Problem(cp.Minimize(A @ cp.log(1 + xx)), [cp.sum(xx) <= 30, xx >= 0, xx <= N]).solve()
    except cp.error.DCPError as e:
        print(f"  Minimize misto Maximize: DCPError ({str(e)[:60]}...)")

    print("\nVYZNAM MU STROPU MESTA")
    xx = cp.Variable(4)
    u11 = cp.Problem(cp.Maximize(A @ cp.log(1 + xx)),
                     [cp.sum(xx) <= DODAVKA, xx >= 0, xx <= N + np.array([1.0, 0, 0, 0])])
    u11.solve(solver=cp.CLARABEL)
    print(f"  strop Mesta 10 -> 11: odvraceno {u11.value:.4f}, prirustek {u11.value - f:+.4f}"
          f" (odhad mu = {mu_s[0]:.4f}); tisicovka se presune z Prumyslu/Venkova do Mesta:"
          f" 60/11 - mu_D = {60 / 11 - mu_D:.4f}")

    print("\nBONUS: KAZDY OKRES ASPON 1 TIS.")
    (xs, fs, nus, mus_s, mus_n), _ = vyres(minimum=1.0)
    print(f"  x = {np.round(xs, 4)}, odvraceno {fs:.4f}, cena spravedlnosti {f - fs:.4f}")
    print(f"  mu_D = {nus:.4f}, mu strop = {np.round(mus_s, 4)}, mu minimum = {np.round(mus_n, 4)}")
    print(f"  Hory: mezni prinos 3/2 = 1,5 < mu_D -> mu minima = mu_D - 1,5 = {nus - 1.5:.4f}")
    for m in (2.0, 3.0):
        (xm, fm, _, _, _), _ = vyres(minimum=m)
        print(f"  minimum {m:g}: x = {np.round(xm, 3)}, odvraceno {fm:.4f}, cena {f - fm:.4f}")
    podil = 0.2 * N
    x2 = cp.Variable(4)
    u2 = cp.Problem(cp.Maximize(A @ cp.log(1 + x2)), [cp.sum(x2) <= 30, x2 >= podil, x2 <= N])
    u2.solve(solver=cp.CLARABEL)
    print(f"  aspon 20 % populace: x = {np.round(x2.value, 3)}, odvraceno {u2.value:.4f}, cena {f - u2.value:.4f}")

    print("\nRUZNE DODAVKY")
    for d in (0, 5, 10, 20, 30, 40, 45, 46, 50, 60):
        xd, nd = rozdeleni(d)
        print(f"  {d:2d} tis.: x = {np.round(xd, 2)}, mu_D = {nd:.3f}, odvraceno {odvraceno(xd):.2f}")
    # zlomove dodavky: kdy se zmeni aktivni mnozina (mu_D projde a_i/(1+n_i) nebo a_i)
    zlomy = sorted(set(np.r_[A / (1 + N), A]))
    print("  zlomy (mu_D -> dodavka):", ", ".join(
        f"mu_D={z:.3f}: {np.clip(A / z - 1, 0, N).sum():.2f}" for z in zlomy))

    obrazek_dodavka(mu_D, f)
    print(f"\nobrazek: {OBRAZKY}/vakciny-dodavka.svg")
