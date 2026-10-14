"""Cena jednoho gradientu podle poctu promennych: numericky vs. autograd.

    uv run python cviceni/C4/kod/cena_gradientu.py

Funkce je prubezna funkce zobecnena na n promennych,

    f(x) = sum_{i=1}^{n-1} [(x_i - 1)^2 + (x_i x_{i+1} - 2)^2],

pro n = 2 presne f z tabule. Numericky gradient (centralni diference, h = 1e-5)
potrebuje 2n vyhodnoceni f, autograd jeden pruchod tam a jeden zpet.
Vypise tabulku casu a nejvetsi rozdil obou gradientu a nakresli
obrazky/python-cena-gradientu.svg. Casy jsou z jednoho pocitace a jen
orientacni, proto se nefixuji v data/cisla.json. Stejne mereni dela
druha cast notebooku gradienty.ipynb.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, NullFormatter

OBRAZKY = Path(__file__).resolve().parent.parent / "obrazky"
torch.set_num_threads(1)          # mereni bez vlivu poctu jader


def f_n(x):
    """Funguje pro pole NumPy i pro tenzor PyTorch."""
    return ((x[:-1] - 1) ** 2 + (x[:-1] * x[1:] - 2) ** 2).sum()


def grad_numericky(x, fce=f_n, h=1e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (fce(x + e) - fce(x - e)) / (2 * h)
    return g


def grad_autograd(x, fce=f_n):
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    fce(xt).backward()
    return xt.grad.numpy()


def cas(grad, x, opakovani):
    casy = []
    for _ in range(opakovani):
        start = time.perf_counter()
        grad(x)
        casy.append(time.perf_counter() - start)
    return min(casy)


rng = np.random.default_rng(0)
N_NUM = [10, 100, 1000, 10000]
N_AUTO = [10, 100, 1000, 10000, 100000]
cas_num, cas_auto = [], []
print("     n   numericky [ms]   autograd [ms]   pomer   max |rozdil|")
for n in N_AUTO:
    x = rng.uniform(0, 2, n)
    grad_autograd(x)                                  # zahrati
    cas_auto.append(cas(grad_autograd, x, 20))
    if n in N_NUM:
        rozdil = np.abs(grad_numericky(x) - grad_autograd(x)).max()
        cas_num.append(cas(grad_numericky, x, 3 if n >= 10000 else 10))
        print(f"{n:>6}   {cas_num[-1] * 1e3:14.2f}   {cas_auto[-1] * 1e3:13.3f}   "
              f"{cas_num[-1] / cas_auto[-1]:5.0f}×   {rozdil:.1e}")
    else:
        print(f"{n:>6}   {'—':>14}   {cas_auto[-1] * 1e3:13.3f}")

# ------------------------------------------------------------------ obrazek
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                     "svg.hashsalt": "mpc-omm-c4-python"})


def cesky(hodnota, _):
    """Popisek osy bez exponentu: 10 000 misto 10^4, 0,001 misto 10^-3."""
    if hodnota >= 1:
        return f"{hodnota:,.0f}".replace(",", "\u2009")
    return f"{hodnota:.10f}".rstrip("0").replace(".", ",")


CERVENA, AKCENT, SEDA = "#c73e1d", "#007f86", "#9ca3af"

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ms_num, ms_auto = np.array(cas_num) * 1e3, np.array(cas_auto) * 1e3
ax.loglog(N_NUM, ms_num, "o-", color=CERVENA, lw=2,
          label="numerický: 2n vyhodnocení f")
ax.loglog(N_AUTO, ms_auto, "s-", color=AKCENT, lw=2,
          label="autograd: průchod tam a zpět")
# voditko: cas umerny poctu vyhodnoceni f (2n), vychazi z bodu n = 10
n_ref = np.array([10, 10000])
ax.loglog(n_ref, ms_num[0] * n_ref / 10, ":", color=SEDA, lw=1.5)
ax.text(1500, ms_num[0] * 150 * 0.45, "∝ n", color="#6b7280", ha="left")
ax.set_ylim(0.1, 2000)
ax.set_xlabel("počet proměnných n")
ax.set_ylabel("čas jednoho gradientu [ms]")
ax.xaxis.set_major_formatter(FuncFormatter(cesky))
ax.yaxis.set_major_formatter(FuncFormatter(cesky))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
for strana in ("top", "right"):
    ax.spines[strana].set_visible(False)
fig.tight_layout()
cesta = OBRAZKY / "python-cena-gradientu.svg"
fig.savefig(cesta, metadata={"Date": None})
print("zapsano", cesta)
