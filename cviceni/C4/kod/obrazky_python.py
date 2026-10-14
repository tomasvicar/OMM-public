"""Obrazky reportu cviceni/C4/reporty/python.html: GD se tremi gradienty.

    uv run python cviceni/C4/kod/obrazky_python.py

Spusti gradientni sestup z (1; 1) s alfa = 0,1 se tremi zamenitelnymi
gradienty (numericky, analyticky, autograd), overi zafixovana cisla
z data/cisla.json (prvni krok, pocet iteraci) a nakresli
obrazky/python-trajektorie.svg. Graf ceny gradientu kresli cena_gradientu.py.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from spolecne import DATA, f, grad_analyticky, grad_numericky  # noqa: E402

OBRAZKY = Path(__file__).resolve().parent.parent / "obrazky"
CISLA = json.loads((DATA / "cisla.json").read_text())


def grad_autograd(x):
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    f(xt[0], xt[1]).backward()
    return xt.grad.numpy()


def gd(grad, x0, alpha, kroku=20000, tol=1e-6):
    """Stejny GD jako v notebooku: konec, kdyz |grad f| < tol."""
    x = np.array(x0, dtype=float)
    cesta = [x]
    for _ in range(kroku):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        cesta.append(x)
    return np.array(cesta)


GRADIENTY = {
    "numerický (centrální diference)": lambda x: grad_numericky(f, x),
    "analytický (vzorec z tabule)": lambda x: grad_analyticky(*x),
    "autograd (PyTorch)": grad_autograd,
}
cesty = {jmeno: gd(g, CISLA["funkce"]["bod"], 0.1) for jmeno, g in GRADIENTY.items()}

ocekavano = CISLA["gd_z_bodu_1_1"]["0.1"]
for jmeno, cesta in cesty.items():
    print(f"{jmeno:>32}: první krok {np.round(cesta[1], 6)}, {len(cesta) - 1} iterací, "
          f"konec {np.round(cesta[-1], 6)}")
    assert np.allclose(cesta[1], CISLA["funkce"]["krok_alfa01"]["x"])
    assert len(cesta) - 1 == ocekavano["iteraci"]
    assert np.allclose(cesta[-1], ocekavano["x"], atol=1e-6)
odchylka = max(np.abs(c - cesty["analytický (vzorec z tabule)"]).max() for c in cesty.values())
print(f"největší odchylka cest od analytické: {odchylka:.1e}")

# ------------------------------------------------------------------ obrazek
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                     "svg.hashsalt": "mpc-omm-c4-python"})
SEDA, ZELENA = "#9ca3af", "#1a7f37"
STYLY = [dict(color="#c73e1d", lw=7, alpha=0.35, solid_capstyle="round"),
         dict(color="#1f4e79", lw=2.2, ls="-"),
         dict(color="#007f86", lw=0, marker="o", ms=3.5, markevery=1)]

X1, X2 = np.meshgrid(np.linspace(0.4, 2.0, 300), np.linspace(0.6, 2.6, 300))
fig, ax = plt.subplots(figsize=(5.2, 6.4))
ax.contour(X1, X2, f(X1, X2), levels=np.geomspace(0.003, 3, 10),
           colors=SEDA, linewidths=0.9)
for (jmeno, cesta), styl in zip(cesty.items(), STYLY):
    ax.plot(cesta[:, 0], cesta[:, 1], label=jmeno, **styl)
ax.plot(*cesty["autograd (PyTorch)"][0], "ks", ms=7)
ax.annotate("start (1; 1)", cesty["autograd (PyTorch)"][0], xytext=(8, -14),
            textcoords="offset points")
ax.annotate("první krok (1,2; 1,2)", cesty["autograd (PyTorch)"][1], xytext=(10, -4),
            textcoords="offset points")
ax.plot(1, 2, "*", color=ZELENA, ms=16, zorder=5)
ax.annotate("minimum (1; 2)", (1, 2), xytext=(-12, 12), textcoords="offset points",
            color=ZELENA, ha="right")
ax.set_xlim(0.4, 2.0)
ax.set_ylim(0.6, 2.6)
ax.set_aspect("equal")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
fig.tight_layout()
cesta = OBRAZKY / "python-trajektorie.svg"
fig.savefig(cesta, metadata={"Date": None})
print("zapsano", cesta)
