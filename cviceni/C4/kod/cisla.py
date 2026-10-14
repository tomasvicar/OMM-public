"""Zafixovana cisla ctvrteho cviceni. Vsechny reporty a notebooky z nich vychazeji.

    uv run python cviceni/C4/kod/cisla.py

Vypise je a ulozi do cviceni/C4/data/cisla.json. Casy behu se nefixuji,
zavisi na pocitaci.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from spolecne import (DATA, f, grad_analyticky, grad_numericky, nacti,  # noqa: E402
                      projekce, transformuj, tv, uhly)

cisla = {}

# --- 1. prubezna funkce v bode (1; 1) -------------------------------------
x0 = np.array([1.0, 1.0])
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(1.0, requires_grad=True)
hodnota = f(x1, x2)
hodnota.backward()
krok = x0 - 0.1 * grad_analyticky(*x0)
cisla["funkce"] = {
    "bod": x0.tolist(),
    "f": f(*x0),
    "analyticky": grad_analyticky(*x0).tolist(),
    "centralni_h01": grad_numericky(f, x0, h=0.1).round(6).tolist(),
    "dopredna_h01": grad_numericky(f, x0, h=0.1, centralni=False).round(6).tolist(),
    "autograd": [x1.grad.item(), x2.grad.item()],
    # uzly grafu: a = x1 x2, r = a - 2, q = x1 - 1, f = q^2 + r^2
    "uzly_dopredu": {"a": 1.0, "r": -1.0, "q": 0.0, "f": 1.0},
    "uzly_zpet": {"df/df": 1.0, "df/dr": -2.0, "df/dq": 0.0, "df/da": -2.0,
                  "df/dx1": -2.0, "df/dx2": -2.0},
    "krok_alfa01": {"x": krok.round(6).tolist(), "f": round(f(*krok), 6)},
}

# kroky GD s alfa = 0,1 na tabuli (studenti dva, reseni tri): x_k, f(x_k), grad f(x_k), k = 0..3
x, kroky = x0.copy(), []
for k in range(4):
    kroky.append({"x": x.round(6).tolist(), "f": round(f(*x), 6),
                  "grad": grad_analyticky(*x).round(6).tolist()})
    x = x - 0.1 * grad_analyticky(*x)
cisla["funkce"]["kroky_alfa01"] = kroky

# GD do konvergence (|grad| < 1e-6) pro tri hodnoty alfa
gd = {}
for alfa in (0.02, 0.1, 0.2):
    x = x0.copy()
    for k in range(1, 20001):
        x = x - alfa * grad_analyticky(*x)
        if np.linalg.norm(grad_analyticky(*x)) < 1e-6 or not np.isfinite(x).all():
            break
    gd[str(alfa)] = {"iteraci": k, "x": np.round(x, 6).tolist(), "f": float(f(*x))}
cisla["gd_z_bodu_1_1"] = gd

# --- 2. registrace MRI ---------------------------------------------------
torch.manual_seed(0)
mri = nacti("mri-128.png")
pravda = [0.10, -0.06, math.radians(12)]
pohyblivy = (transformuj(mri, torch.tensor(pravda))
             + 0.02 * torch.randn_like(mri)).detach()


def registruj(start, kroku=300, lr=0.02):
    p = torch.tensor(start, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(kroku):
        opt.zero_grad()
        ztrata = ((transformuj(mri, p) - pohyblivy) ** 2).mean()
        ztrata.backward()
        opt.step()
    return p.detach(), ztrata.item()


p, ztrata = registruj([0.0, 0.0, 0.0])
cisla["registrace"] = {
    "obraz": "mri-128.png", "sum_sigma": 0.02, "seed": 0,
    "pravda": {"tx": 0.10, "ty": -0.06, "theta_stupne": 12,
               "tx_px": 0.10 * 64, "ty_px": -0.06 * 64},
    "kroku": 300, "lr": 0.02, "start": [0, 0, 0],
    "vysledek": {"tx": round(p[0].item(), 4), "ty": round(p[1].item(), 4),
                 "theta_stupne": round(math.degrees(p[2].item()), 2)},
    "mse": round(ztrata, 6),
}

# past: velka rotace 150 stupnu skonci v lokalnim minimu, multistart pomuze
pohyblivy_puvodni = pohyblivy
pohyblivy = (transformuj(mri, torch.tensor([0.10, -0.06, math.radians(150)]))
             + 0.02 * torch.randn_like(mri)).detach()
p, z = registruj([0.0, 0.0, 0.0])
multistart = []
for start_st in (0, 90, 180, 270):
    q, zq = registruj([0.0, 0.0, math.radians(start_st)])
    multistart.append({"start_stupne": start_st,
                       "theta_stupne": round(math.degrees(q[2].item()) % 360, 2),
                       "mse": round(zq, 6)})
cisla["registrace_past"] = {
    "pravda_theta_stupne": 150,
    "z_nuly": {"tx": round(p[0].item(), 4), "ty": round(p[1].item(), 4),
               "theta_stupne": round(math.degrees(p[2].item()), 2), "mse": round(z, 6)},
    "multistart": multistart,
}
pohyblivy = pohyblivy_puvodni

# --- 3. rekonstrukce CT ----------------------------------------------------
ct = nacti("ct-128.png")
n = ct.shape[-1]
yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
kruh = (xx ** 2 + yy ** 2 <= 1).float()


def rekonstruuj(pocet_uhlu, sum_sigma=0.0, lam=0.0, kroku=300, lr=0.05):
    u = uhly(pocet_uhlu)
    torch.manual_seed(0)
    sino = projekce(ct, u).detach()
    sino = sino + sum_sigma * torch.randn_like(sino)
    x = torch.zeros_like(ct, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(kroku):
        opt.zero_grad()
        ztrata = ((projekce(x, u) - sino) ** 2).mean() + lam * tv(x)
        ztrata.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    rmse = (((x.detach() - ct) ** 2 * kruh).sum() / kruh.sum()).sqrt().item()
    return round(rmse, 4)


cisla["ct"] = {
    "obraz": "ct-128.png", "pixelu": n * n, "kroku": 300, "lr": 0.05,
    "omezeni": "po kazdem kroku clamp na <0, 1>",
    "rmse_definice": "uvnitr kruhu vepsaneho do obrazu",
    "zakladni": {"uhlu": 60, "rmse": rekonstruuj(60)},
    "malo_uhlu": {"uhlu": 12, "rmse": rekonstruuj(12)},
    "sum": {"uhlu": 60, "sigma": 1.0,
            "tv": {str(lam): rekonstruuj(60, 1.0, lam) for lam in (0.0, 0.001, 0.004, 0.01)}},
    "numericky_gradient_projekci": 2 * n * n,
}

for klic, hodnota in cisla.items():
    print(klic, json.dumps(hodnota, ensure_ascii=False))
(DATA / "cisla.json").write_text(json.dumps(cisla, ensure_ascii=False, indent=1) + "\n")
print(f"ulozeno {DATA / 'cisla.json'}")
