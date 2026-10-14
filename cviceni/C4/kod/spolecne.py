"""Spolecne funkce ctvrteho cviceni: prubezna funkce, registrace a projekce CT.

Importuji je skripty v cviceni/C4/kod/, aby vsechna cisla v reportech
vznikla stejnym kodem. Notebooky maji tytez funkce vypsane primo v sobe.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DATA = Path("cviceni/C4/data")


# --- prubezna funkce -------------------------------------------------------

def f(x1, x2):
    """f(x1, x2) = (x1 - 1)^2 + (x1 x2 - 2)^2, minimum v (1; 2), f = 0."""
    return (x1 - 1) ** 2 + (x1 * x2 - 2) ** 2


def grad_analyticky(x1, x2):
    return np.array([2 * (x1 - 1) + 2 * (x1 * x2 - 2) * x2, 2 * (x1 * x2 - 2) * x1])


def grad_numericky(fce, x, h=1e-5, centralni=True):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        if centralni:
            g[i] = (fce(*(x + e)) - fce(*(x - e))) / (2 * h)
        else:
            g[i] = (fce(*(x + e)) - fce(*x)) / h
    return g


# --- obrazky ---------------------------------------------------------------

def nacti(jmeno: str) -> torch.Tensor:
    """Sedotonovy PNG jako tenzor 1 x 1 x N x N v rozsahu 0..1."""
    a = np.asarray(Image.open(DATA / jmeno), dtype=np.float32) / 255
    return torch.from_numpy(a)[None, None]


# --- registrace ------------------------------------------------------------

def transformuj(obr: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Otoci obraz o p[2] (radiany) a posune o (p[0], p[1]) v normovanych
    souradnicich (-1..1 pres cely obraz). Bilinearni interpolace, takze je
    vysledek diferencovatelny podle p."""
    c, s = torch.cos(p[2]), torch.sin(p[2])
    theta = torch.stack([torch.stack([c, -s, p[0]]),
                         torch.stack([s, c, p[1]])])[None]
    mrizka = F.affine_grid(theta, obr.shape, align_corners=False)
    return F.grid_sample(obr, mrizka, mode="bilinear", align_corners=False)


# --- CT --------------------------------------------------------------------

def uhly(pocet: int) -> torch.Tensor:
    """Rovnomerne uhly v intervalu <0, pi)."""
    return torch.arange(pocet) * math.pi / pocet


def projekce(obr: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Paralelni projekce (Radonova transformace): obraz se otoci o kazdy
    uhel a secte po sloupcich. Vysledek je sinogram (pocet uhlu x N)."""
    n = obr.shape[-1]
    c, s, z = torch.cos(u), torch.sin(u), torch.zeros_like(u)
    theta = torch.stack([torch.stack([c, -s, z], 1), torch.stack([s, c, z], 1)], 1)
    mrizka = F.affine_grid(theta, (len(u), 1, n, n), align_corners=False)
    otocene = F.grid_sample(obr.expand(len(u), 1, n, n), mrizka, align_corners=False)
    return otocene.sum(dim=2)[:, 0, :]


def tv(obr: torch.Tensor) -> torch.Tensor:
    """Celkova variace (hladka verze), penalizace proti sumu."""
    dx = obr[..., :, 1:] - obr[..., :, :-1]
    dy = obr[..., 1:, :] - obr[..., :-1, :]
    return torch.sqrt(dx[..., :-1, :] ** 2 + dy[..., :, :-1] ** 2 + 1e-6).sum()
