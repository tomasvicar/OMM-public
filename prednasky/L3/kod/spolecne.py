"""Společné nastavení a paleta pro generátory obrázků třetí přednášky."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "obrazky"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "mpc-omm-l3"
SVG_METADATA = {"Date": None}

MODRA = "#12355b"
TYRKYSOVA = "#007f86"
CERVENA = "#c73e1d"
ORANZOVA = "#e9a23b"
SVETLE_MODRA = "#d9eef7"
SEDIVA = "#eef2f4"

# Pekárna jako QP — čísla platná pro celou přednášku (ověřená v qp_pekarna_cvxpy.py).
#
# Navazuje na druhou přednášku beze švu: tam byl zisk z kusu pevných 14 Kč
# (chléb) a 8 Kč (bageta). Dnes tytéž marže s objemem klesají, protože se musí
# slevovat,
#     p1(x1) = 14 - 0,025 x1,   p2(x2) = 8 - 0,024 x2,
# takže při nulovém objemu vyjde přesně původní zadání a oba zisky jdou přímo
# porovnat. Účel  max P.x - D.x^2  se převádí na  min 1/2 x^T Q x + c^T x
# s  Q = diag(2 D)  a  c = -P.
A_PEKARNA = ((0.5, 0.2), (3.0, 2.0))
B_PEKARNA = (80.0, 600.0)
P_PEKARNA = (14.0, 8.0)        # zisk z kusu při nulovém objemu (z druhé přednášky)
D_PEKARNA = (0.025, 0.024)     # o kolik marže klesne s každým dalším kusem
Q_PEKARNA = (0.05, 0.048)      # diagonála Q = 2 D
C_PEKARNA = (-14.0, -8.0)      # c = -P
X_QP = (120.0, 100.0)          # optimum kvadratické úlohy
X_LP = (100.0, 150.0)          # optimum lineární úlohy z druhé přednášky
MU_QP = (16.0, 0.0)            # multiplikátory: mouka aktivní, pec s rezervou
MARZE_QP = (11.0, 5.6)         # marže v optimu [Kč/ks]
ZISK_QP = 1880.0               # hodnota účelu v optimu [Kč]
X_NEOMEZENE = (280.0, 500.0 / 3.0)   # stacionární bod bez omezení


def zisk(x1, x2):
    """Denní zisk pekárny [Kč] při klesajících maržích; účel úlohy je -zisk."""
    return (P_PEKARNA[0] * x1 - D_PEKARNA[0] * x1**2
            + P_PEKARNA[1] * x2 - D_PEKARNA[1] * x2**2)


def uloz_gif(snimky, doby, jmeno: str, dpi: int = 120) -> None:
    """Uloží posloupnost figur jako smyčkovaný GIF.

    `snimky` jsou hotové figury v pořadí přehrávání, `doby` jejich doby
    zobrazení v milisekundách. **První snímek je zároveň statickou zálohou**
    (v PDF nebo v tisku se ukáže jen on), takže do něj patří výsledný stav.
    """
    obrazky = []
    for fig in snimky:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        obrazky.append(Image.open(buffer).convert("RGB")
                       .quantize(colors=96, method=Image.MEDIANCUT))
    obrazky[0].save(OUT / f"{jmeno}.gif", save_all=True, append_images=obrazky[1:],
                    duration=list(doby), loop=0, optimize=True, disposal=2)
