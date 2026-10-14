"""Pripravi snimky pro ctvrte cviceni ze stazenych originalu.

    uv run python cviceni/C4/kod/priprav_data.py

Originaly jsou z Wikimedia Commons (puvod a licence v cviceni/C4/zdroje.md):
    data/mri-mozek-t2-axialni.jpg  1080 x 1080, CC BY-SA 4.0
    data/ct-mozek-axialni.png      646 x 468, CC0; vpravo je vlozeny
                                   sagitalni nahled, ten se odrizne
Vystup jsou ctvercove sedotonove snimky 128 x 128 (PNG):
    data/mri-128.png   pro registraci
    data/ct-128.png    pro rekonstrukci
"""

from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("cviceni/C4/data")
N = 128


def ctverec(obr: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Oriznuti na box, doplneni cernou na ctverec a zmenseni na N x N."""
    obr = obr.convert("L").crop(box)
    s = max(obr.size)
    platno = Image.new("L", (s, s), 0)
    platno.paste(obr, ((s - obr.width) // 2, (s - obr.height) // 2))
    return platno.resize((N, N), Image.LANCZOS)


# MRI: hlava je uprostred, staci ubrat cerny okraj
mri = ctverec(Image.open(DATA / "mri-mozek-t2-axialni.jpg"), (170, 100, 910, 1040))
# CT: jen leva cast s axialnim rezem; nahled vpravo zacina na x = 362
ct = ctverec(Image.open(DATA / "ct-mozek-axialni.png"), (0, 40, 360, 468))

for jmeno, obr in [("mri-128.png", mri), ("ct-128.png", ct)]:
    obr.save(DATA / jmeno)
    a = np.asarray(obr) / 255
    print(f"{jmeno}: {obr.size}, prumer {a.mean():.3f}, max {a.max():.3f}")
