# Zdroje převzatých ilustrací

- `obrazky/kohoutky.jpg` — převzato z původní L1 v
  `/var/home/tom/code/tmp/68c3b0ca0d60e3a971ef17f6/L1/imgs/water_tap.jpg`;
  vloženo na žádost autora materiálů.
- `obrazky/auto-openclipart.svg` — [Car side view](https://openclipart.org/detail/9792/car-side-view),
  autor Anonymous, Openclipart (CC0/public domain).

Všechny ostatní ilustrace v `obrazky/` vytváří
[`kod/generuj_obrazky.py`](kod/generuj_obrazky.py).

`radioterapie-plany.png` a `radioterapie-model.svg` používají tutéž geometrii
řezu i tutéž matici dávek jako živá ukázka na prvním počítačovém cvičení
([`cviceni/C1/kod/ukazka_radioterapie.py`](../../cviceni/C1/kod/ukazka_radioterapie.py)),
aby přednáška i cvičení ukazovaly doslova stejného pacienta. Plán B v mapách
dávky je skutečné optimum konvexního kvadratického programu, ne kresba od oka;
plány A a C nikdo neoptimalizoval. Mapy dávky jsou rastrové (PNG), protože jde
o datové heatmapy.

`lokalni-globalni.gif` je jediný animovaný obrázek přednášky: dvě trajektorie
gradientního sestupu z různých startů se počítají skutečně, ne kreslí od oka.
Jeho první snímek je výsledný stav, aby v PDF nebo v tisku dával smysl i bez
přehrávání.
