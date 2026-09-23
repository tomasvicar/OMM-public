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

## Revize 23. 9. 2026

- Příběh pokus → model → řešení, limit kohoutku a auta a rozsah matematického
  dodatku vycházejí z místních materiálů autora:
  `../../MLR/MLR/lectures/L1/index.qmd` a
  `../../AUI/lectures/bayesian-optimization/index.qmd`
  (cesty vztažené ke kořeni tohoto repozitáře). Text je adaptovaný pro toto publikum.
- Mapa předmětu vychází z aktuálního dokumentu
  `osnova/Harmonogram MPC-EAL 2026_27 2.docx`; čitelný přepis a vyznačení
  odlišnosti kódu předmětu jsou v `osnova/HARMONOGRAM-2026-27.md`.
- Nové diagramy `revize-*.svg` počítá a kreslí
  [`kod/revize_priklady.py`](kod/revize_priklady.py), včetně numerického
  ověření omezeného optima kohoutku. Matematický dodatek `zaklady-*.svg`
  generuje [`kod/revize_matematika.py`](kod/revize_matematika.py).
  Oba moduly spouští hlavní generátor; nejde o převzaté bitmapy.
- Revize prošla nezávislou oponenturou: byly opraveny záměna neznámého
  předpisu s neexistencí funkce, podmínky tvrzení o vrcholu LP a značení
  skalárů a vektorů. Následoval přepočet příkladů a vizuální kontrola renderu.

Druhá nezávislá kontrola na výslovnou žádost autora doplnila opravy
orientace údolí v poznámkách, vektorů v původních SVG a hranic prázdné
přípustné množiny. Zavádějící úplné pořadí obtížnosti tříd bylo nahrazeno
vysvětlením vlivu konvexity, celočíselnosti a ceny vyhodnocení.
