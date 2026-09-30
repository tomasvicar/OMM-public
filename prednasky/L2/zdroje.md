# Zdroje ilustrací

Všechny ilustrace v `obrazky/` vytváří [`kod/generuj_obrazky.py`](kod/generuj_obrazky.py);
žádný obrázek není převzatý.

Dva obrázky jsou animované smyčky (GIF), protože jejich sdělení nese pohyb:
`vrstevnice-posun.gif` (vrstevnice putuje ve směru $\mathbf c$ až k poslednímu
dotyku ve vrcholu) a `l1-vs-mnc.gif` (odlehlé měření stoupá a přímka nejmenších
čtverců se za ním otáčí). U obou je první snímek výsledný stav, aby v PDF nebo
v tisku dávaly smysl i bez přehrávání. Ostatní obrázky jsou vektorové (SVG).

Data v obrázcích, která nejsou jen školní čísla:

- `radioterapie-matice.svg` — matice dávek se počítá z geometrie (Gaussovský
  profil svazku a exponenciální útlum s hloubkou), nejde o klinická data; řez,
  nádor, mícha i profil svazku jsou tytéž jako v první přednášce a v živé
  ukázce [`cviceni/C1/kod/ukazka_radioterapie.py`](../../cviceni/C1/kod/ukazka_radioterapie.py),
  jen se ozařuje ze tří směrů místo osmi;
- `l1-vs-mnc.gif` — používá tatáž data jako
  [`kod/lp_triky_cvxpy.py`](kod/lp_triky_cvxpy.py); poslední vzorek se
  v animaci zvedá z 17,0 na 25,0 mV a obě proložení se v každém snímku
  počítají znovu;
- `stinove-ceny.svg` a `posun-omezeni.svg` — hodnoty ověřené v CVXPY proti
  [`kod/lp_pekarna_cvxpy.py`](kod/lp_pekarna_cvxpy.py).
