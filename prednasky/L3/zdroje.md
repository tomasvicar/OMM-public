# Zdroje ilustrací

Všechny ilustrace v `obrazky/` vytváří [`kod/generuj_obrazky.py`](kod/generuj_obrazky.py),
který postupně spustí generátory `kod/obrazky_blok*.py` (čísla v názvech
pocházejí z pořadí bloků před přeskládáním 5. 10. 2026, aktuální pořadí udává
seznam `BLOKY`); společnou paletu a závazná čísla hlavního příkladu drží
[`kod/spolecne.py`](kod/spolecne.py). **Žádný obrázek není převzatý.**

Rastrové výstupy jsou jen animace (GIF): `vrstevnice-posun.gif`,
`stacionarni-bod-sestup.gif`, `lagrange-skolni.gif`, `lagrange-citlivost.gif`,
`komplementarita-pec.gif`, `dualita-cena-mouky.gif`, `penalizace.gif`,
`bariera-cesta.gif`, `poplatek-mouka.gif` a `dualni-mezera.gif` — nesou pohyb (posun vrstevnice, sestup, rostoucí
penalizace), který by statická podoba ztratila. První snímek je vždy výsledný
stav. Ostatní obrázky jsou vektorové (SVG). Regularizace, Lagrangeova podmínka
a srovnání pokuty s bariérou mají na slidech navíc interaktivní widget
(vanilla JS přímo v `index.qmd`). Po zjednodušení bloků o lagrangiánu, KKT
a dualitě (5. 10. 2026) se na slidech už nepoužívají `lagrange-3d.svg`,
`lagrange-skolni.gif`, `lagrange-citlivost.gif`, `objem-povrch.svg`,
`znamenko-mu.svg`, `kkt-pekarna-tipy.svg` a `dualita-cena-mouky.gif`;
soubory i generátory zůstávají, odkazují na ně podcasty.

Data v obrázcích, která nejsou jen školní čísla:

- `vrstevnice-posun.gif`, `pekarna-cena.svg`, `komplementarita-pec.gif`,
  `kkt-pekarna-tipy.svg`, `dualita-cena-mouky.gif`, `penalizace.gif`,
  `bariera-cesta.gif`, `obe-cesty-cena.svg` a `poplatek-mouka.gif` — přípustná oblast a optimum
  pekárny jako QP, hodnoty ověřené v CVXPY proti
  [`kod/qp_pekarna_cvxpy.py`](kod/qp_pekarna_cvxpy.py). **Zisk z jednoho kusu**
  (ne cena — druhá přednáška maximalizuje zisk, a aby šla obě čísla srovnat,
  zůstává jím i tady) začíná na hodnotách z druhé přednášky a klesá s objemem,
  $p_1=14-0{,}025x_1$ a $p_2=8-0{,}024x_2$, takže oba zisky jdou přímo
  porovnat; optimum je $\mathbf x^\star=(120,100)$, zisk 1880 Kč,
  $\boldsymbol\mu=(16,0)$ a stacionární bod bez omezení $(280;\,166{,}7)$;
- widget regularizace a `svm-odstup.svg` — tatáž data a tytéž výsledky jako
  [`kod/qp_priklady_cvxpy.py`](kod/qp_priklady_cvxpy.py);
- `stacionarni-typy.svg` — matice $Q$ mají schválně **nenulový mimodiagonální
  prvek**; na samotné diagonále by test přes determinant a stopu neměl smysl,
  protože typ bodu by se přečetl rovnou z čísel;
- `bariera-cesta.gif` — centrální cesta se počítá numericky, není
  zakreslená od oka;
- `lagrange-citlivost.gif` — $p^\star(b)=b^2/2$ ze školního příkladu
  $\min x_1^2+x_2^2$ za $x_1+x_2=b$;
- `lp-dual-pekarna.svg` — LP pekárna z druhé přednášky, $\mathbf x^\star=(100;150)$,
  stínové ceny $\mathbf y^\star=(10;3)$ a hodnota 2600 Kč ověřené asserty
  v [`kod/obrazky_dualita_lp.py`](kod/obrazky_dualita_lp.py);
- `lagrangian-sedlo.svg` — vrstevnice lagrangiánu dávky $(x-3)^2+\mu(x-2)$;
  sedlo $(2;2)$ s hodnotou 1 ověřené assertem v
  [`kod/obrazky_dualita_lp.py`](kod/obrazky_dualita_lp.py);
- `dualni-mezera.gif` — dávka $\min(x-3)^2$ za $x\leq2$ řešená bariérou;
  body $x_t$ se počítají numericky a mezera $f_t-d_t=1/t$ je v generátoru
  ověřená assertem.

Výpisy kódu na slidech jsou **zkrácené** — proměnné i formát tisku se od
skriptů liší tak, aby se vešly na slide. Čísla v nich jsou ale doslova ta, která
skripty vytisknou.
