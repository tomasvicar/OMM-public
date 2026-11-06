# Zdroje – S18 šifrovaný deník lékárníka

## Texty (díla volná, e-knihy z Wikizdrojů pod CC BY-SA 3.0)

Texty se stahují z Wikizdrojů (<https://cs.wikisource.org>) nástrojem
ws-export (`https://ws-export.wmcloud.org/?lang=cs&page=<název>&format=txt`)
(staženo 5. 11. 2026). Wikizdroje u každého díla uvádějí licenci
*PD old 70* – autor zemřel před více než 70 lety, samotné dílo je v ČR
i EU volné (public domain). Přepis a úprava e-knihy jsou ale dílem
dobrovolníků Wikizdrojů a exportovaná e-kniha se šíří pod licencí
**Creative Commons Uveďte původ – Zachovejte licenci 3.0 Unported**
(CC BY-SA 3.0, <https://creativecommons.org/licenses/by-sa/3.0/>) nebo
GNU FDL; přispěvatelé jsou vyjmenovaní na konci každé e-knihy (oddíl
„O e-knize“).

Archiv `data-S18.zip` obsahuje **celé texty** čtyř děl korpusu
(`data/korpus/`) v podobě, v jaké je export vydal, včetně údajů
o vydání, licenci a přispěvatelích – tím je splněna povinnost uvést
původ. Tabulka tetragramů a šifrované úseky jsou z těchto textů odvozené
(velká písmena A–Z, zašifrování) a šíří se pod stejnou licencí
CC BY-SA 3.0. Licence a odkazy jsou zopakované v `data/README.txt`
v archivu.

**Korpus pro tabulku tetragramů** (`data/tetragramy.npy`),
celkem 1,24 mil. písmen:

| dílo | autor (úmrtí) | vydání podle Wikizdrojů |
|---|---|---|
| Babička | Božena Němcová (1862) | 7. vyd., Česká grafická unie, Praha 1924 |
| Povídky malostranské | Jan Neruda (1891) | 2. vyd., Ed. Valečka, Praha 1885 |
| Krakatit | Karel Čapek (1938) | Továrna na absolutno; Krakatit |
| Válka s mloky | Karel Čapek (1938) | Československý spisovatel, Praha 1981 (Spisy, sv. 9) |

**Testovací (zašifrované) texty** – jiné dílo i autor než korpus, aby
tabulka „neznala“ šifrovaný text:

| dílo | autor (úmrtí) | vydání podle Wikizdrojů |
|---|---|---|
| Filosofská historie | Alois Jirásek (1930) | J. Otto, Praha 1898 |

Úseky (40–1000 písmen; Playfair 600 a 1000) se berou z náhodných, nepřekrývajících se míst
(pevný seed); místo je zapsané v pravdě u klíče `zdroj` (v archivu
`data/pravda-useky.json`, jen úseky S…).

**Recepty z deníku** (instance R01–R10; otevřený text má jen vyučující,
v archivu jsou jen zašifrované) jsou napsané
pro tuto úlohu v archaizujícím stylu; nejde o skutečné recepty ani
o lékařské doporučení (dávkování je smyšlené). Volné užití.

Projekt Gutenberg byl zvažován, ale česky má jen asi tucet titulů, z nichž
část jsou překlady nebo díla autorů zemřelých po roce 1956 (např.
F. Omelka) – proto Wikizdroje.

## Úprava textu

Velká písmena A–Z bez diakritiky (rozklad Unicode NFD a odstranění
akcentů), bez mezer a interpunkce; „CH“ zůstává jako dvě písmena. U šifry
Playfair se W píše jako V (čtverec má 25 polí), dvojice stejných písmen se
oddělí X, lichý konec se doplní X – klasická pravidla Playfairu.

Tabulka tetragramů: $\log_{10}$ relativní četnosti každého ze $26^4$
tetragramů v korpusu; tetragram, který se v korpusu nevyskytl, dostane
$\log_{10}(0{,}01/N)$ (stokrát vzácnější než jednou viděný, $N$ = počet
tetragramů korpusu).

## Literatura a pozadí

- Jakobsen T. (1995): *A fast method for cryptanalysis of substitution
  ciphers.* Cryptologia 19(3), 265–274, doi:10.1080/0161-119591883944 –
  hill climbing nad klíčem
  substituce se skórem z n-gramů (postupné zpřesňování odhadu klíče).
- Cowan M. J. (2008): *Breaking short Playfair ciphers with the simulated
  annealing algorithm.* Cryptologia 32(1), 71–83,
  doi:10.1080/01611190701743658 – žíhání pro krátké texty v Playfairu,
  skóre z tetragramů.
- Lasry G. (2018): *A Methodology for the Cryptanalysis of Classical
  Ciphers with Search Metaheuristics.* kassel university press,
  ISBN 978-3-7376-0458-1 (disertace, Univerzita Kassel 2017, open access
  v repozitáři KOBRA) – metodika luštění klasických šifer hill climbingem
  a žíháním, návrh skórovací funkce a sousedství.
- Shannon C. E. (1949): *Communication theory of secrecy systems.* Bell
  System Technical Journal 28(4), 656–715 – vzdálenost jednoznačnosti
  (*unicity distance*): pod určitou délkou má šifrový text víc
  „smysluplných“ rozluštění.
- Praktický popis tetragramového skóre: <http://practicalcryptography.com/>
  (Cryptanalysis – Quadgram Statistics as a Fitness Measure).

Bibliografické údaje Jakobsena, Cowana a Lasryho jsou ověřené
(5. 11. 2026) proti záznamům Crossref (DOI), KOBRA Kassel a DBLP; obsah
je shrnutý podle abstraktů a sekundárních popisů, plné texty
neprocházené.

## Archiv

`data-S18.zip` obsahuje `simulator.py` a ve složce `data/` soubory
`instance.json`, `pravda-useky.json` (jen pro vyhodnocení, pravda
k úsekům S…), `tetragramy.npy`, `README.txt` a v `korpus/` čtyři díla
korpusu ve stažené podobě. Celé dílo, ze kterého jsou šifrované úseky
(Filosofská historie), v archivu není; otevřený text mají jen kalibrační
úseky S…, recepty R… a listy Playfairu P… pravdu v archivu nemají.
