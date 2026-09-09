"""Slozi reporty vsech ukazek do JEDINEHO samostatneho HTML souboru.

Duvod: reporty v cviceni/C1/reporty/ odkazuji na obrazky, styl a na sebe
navzajem relativnimi cestami. Kdyz se jeden takovy soubor otevre v prohlizeci
bezicim v sandboxu (Flatpak) nebo se posle nekomu mailem, sourozenecke soubory
nejsou videt a stranka je bez obrazku. Vysledny cviceni/C1/ukazky.html ma
v sobe vsechno: styl, obrazky jako data URI, MathJax i vsechny ukazky.

Ukazky nejsou pod sebou, ale v ZALOZKACH - prepina se mezi nimi lista nahore,
zobrazena je vzdy jen jedna. Odkazy s kotvou (#raketa, #plavcik-hriste) zalozku
samy prepnou, takze funguji i mezi ukazkami.

Posledni zalozka "Pokracovani cviceni" neni ukazka, ale zbytek hodiny - teorie,
tabulovy vypocet optimalniho hematokritu pres derivaci, overeni konvexity
a prechod do notebooku. V liste je oddelena carou, aby bylo videt, ze rada
serazena podle poctu promennych konci topologii.

Pasaze urcene vyucujicimu jsou v reportech oznacene komentari
<!-- vyucujici --> ... <!-- /vyucujici -->:

    uv run python cviceni/C1/kod/zabal_ukazky.py
        cviceni/C1/ukazky.html - vsechno vcetne poznamek, prepinac v liste je
        umi schovat i bez prestavby

    uv run python cviceni/C1/kod/zabal_ukazky.py --pro-studenty
        cviceni/C1/ukazky-pro-studenty.html - poznamky pro vyucujiciho jsou
        ze souboru uplne vyhozene, ne jen skryte
"""

import base64
import re
import sys
import urllib.request
from pathlib import Path

KOREN = Path("cviceni/C1")
REPORTY = KOREN / "reporty"
PRO_STUDENTY = "--pro-studenty" in sys.argv
VYSTUP = KOREN / ("ukazky-pro-studenty.html" if PRO_STUDENTY else "ukazky.html")
MATHJAX = REPORTY / "vendor" / "mathjax.js"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# poradi sekci ve vysledne strance (podle poctu promennych, vzestupne);
# klic = jmeno souboru bez pripony
# POZN.: "hematokrit" je archivovany (viz cviceni/C1/ukazka-hematokrit.md) -
# nesl totez sdeleni jako dychani. Vratit se da pridanim "hematokrit" zpet
# za "plavcik"; report, obrazek i notebook zustavaji na disku a funkcni.
UKAZKY = ["plavcik", "dychani", "brachistochrona", "radioterapie",
          "raketa", "topologie"]
# za radou ukazek jeste zbytek hodiny: teorie, tabulovy vypocet pres derivaci,
# konvexita, prechod do notebooku. Neni to ukazka, proto stoji mimo UKAZKY
# a v liste je oddelena.
DALSI = ["pokracovani"]
NADPISY = {
    "plavcik": "Plavčík a tonoucí",
    "hematokrit": "Optimální hematokrit",   # archivovano, nadpis ceka na navrat
    "dychani": "Optimální dýchání",
    "brachistochrona": "Brachistochrona",
    "radioterapie": "Plánování radioterapie",
    "raketa": "Měkké přistání rakety",
    "topologie": "Topologická optimalizace",
    "pokracovani": "Pokračování cvičení",
}

MIME = {".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def data_uri(cesta: Path) -> str:
    """Nacte obrazek a vrati ho jako data URI."""
    mime = MIME[cesta.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(cesta.read_bytes()).decode()


def telo(html: str) -> str:
    """Vytahne obsah <body>."""
    return re.search(r"<body[^>]*>(.*)</body>", html, re.S).group(1).strip()


def zabal_obrazky(text: str) -> str:
    """Nahradi relativni odkazy na obrazky vlozenymi daty."""
    def nahrada(m):
        soubor = KOREN / "obrazky" / Path(m.group(1)).name
        return f'src="{data_uri(soubor)}"'
    return re.sub(r'src="\.\./obrazky/([^"]+)"', nahrada, text)


def zabal_odkazy(text: str) -> str:
    """Odkazy na lokalni soubory (.md, .py, .html) zmeni na prosty text.

    V samostatnem HTML by nefungovaly - cilove soubory u nej nejsou.
    Interni kotvy (#...) a webove adresy zustavaji.
    """
    def nahrada(m):
        cil, popis = m.group(1), m.group(2)
        if cil.startswith(("#", "http://", "https://", "mailto:")):
            return m.group(0)
        soubor, _, kotva = cil.partition("#")
        # odkaz na sourozenecky report -> kotva; odkaz jinam (../, jine cviceni)
        # zadnou sekci v teto strance nema, takze z nej zbyde prosty text
        if soubor.endswith(".html") and "/" not in soubor:
            return f'<a href="#{kotva or Path(soubor).stem}">{popis}</a>'
        return f"<code>{popis}</code>"
    return re.sub(r'<a href="([^"]+)">(.*?)</a>', nahrada, text, flags=re.S)


def bez_vyucujiciho(text: str) -> str:
    """Vyhodi bloky <!-- vyucujici --> ... <!-- /vyucujici --> i vlozene spany."""
    text = re.sub(r"<!-- vyucujici -->.*?<!-- /vyucujici -->", "", text, flags=re.S)
    return re.sub(r'<span class="pro-vyucujiciho vlozene">.*?</span>', "", text, flags=re.S)


def uprav(text: str) -> str:
    text = zabal_odkazy(zabal_obrazky(text))
    return bez_vyucujiciho(text) if PRO_STUDENTY else text


if not MATHJAX.exists():                          # neni ve verzovani, stahne se
    MATHJAX.parent.mkdir(parents=True, exist_ok=True)
    print(f"stahuji MathJax z {MATHJAX_URL}")
    urllib.request.urlretrieve(MATHJAX_URL, MATHJAX)

styl = (REPORTY / "styl.css").read_text(encoding="utf-8")
styl += """
/* --- doplnky pro jednosouborovou verzi se zalozkami --- */
body { max-width: 940px; }
nav.zalozky { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap;
  align-items: center; gap: .35rem; padding: .5rem 0; margin-bottom: 1.6rem;
  background: rgba(255,255,255,.97); border-bottom: 2px solid var(--ram); }
nav.zalozky button { font: inherit; font-size: .92rem; padding: .35rem .8rem; cursor: pointer;
  border: 1px solid transparent; border-radius: 4px 4px 0 0; background: none; color: var(--akcent); }
nav.zalozky button:hover { background: var(--box); }
nav.zalozky button.aktivni { background: var(--akcent); color: #fff; font-weight: 600; }
nav.zalozky .prepinac-vyucujici { margin-left: auto; display: flex; align-items: center; gap: .4rem;
  font-size: .85rem; color: var(--tlum); }
nav.zalozky .prepinac-vyucujici input { accent-color: var(--akcent2); }
nav.zalozky .oddelovac { width: 1px; height: 1.5rem; margin: 0 .35rem; background: var(--ram); }
nav.zalozky button.dalsi { color: var(--akcent2); }
nav.zalozky button.dalsi.aktivni { background: var(--akcent2); color: #fff; }
section.ukazka { scroll-margin-top: 6rem; }   /* lista se od osmi zalozek lame do dvou radku */
section.ukazka > header { margin-top: 0; }
.nahoru { display: block; margin-top: 2.5rem; font-size: .85rem; }
@media print { nav.zalozky { display: none; } section.ukazka[hidden] { display: block !important; } }
"""

casti = ['<section class="ukazka" id="prehled">\n'
         + uprav(telo((REPORTY / "index.html").read_text(encoding="utf-8")))
         + "\n</section>"]

for jmeno in UKAZKY + DALSI:
    obsah = uprav(telo((REPORTY / f"{jmeno}.html").read_text(encoding="utf-8")))
    casti.append(
        f'<section class="ukazka" id="{jmeno}" hidden>\n{obsah}\n'
        f'<a class="nahoru" href="#prehled">↑ zpět na přehled ukázek</a>\n</section>'
    )

prepinac = "" if PRO_STUDENTY else (
    '<label class="prepinac-vyucujici" title="Skryje pasáže označené „pro vyučujícího“">'
    '<input type="checkbox" id="prepinac-vyucujici" checked> poznámky pro vyučujícího</label>')
nav = ('<nav class="zalozky" role="tablist">'
       + '<button data-cil="prehled" role="tab">Přehled</button>'
       + "".join(f'<button data-cil="{j}" role="tab">{NADPISY[j]}</button>' for j in UKAZKY)
       + '<span class="oddelovac"></span>'
       + "".join(f'<button data-cil="{j}" role="tab" class="dalsi">{NADPISY[j]}</button>'
                 for j in DALSI)
       + prepinac + "</nav>")

# prepinani zalozek: bez zalozky by byly vsechny ukazky pod sebou na jedne strance
zalozky_js = """
(function () {
  "use strict";
  var tlacitka = Array.prototype.slice.call(document.querySelectorAll("nav.zalozky button"));
  var sekce = {};
  Array.prototype.forEach.call(document.querySelectorAll("section.ukazka"), function (s) {
    sekce[s.id] = s;
  });

  function prepni(id, posun) {
    if (!sekce[id]) { return false; }
    Object.keys(sekce).forEach(function (k) { sekce[k].hidden = (k !== id); });
    tlacitka.forEach(function (b) {
      var akt = b.getAttribute("data-cil") === id;
      b.classList.toggle("aktivni", akt);
      b.setAttribute("aria-selected", akt ? "true" : "false");
    });
    if (posun !== false) { window.scrollTo(0, 0); }
    return true;
  }

  function zHashe(posun) {
    var h = decodeURIComponent(location.hash.slice(1));
    if (sekce[h]) { return prepni(h, posun); }
    var cil = h && document.getElementById(h);
    if (cil) {
      var s = cil.closest("section.ukazka");
      if (s) { prepni(s.id, false); cil.scrollIntoView(); return true; }
    }
    return prepni("prehled", posun);
  }

  tlacitka.forEach(function (b) {
    b.addEventListener("click", function () {
      var cil = b.getAttribute("data-cil");
      prepni(cil);
      history.replaceState(null, "", "#" + cil);
    });
  });
  window.addEventListener("hashchange", function () { zHashe(true); });

  var prepinac = document.getElementById("prepinac-vyucujici");
  if (prepinac) {
    prepinac.addEventListener("change", function (e) {
      document.body.classList.toggle("bez-vyucujiciho", !e.target.checked);
    });
  }

  // zalozky se zapnou az po vysazeni vzorcu, aby MathJax meril na viditelnem obsahu
  function start() { zHashe(false); }
  if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(start).catch(start);
  } else {
    start();
  }
})();
"""

stranka = f"""<!doctype html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ukázky pro první cvičení MPC-OMM{" (studentská verze)" if PRO_STUDENTY else ""}</title>
<style>
{styl}
</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }},
                    options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'] }} }};
</script>
<script>
{MATHJAX.read_text(encoding="utf-8")}
</script>
</head>
<body id="nahoru">
{nav}
{"".join(casti)}
<script>
{zalozky_js}
</script>
<script>
{(REPORTY / "lupa.js").read_text(encoding="utf-8")}
</script>
</body>
</html>
"""

VYSTUP.write_text(stranka, encoding="utf-8")
print(f"hotovo: {VYSTUP} ({VYSTUP.stat().st_size/1e6:.1f} MB)"
      + (" — bez poznámek pro vyučujícího" if PRO_STUDENTY else ""))
