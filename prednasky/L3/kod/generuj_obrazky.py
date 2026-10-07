"""Vytvoří všechny ilustrace třetí přednášky.

Spuštění:

    cd prednasky/L3/kod && uv run python generuj_obrazky.py

Obrázky jsou kvůli přehlednosti rozdělené do souborů ``obrazky_blok*.py``;
čísla v názvech pocházejí z dřívějšího pořadí bloků (před přeskládáním
5. 10. 2026), aktuální pořadí v přednášce udává seznam ``BLOKY``. Tenhle
skript je jen spustí všechny za sebou. Společnou paletu, cestu k výstupu a závazná čísla hlavního
příkladu (pekárna jako QP) drží ``spolecne.py``.

Nastavením proměnné prostředí ``OMM_NAHLEDY`` na adresář se vedle SVG uloží
i rastrové náhledy pro rychlou vizuální kontrolu; do repozitáře nepatří.
"""

import runpy

BLOKY = (
    ("obrazky_blok1", "od přímek k zakřivenému světu"),
    ("obrazky_blok2", "katalog kvadratických úloh"),
    ("obrazky_blok3", "bez omezení: gradient a Hessián"),
    ("obrazky_blok7", "omezení jako pokuta a zeď"),
    ("obrazky_blok4_cena", "obě cesty vedou k jedné ceně"),
    ("obrazky_blok5_poplatek", "lagrangián: lineární poplatek"),
    ("obrazky_blok4", "lagrangián: rovnost a multiplikátor"),
    ("obrazky_blok5", "KKT: pravidla pro správnou cenu"),
    ("obrazky_blok6_dualita", "dualita: hledání správné ceny"),
    ("obrazky_dualita_lp", "dualita: duál LP a duální mezera"),
)


def main() -> None:
    for jmeno, popis in BLOKY:
        print(f"→ {jmeno}: {popis}")
        runpy.run_module(jmeno, run_name="__main__")


if __name__ == "__main__":
    main()
