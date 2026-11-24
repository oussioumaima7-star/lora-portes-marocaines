"""Écarte des images après le tri : supprime le fichier, retire sa ligne de sources.csv
et note son titre dans raw/_rejected.txt (pour ne jamais la re-télécharger).

Usage : python data/reject.py 0 1 2 15 23 o3 o7   (commons_XXXX.jpg ; préfixe "o" = openverse_XXXX.jpg)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SOURCES = ROOT / "sources.csv"
REJECTED = RAW / "_rejected.txt"


def main():
    # "12" -> commons_0012.jpg ; "o12" -> openverse_0012.jpg
    names = {f"openverse_{int(n[1:]):04d}.jpg" if n.lower().startswith("o") else f"commons_{int(n):04d}.jpg"
             for n in sys.argv[1:]}
    if not names:
        raise SystemExit("Donner les numéros des images à écarter, ex : python data/reject.py 3 7 12")

    with open(SOURCES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    keep = [r for r in rows if r["file_name"] not in names]
    rejected_titles = [t for r in rows if r["file_name"] in names for t in (r["title"], r["source_url"]) if t]

    with open(REJECTED, "a", encoding="utf-8") as f:
        for t in rejected_titles:
            f.write(t + "\n")
    with open(SOURCES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)

    removed = 0
    for n in sorted(names):
        p = RAW / n
        if p.exists():
            p.unlink()
            removed += 1
    print(f"{removed} fichiers supprimés, {len(rejected_titles)} titres ajoutés à {REJECTED.name}")
    print(f"Il reste {len(keep)} images dans sources.csv "
          f"et {len(list(RAW.glob('*.jpg')))} fichiers dans raw/")


if __name__ == "__main__":
    main()
