"""Télécharge des images candidates depuis Wikimedia Commons, AVEC leurs licences.

Pourquoi ce script ?
  Le sujet exige des images "autorisées" et une trace auteur / URL / licence.
  L'API de Commons renvoie ces métadonnées : on les écrit directement dans
  data/sources.csv au moment du téléchargement, donc rien n'est perdu.

Usage :
  python data/download_commons.py --category "Doors in Morocco" --depth 1 --max 150

Ensuite : regarder les images de data/raw/ et SUPPRIMER celles qui ne conviennent
pas (plusieurs portes, personnes, filigrane, porte coupée, flou...). Garder 30 à 60.
"""
import argparse
import csv
import html
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia demande un User-Agent identifiable (sinon requêtes refusées)
HEADERS = {"User-Agent": "EtudiantLoRAProjet/1.0 (projet universitaire; contact via GitHub)"}

# Licences acceptées : domaine public et Creative Commons sans clause NC/ND.
ALLOWED = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [\d.]+|cc by(-sa)?)", re.I)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
SOURCES_CSV = ROOT / "sources.csv"
FIELDS = ["file_name", "title", "author", "license", "license_url", "source_url"]


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def list_subcategories(category: str) -> list[str]:
    data = api_get({"action": "query", "list": "categorymembers",
                    "cmtitle": f"Category:{category}", "cmtype": "subcat", "cmlimit": 100})
    return [m["title"].removeprefix("Category:") for m in data["query"]["categorymembers"]]


def list_files(category: str, thumb_width: int):
    """Itère sur les fichiers d'une catégorie avec URL de vignette + métadonnées."""
    params = {"action": "query", "generator": "categorymembers",
              "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": 50,
              "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
              "iiurlwidth": thumb_width}
    while True:
        data = api_get(params)
        for page in data.get("query", {}).get("pages", {}).values():
            if "imageinfo" in page:
                yield page["title"], page["imageinfo"][0]
        if "continue" not in data:
            break
        params.update(data["continue"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", default=None,
                    help="Catégorie Commons (répétable). Défaut : 'Doors in Morocco'")
    ap.add_argument("--depth", type=int, default=1, help="Profondeur des sous-catégories")
    ap.add_argument("--max", type=int, default=150, help="Nombre max d'images candidates")
    ap.add_argument("--min-size", type=int, default=600, help="Côté minimal de l'original (px)")
    ap.add_argument("--thumb-width", type=int, default=1024)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    categories = args.category or ["Doors in Morocco"]

    # Exploration des sous-catégories (ex : Doors in Fes, Doors in Marrakesh...)
    frontier, seen_cats = list(categories), set(categories)
    for _ in range(args.depth):
        nxt = []
        for c in frontier:
            for sub in list_subcategories(c):
                if sub not in seen_cats:
                    seen_cats.add(sub)
                    nxt.append(sub)
        frontier = nxt
    print(f"{len(seen_cats)} catégories explorées")

    existing = set()
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, encoding="utf-8") as f:
            existing = {row["title"] for row in csv.DictReader(f)}

    new_rows, skipped = [], {"licence": 0, "taille": 0, "format": 0}
    for cat in sorted(seen_cats):
        for title, info in list_files(cat, args.thumb_width):
            if len(existing) + len(new_rows) >= args.max:
                break
            if title in existing or any(r["title"] == title for r in new_rows):
                continue
            if info.get("mime") not in ("image/jpeg", "image/png"):
                skipped["format"] += 1
                continue
            if min(info.get("width", 0), info.get("height", 0)) < args.min_size:
                skipped["taille"] += 1
                continue
            meta = info.get("extmetadata", {})
            license_name = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not ALLOWED.match(license_name) or re.search(r"\b(nc|nd)\b", license_name, re.I):
                skipped["licence"] += 1
                continue

            idx = len(existing) + len(new_rows)
            file_name = f"commons_{idx:04d}.jpg"
            img = requests.get(info.get("thumburl", info["url"]), headers=HEADERS, timeout=60)
            img.raise_for_status()
            (RAW_DIR / file_name).write_bytes(img.content)
            new_rows.append({
                "file_name": file_name,
                "title": title,
                "author": strip_html(meta.get("Artist", {}).get("value", "inconnu"))[:200],
                "license": license_name,
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "source_url": info.get("descriptionurl", ""),
            })
            print(f"  + {file_name}  [{license_name}]  {title}")
            time.sleep(0.2)  # rester poli avec l'API

    write_header = not SOURCES_CSV.exists()
    with open(SOURCES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)

    print(f"\n{len(new_rows)} images ajoutées dans {RAW_DIR}")
    print(f"Ignorées : {skipped}")
    print("Étape suivante : trier à la main data/raw/ puis lancer data/prepare.py")


if __name__ == "__main__":
    main()
