"""Télécharge des images candidates depuis Wikimedia Commons, AVEC leurs licences.

Pourquoi ce script ?
  Le sujet exige des images "autorisées" et une trace auteur / URL / licence.
  L'API de Commons renvoie ces métadonnées : on les écrit directement dans
  data/sources.csv au moment du téléchargement, donc rien n'est perdu.

Usage :
  python data/download_commons.py --category "Doors in Morocco" --depth 2 --max 100

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
REJECTED = RAW_DIR / "_rejected.txt"   # dans raw/ -> sauvegardé sur Drive avec les images
FIELDS = ["file_name", "title", "author", "license", "license_url", "source_url"]


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def api_get(params: dict) -> dict:
    return fetch(API, params={**params, "format": "json"}).json()


def fetch(url: str, params: dict | None = None, tries: int = 6) -> requests.Response:
    """GET avec nouvelles tentatives : en cas de 429 (trop de requêtes) ou d'erreur
    serveur, on attend de plus en plus longtemps (backoff exponentiel) au lieu de planter."""
    wait = 5
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            print(f"    réseau : {e} -> nouvel essai dans {wait}s")
        else:
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            # Wikimedia indique parfois combien de secondes attendre
            wait = int(r.headers.get("Retry-After", wait)) if r.headers.get(
                "Retry-After", "").isdigit() else wait
            print(f"    HTTP {r.status_code} -> pause de {wait}s (essai {attempt + 1}/{tries})")
        time.sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(f"Échec après {tries} essais : {url}")


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
    ap.add_argument("--max", type=int, default=150, help="Nombre max de NOUVELLES images à ajouter")
    ap.add_argument("--min-size", type=int, default=600, help="Côté minimal de l'original (px)")
    # Tailles "standard" de Wikimedia : 500, 960, 1280... Les autres sont fortement limitées.
    ap.add_argument("--thumb-width", type=int, default=960)
    ap.add_argument("--delay", type=float, default=1.5, help="Pause entre deux images (s)")
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

    # Titres déjà vus = gardés (sources.csv) + écartés au tri (raw/_rejected.txt)
    # -> une image supprimée au tri n'est JAMAIS re-téléchargée.
    existing, used_names = set(), {p.name for p in RAW_DIR.glob("commons_*.jpg")}
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["title"])
                used_names.add(row["file_name"])
    if REJECTED.exists():
        existing |= set(REJECTED.read_text(encoding="utf-8").splitlines())
    # Numéro suivant libre : on n'écrase jamais un fichier existant
    next_idx = 1 + max([int(n[8:12]) for n in used_names if n[8:12].isdigit()], default=-1)
    print(f"{len(existing)} titres déjà connus, prochain fichier : commons_{next_idx:04d}.jpg")

    new_rows, skipped = [], {"licence": 0, "taille": 0, "format": 0, "erreur": 0}
    for cat in sorted(seen_cats):
        for title, info in list_files(cat, args.thumb_width):
            if len(new_rows) >= args.max:
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

            file_name = f"commons_{next_idx:04d}.jpg"
            try:
                img = fetch(info.get("thumburl", info["url"]))
            except Exception as e:  # une image ratée ne doit pas arrêter toute la collecte
                print(f"  ! ignorée ({e}) : {title}")
                skipped["erreur"] += 1
                continue
            (RAW_DIR / file_name).write_bytes(img.content)
            row = {
                "file_name": file_name,
                "title": title,
                "author": strip_html(meta.get("Artist", {}).get("value", "inconnu"))[:200],
                "license": license_name,
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "source_url": info.get("descriptionurl", ""),
            }
            new_rows.append(row)
            next_idx += 1
            write_header = not SOURCES_CSV.exists()
            with open(SOURCES_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(f"  + {file_name}  [{license_name}]  {title}")
            time.sleep(args.delay)  # rester poli avec Wikimedia

    print(f"\n{len(new_rows)} images ajoutées dans {RAW_DIR}")
    print(f"Ignorées : {skipped}")
    print("Étape suivante : trier à la main data/raw/ puis lancer data/prepare.py")


if __name__ == "__main__":
    main()
