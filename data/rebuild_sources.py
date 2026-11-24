"""Reconstruit data/sources.csv si le fichier a été perdu (ex : session Colab réinitialisée).

Principe : on relit la liste des images candidates sur Wikimedia Commons (métadonnées seulement),
on télécharge une MINI-vignette de 120 px de chacune, et on la compare à chaque image
de data/raw/ avec un hash perceptuel. Deux images identiques à des tailles différentes
ont presque le même hash -> on retrouve auteur, licence et URL de chaque fichier.

Usage :
  python data/rebuild_sources.py --category "Doors in Morocco" --depth 1
"""
import argparse
import csv

from PIL import Image
import io

from download_commons import (ALLOWED, FIELDS, RAW_DIR, SOURCES_CSV, fetch, list_files,
                              list_subcategories, strip_html)
from prepare import dhash, hamming
import re
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--min-size", type=int, default=600)
    ap.add_argument("--threshold", type=int, default=10, help="Distance max pour accepter un match")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    raw = {p.name: dhash(Image.open(p)) for p in sorted(RAW_DIR.glob("*.jpg"))}
    print(f"{len(raw)} images dans {RAW_DIR}")
    if not raw:
        raise SystemExit("Aucune image dans data/raw : vérifier le lien vers Google Drive.")

    categories = args.category or ["Doors in Morocco"]
    frontier, seen = list(categories), set(categories)
    for _ in range(args.depth):
        nxt = []
        for c in frontier:
            for sub in list_subcategories(c):
                if sub not in seen:
                    seen.add(sub)
                    nxt.append(sub)
        frontier = nxt

    found, unmatched = {}, set(raw)
    for cat in sorted(seen):
        if not unmatched:
            break
        for title, info in list_files(cat, 120):
            if not unmatched:
                break
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if min(info.get("width", 0), info.get("height", 0)) < args.min_size:
                continue
            meta = info.get("extmetadata", {})
            lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not ALLOWED.match(lic) or re.search(r"\b(nc|nd)\b", lic, re.I):
                continue
            try:
                thumb = Image.open(io.BytesIO(fetch(info.get("thumburl", info["url"])).content))
            except Exception as e:
                print(f"  ! {title}: {e}")
                continue
            h = dhash(thumb)
            best = min(unmatched, key=lambda f: hamming(h, raw[f]))
            if hamming(h, raw[best]) <= args.threshold:
                unmatched.discard(best)
                found[best] = {
                    "file_name": best, "title": title,
                    "author": strip_html(meta.get("Artist", {}).get("value", "inconnu"))[:200],
                    "license": lic,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "source_url": info.get("descriptionurl", ""),
                }
                print(f"  = {best}  <-  {title}  [{lic}]")
            time.sleep(args.delay)

    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(found[k] for k in sorted(found))
    print(f"\n{len(found)}/{len(raw)} images retrouvées -> {SOURCES_CSV}")
    if unmatched:
        print("Non retrouvées (à supprimer de data/raw ou à compléter à la main) :")
        for f in sorted(unmatched):
            print("  -", f)


if __name__ == "__main__":
    main()
