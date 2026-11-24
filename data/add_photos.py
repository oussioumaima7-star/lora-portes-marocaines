"""Ajoute les photos prises par l'équipe (raw/equipe_*.jpg) dans sources.csv.

Usage :
  python data/add_photos.py --author "Oumaima Oussi"

Les photos de l'équipe sont publiées sous licence CC BY 4.0 (auteur = membre qui a pris la photo).
Si plusieurs membres ont pris des photos, lancer la commande une fois par auteur avec --prefix,
ex : equipe_oumaima_001.jpg  ->  --prefix equipe_oumaima --author "Oumaima Oussi"
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SOURCES = ROOT / "sources.csv"
FIELDS = ["file_name", "title", "author", "license", "license_url", "source_url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True)
    ap.add_argument("--prefix", default="equipe")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="Réduit les photos de téléphone (souvent 4000 px) pour alléger Drive")
    args = ap.parse_args()

    known = set()
    if SOURCES.exists():
        with open(SOURCES, encoding="utf-8") as f:
            known = {r["file_name"] for r in csv.DictReader(f)}

    photos = sorted(p for p in RAW.iterdir()
                    if p.name.lower().startswith(args.prefix.lower())
                    and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    rows = []
    for p in photos:
        if p.suffix.lower() == ".heic":
            print(f"  ! {p.name} : format HEIC (iPhone) non lu -> l'exporter en JPG")
            continue
        # Normalise : orientation EXIF, taille raisonnable, extension .jpg
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        out = p.with_suffix(".jpg")
        img.save(out, quality=92)
        if out != p:
            p.unlink()
        if out.name in known:
            continue
        rows.append({"file_name": out.name, "title": f"Photo de l'équipe ({out.stem})",
                     "author": args.author, "license": "CC BY 4.0",
                     "license_url": "https://creativecommons.org/licenses/by/4.0/",
                     "source_url": ""})

    write_header = not SOURCES.exists()
    with open(SOURCES, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} photos ajoutées à sources.csv (auteur : {args.author})")


if __name__ == "__main__":
    main()
