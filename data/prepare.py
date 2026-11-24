"""Prépare le mini-jeu : 512x512, suppression des doublons, découpage train / heldout.

Usage :
  python data/prepare.py --heldout 8 --mode crop

Entrée  : data/raw/*.jpg|png (déjà triées à la main) + data/sources.csv
Sorties : data/processed/train/*.png, data/processed/heldout/*.png
          data/captions.csv  (à compléter / vérifier À LA MAIN)
          data/prepare_report.txt

Choix expliqués :
- 512x512 : résolution native de Stable Diffusion 1.5. Entraîner à une autre taille
  dégrade la qualité ou coûte plus de mémoire.
- Doublons : deux photos quasi identiques comptent "double" et poussent le modèle
  à les recopier (surapprentissage). On les détecte avec un hash perceptuel (dHash).
- heldout : 5 à 10 images jamais vues -> servent à mesurer la généralisation.
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"
CAPTIONS = ROOT / "captions.csv"


def dhash(img: Image.Image, size: int = 8) -> int:
    """Hash perceptuel : compare la luminosité de pixels voisins sur une vignette 9x8.
    Deux images visuellement proches ont des hash qui diffèrent de peu de bits."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def to_square(img: Image.Image, res: int, mode: str) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    if mode == "crop":
        # Recadrage centré : simple, mais coupe le haut/bas des portes très hautes
        return ImageOps.fit(img, (res, res), Image.LANCZOS, centering=(0.5, 0.45))
    # mode "pad" : garde toute la porte, fond = version floutée de l'image
    bg = ImageOps.fit(img, (res, res), Image.LANCZOS).filter(ImageFilter.GaussianBlur(25))
    fg = ImageOps.contain(img, (res, res), Image.LANCZOS)
    bg.paste(fg, ((res - fg.width) // 2, (res - fg.height) // 2))
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--heldout", type=int, default=8, help="Nb d images réservées au test (max 20 %% du jeu)")
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="Distance de Hamming max pour considérer deux images comme doublons")
    ap.add_argument("--min-side", type=int, default=400)
    ap.add_argument("--trigger", default="mrcdoor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    report, kept, hashes = [], [], []
    for p in files:
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # fichier corrompu
            report.append(f"ILLISIBLE  {p.name}: {e}")
            continue
        if min(img.size) < args.min_side:
            report.append(f"TROP PETITE {p.name} {img.size}")
            continue
        h = dhash(img)
        dup = next((k for k, hk in zip(kept, hashes) if hamming(h, hk) <= args.dup_threshold), None)
        if dup is not None:
            report.append(f"DOUBLON    {p.name} ~ {dup.name}")
            continue
        kept.append(p)
        hashes.append(h)

    random.Random(args.seed).shuffle(kept)
    n_held = min(args.heldout, max(len(kept) // 5, 1))
    splits = {"heldout": kept[:n_held], "train": kept[n_held:]}

    # Légendes existantes conservées (on ne réécrit jamais le travail manuel)
    old_captions = {}
    if CAPTIONS.exists():
        with open(CAPTIONS, encoding="utf-8") as f:
            old_captions = {r["source"]: r["caption"] for r in csv.DictReader(f)}

    rows = []
    for split, paths in splits.items():
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        for i, p in enumerate(sorted(paths)):
            name = f"{split}_{i:03d}.png"
            to_square(Image.open(p), args.resolution, args.mode).save(d / name)
            caption = old_captions.get(p.name, f"a photo of a {args.trigger}, TODO décrire")
            rows.append({"file_name": name, "split": split, "source": p.name, "caption": caption})

    with open(CAPTIONS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "split", "source", "caption"])
        w.writeheader()
        w.writerows(rows)

    summary = (f"{len(files)} images brutes -> {len(kept)} gardées "
               f"({len(splits['train'])} train, {len(splits['heldout'])} heldout)")
    report.insert(0, summary)
    (ROOT / "prepare_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(summary)
    print("\n".join(report[1:]) or "Aucune image écartée.")
    if len(splits["train"]) < 20:
        print("ATTENTION : moins de 20 images d'entraînement, le sujet en demande 20 à 50.")
    print(f"Étape suivante : compléter les légendes dans {CAPTIONS} "
          "(ou lancer src/auto_caption.py puis corriger).")


if __name__ == "__main__":
    main()
