"""Pré-remplit les légendes avec BLIP, à VÉRIFIER ensuite à la main.

Usage : python -m src.auto_caption [--overwrite]

Pourquoi des légendes ?
  La LoRA apprend l'association texte -> image. Une bonne légende décrit ce qui
  VARIE d'une image à l'autre (couleur, matière, décor) ; le mot déclencheur
  "mrcdoor" absorbe alors ce qui est COMMUN (le style "porte marocaine").
  Si on ne décrit pas la couleur, le modèle risque de coller la couleur au concept.
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "data" / "captions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overwrite", action="store_true", help="Réécrit aussi les légendes déjà faites")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    trigger = cfg["trigger_word"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)

    with open(CAPTIONS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if "TODO" not in r["caption"] and not args.overwrite:
            continue
        img = Image.open(ROOT / "data" / "processed" / r["split"] / r["file_name"]).convert("RGB")
        inputs = processor(img, "a photograph of", return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=30)
        desc = processor.decode(out[0], skip_special_tokens=True)
        desc = desc.removeprefix("a photograph of").strip()
        r["caption"] = f"a photo of a {trigger}, {desc}"
        print(f"{r['file_name']}: {r['caption']}")

    with open(CAPTIONS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "split", "source", "caption"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nLégendes écrites dans {CAPTIONS}. RELIRE et corriger chaque ligne.")


if __name__ == "__main__":
    main()
