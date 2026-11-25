"""Évaluation chiffrée des LoRA : respect du texte, fidélité au domaine,
surapprentissage, diversité, fuite de style et coût.

Usage :
  python -m src.evaluate --loras results/loras/rank4/final results/loras/rank8/final ...
  python -m src.evaluate --loras results/loras/rank8_steps2000/checkpoint-* --quick

Toutes les mesures utilisent les embeddings CLIP (ViT-B/32), similarité cosinus :
  clip_score      : image <-> prompt (x100). Respect du texte. Le mot "mrcdoor" est
                    remplacé par "moroccan door" car CLIP ne connaît pas notre mot inventé.
  domain_sim      : image générée <-> images RÉELLES heldout. Proximité au domaine.
  train_nn_sim    : image générée <-> image d'entraînement la plus proche.
  copy_rate       : % d'images avec train_nn_sim > 0.95 -> quasi-copies = surapprentissage.
  diversity       : 1 - similarité moyenne entre images d'un même prompt (seeds différentes).
  leakage_sim     : prompts SANS mot déclencheur, image avec LoRA <-> image du modèle de base
                    (même seed). Proche de 1 = la LoRA ne "contamine" pas le reste du modèle.
  sec_per_image   : coût de génération.

Pourquoi pas la FID ? Elle demande des milliers d'images réelles ; avec 5-10 images
heldout elle n'a aucune valeur statistique. On le justifie dans le rapport.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import load_config
from src.generate import LoraSwitcher, generate, load_pipeline

ROOT = Path(__file__).resolve().parents[1]
CLIP_ID = "openai/clip-vit-base-patch32"


class Clip:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_ID).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def images(self, imgs):
        out = []
        for i in range(0, len(imgs), 16):
            x = self.proc(images=imgs[i:i + 16], return_tensors="pt").to("cuda")
            # Calcul explicite (vision -> projection) : marche avec toutes les versions de
            # transformers (la v5 renvoie un objet au lieu d'un tenseur via get_image_features)
            pooled = self.model.vision_model(pixel_values=x["pixel_values"]).pooler_output
            out.append(F.normalize(self.model.visual_projection(pooled), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def texts(self, texts):
        x = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        pooled = self.model.text_model(input_ids=x["input_ids"],
                                       attention_mask=x.get("attention_mask")).pooler_output
        return F.normalize(self.model.text_projection(pooled), dim=-1)


def load_dir(d: Path):
    return [Image.open(p).convert("RGB") for p in sorted(d.glob("*.png"))]


def lora_label(path: str | None) -> str:
    if path is None:
        return "base"
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def make_grid(images, rows, cols, size=256):
    grid = Image.new("RGB", (cols * size, rows * size), "white")
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        grid.paste(im.resize((size, size)), (c * size, r * size))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loras", nargs="+", required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--quick", action="store_true", help="Échelle 1.0 et 2 seeds seulement")
    args = ap.parse_args()

    ecfg = load_config(ROOT / args.config)
    bcfg = load_config(ROOT / args.base_config)
    trigger = bcfg["trigger_word"]
    scales = [1.0] if args.quick else [s for s in ecfg["lora_scales"] if s > 0]
    seeds = ecfg["seeds"][:2] if args.quick else ecfg["seeds"]
    groups = {g: ecfg[f"{g}_prompts"] for g in ("domain", "generalization", "leakage")}
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    clip = Clip()
    train_emb = clip.images(load_dir(ROOT / bcfg["train_dir"]))
    held_emb = clip.images(load_dir(ROOT / bcfg["heldout_dir"]))

    pipe = load_pipeline(bcfg["base_model"])
    sw = LoraSwitcher(pipe)

    def run(lora, scale):
        """Génère (ou relit si déjà présentes) les images d'une configuration."""
        d = out_root / lora_label(lora).replace("/", "__") / f"scale_{scale}"
        d.mkdir(parents=True, exist_ok=True)
        sw.use(lora, scale)
        res, times = {}, []
        for g, prompts in groups.items():
            for pi, prompt in enumerate(prompts):
                imgs = []
                for s in seeds:
                    f = d / f"{g}_{pi}_seed{s}.png"
                    if f.exists():
                        imgs.append(Image.open(f).convert("RGB"))
                        continue
                    im, dt = generate(pipe, prompt, seed=s, steps=ecfg["num_inference_steps"],
                                      guidance=ecfg["guidance_scale"],
                                      negative_prompt=ecfg["negative_prompt"])
                    im[0].save(f)
                    imgs.append(im[0])
                    times.append(dt)
                res[(g, pi)] = imgs
        return res, (sum(times) / len(times) if times else None)

    print("Génération de référence (modèle de base)...")
    base_res, base_time = run(None, 0.0)
    base_emb = {k: clip.images(v) for k, v in base_res.items()}

    configs = [(None, 0.0)] + [(l, s) for l in args.loras for s in scales]
    rows = []
    for lora, scale in configs:
        print(f"Évaluation {lora_label(lora)} @ scale {scale}")
        res, t = (base_res, base_time) if lora is None else run(lora, scale)
        for g in ("domain", "generalization"):
            cs, ds, nn, div = [], [], [], []
            for pi, prompt in enumerate(groups[g]):
                emb = clip.images(res[(g, pi)])
                txt = clip.texts([prompt.replace(trigger, "moroccan door")])
                cs.append((emb @ txt.T).mean().item() * 100)
                ds.append((emb @ held_emb.T).mean().item())
                nn.extend((emb @ train_emb.T).max(dim=1).values.tolist())
                sim = emb @ emb.T
                n = sim.shape[0]
                div.append(1 - (sim.sum() - n).item() / (n * (n - 1)))
            rows.append({
                "lora": lora_label(lora), "scale": scale, "group": g,
                "clip_score": round(sum(cs) / len(cs), 2),
                "domain_sim": round(sum(ds) / len(ds), 4),
                "train_nn_sim": round(sum(nn) / len(nn), 4),
                "copy_rate_%": round(100 * sum(x > 0.95 for x in nn) / len(nn), 1),
                "diversity": round(sum(div) / len(div), 4),
                "leakage_sim": None,
                "sec_per_image": round(t, 2) if t else None,
            })
        leak = [F.cosine_similarity(clip.images(res[("leakage", pi)]), base_emb[("leakage", pi)]
                                    ).mean().item() for pi in range(len(groups["leakage"]))]
        rows[-1]["leakage_sim"] = rows[-2]["leakage_sim"] = round(sum(leak) / len(leak), 4)

        # Grille visuelle pour le rapport : une ligne par prompt, une colonne par seed
        if lora is not None:
            ims = [im for k in itertools.chain(*[[(g, i) for i in range(len(p))]
                                                for g, p in groups.items()]) for im in res[k]]
            name = lora_label(lora).replace("/", "__")
            make_grid(ims, rows=len(ims) // len(seeds), cols=len(seeds)).save(
                out_root / f"grid_{name}_scale{scale}.jpg", quality=90)

    # Ajout des coûts d'entraînement (metrics_train.json écrit par train_lora.py)
    for r in rows:
        if r["lora"] == "base":
            continue
        mfile = ROOT / "results" / "loras" / r["lora"].split("/")[0] / "metrics_train.json"
        if mfile.exists():
            m = json.loads(mfile.read_text())
            r.update({k: m[k] for k in ("rank", "train_time_min", "peak_vram_gb", "lora_size_mb")})
            ckpt = r["lora"].split("/")[1]
            # pour un checkpoint intermédiaire, le nombre d'étapes est dans son nom
            r["steps"] = int(ckpt.split("-")[1]) if ckpt.startswith("checkpoint-") else m["steps"]

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_root / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nRésultats : {out_root / 'metrics.csv'}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
