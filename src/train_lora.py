"""Entraînement d'une LoRA sur le U-Net de Stable Diffusion 1.5 (< 12 Go de VRAM).

Usage :
  python -m src.train_lora --config configs/rank8.yaml

Principe (à savoir expliquer en soutenance) :
  1. On gèle TOUT le modèle (VAE, encodeur de texte, U-Net).
  2. Dans les couches d'attention du U-Net, on ajoute deux petites matrices A et B :
         W' = W + (alpha / r) * B @ A      avec A : r x d,  B : d x r,  r << d
     Seules A et B sont entraînées -> quelques Mo au lieu de ~3,4 Go.
  3. Objectif de diffusion classique : on bruite l'image latente à un instant t
     aléatoire et le U-Net doit prédire le bruit ajouté (perte MSE).

Astuce mémoire : comme le VAE et l'encodeur de texte sont gelés, on calcule UNE fois
les latents des images et les embeddings des légendes, puis on les retire du GPU.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(cfg):
    """Retourne la liste (chemin image, légende) du split train."""
    rows = []
    with open(ROOT / "data" / "captions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            if "TODO" in r["caption"]:
                raise ValueError(f"Légende non remplie pour {r['file_name']} (data/captions.csv)")
            rows.append((ROOT / cfg["train_dir"] / r["file_name"], r["caption"]))
    if not rows:
        raise ValueError("Aucune image d'entraînement : lancer data/prepare.py d'abord.")
    return rows


@torch.no_grad()
def precompute(cfg, rows, device):
    """Encode une seule fois images -> latents et légendes -> embeddings."""
    base = cfg["base_model"]
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder",
                                                 torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(device)

    res = cfg["resolution"]
    tf = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(res) if cfg["center_crop"] else transforms.RandomCrop(res),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # pixels dans [-1, 1], comme attendu par le VAE
    ])

    samples = []
    for path, caption in rows:
        pixels = tf(Image.open(path).convert("RGB")).unsqueeze(0)
        variants = [pixels, torch.flip(pixels, dims=[3])] if cfg["random_flip"] else [pixels]
        dists = []
        for v in variants:
            d = vae.encode(v.to(device, torch.float16)).latent_dist
            # on garde moyenne et écart-type : on tirera un échantillon différent à chaque pas
            dists.append((d.mean.cpu(), d.std.cpu()))
        ids = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        emb = text_encoder(ids)[0].cpu()
        samples.append({"dists": dists, "emb": emb})

    scaling = vae.config.scaling_factor
    del vae, text_encoder
    torch.cuda.empty_cache()
    return samples, scaling


def save_lora(unet, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=str(out_dir),
                                              unet_lora_layers=state, safe_serialization=True)
    return sum(p.stat().st_size for p in out_dir.glob("*.safetensors")) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)

    if not torch.cuda.is_available():
        raise SystemExit("GPU requis (Colab : Exécution > Modifier le type d'exécution > T4 GPU).")
    device = torch.device("cuda")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    out_dir = ROOT / cfg["output_dir"] / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    rows = load_dataset(cfg)
    print(f"{len(rows)} images d'entraînement")
    samples, scaling = precompute(cfg, rows, device)

    # --- U-Net gelé en fp16 + LoRA entraînable ---
    unet = UNet2DConditionModel.from_pretrained(cfg["base_model"], subfolder="unet",
                                                torch_dtype=torch.float16).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=cfg["lora_dropout"], init_lora_weights="gaussian",
                                target_modules=cfg["target_modules"]))
    cast_training_params(unet, dtype=torch.float32)  # seuls les poids LoRA passent en fp32
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet.train()

    params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Paramètres entraînables : {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f} %)")

    if cfg["use_8bit_adam"]:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=cfg["learning_rate"], weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=1e-2)
    lr_sched = get_scheduler(cfg["lr_scheduler"], optimizer=optimizer,
                             num_warmup_steps=cfg["lr_warmup_steps"],
                             num_training_steps=cfg["max_train_steps"])
    noise_sched = DDPMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")
    scaler = torch.amp.GradScaler("cuda")

    # Itérateur "par époques" : chaque image est vue autant de fois que les autres
    def batches():
        while True:
            order = list(range(len(samples)))
            random.shuffle(order)
            for i in range(0, len(order), cfg["train_batch_size"]):
                yield order[i:i + cfg["train_batch_size"]]

    accum = cfg["gradient_accumulation_steps"]
    log_path = out_dir / "train_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.writer(log_f)
    logger.writerow(["step", "loss", "lr"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    it = batches()
    step, running, ema = 0, 0.0, None
    epochs = cfg["max_train_steps"] * accum * cfg["train_batch_size"] / len(samples)
    print(f"{cfg['max_train_steps']} étapes ≈ {epochs:.1f} passages sur le jeu")

    while step < cfg["max_train_steps"]:
        for _ in range(accum):
            idx = next(it)
            lat, emb = [], []
            for i in idx:
                mean, std = random.choice(samples[i]["dists"])  # flip aléatoire
                lat.append((mean + std * torch.randn_like(mean)) * scaling)
                emb.append(samples[i]["emb"])
            latents = torch.cat(lat).to(device, torch.float16)
            enc = torch.cat(emb).to(device, torch.float16)

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],),
                              device=device).long()
            noisy = noise_sched.add_noise(latents, noise, t)
            target = (noise if noise_sched.config.prediction_type == "epsilon"
                      else noise_sched.get_velocity(latents, noise, t))

            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy, t, enc).sample
            loss = F.mse_loss(pred.float(), target.float()) / accum
            scaler.scale(loss).backward()
            running += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_sched.step()
        step += 1

        ema = running if ema is None else 0.98 * ema + 0.02 * running
        logger.writerow([step, f"{running:.5f}", f"{lr_sched.get_last_lr()[0]:.2e}"])
        running = 0.0
        if step % 25 == 0:
            el = time.time() - t0
            print(f"step {step}/{cfg['max_train_steps']}  loss(ema)={ema:.4f}  "
                  f"{el / step:.2f}s/step  VRAM max={torch.cuda.max_memory_allocated() / 1e9:.2f} Go")
        if step % cfg["checkpoint_every"] == 0 and step < cfg["max_train_steps"]:
            save_lora(unet, out_dir / f"checkpoint-{step}")
    log_f.close()

    train_time = time.time() - t0
    size_mb = save_lora(unet, out_dir / "final")
    metrics = {
        "experiment": cfg["experiment_name"],
        "rank": cfg["lora_rank"],
        "steps": cfg["max_train_steps"],
        "n_train_images": len(samples),
        "trainable_params": n_trainable,
        "trainable_percent": round(100 * n_trainable / n_total, 3),
        "train_time_min": round(train_time / 60, 2),
        "sec_per_step": round(train_time / cfg["max_train_steps"], 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "lora_size_mb": round(size_mb, 2),
        "final_loss_ema": round(ema, 5) if ema is not None and not math.isnan(ema) else None,
    }
    (out_dir / "metrics_train.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
