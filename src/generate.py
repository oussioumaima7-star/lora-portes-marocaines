"""Génération d'images avec ou sans LoRA, à intensité réglable.

Usage :
  python -m src.generate --lora results/loras/rank8/final --prompt "a mrcdoor in the snow" \
         --scale 0.8 --seed 0 --out results/samples

Le "scale" multiplie la contribution de la LoRA : W' = W + scale * (alpha/r) * B@A
  scale = 0   -> modèle de base, la LoRA est ignorée
  scale = 1   -> LoRA telle qu'entraînée
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_pipeline(base_model: str = BASE):
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16,
                                                   safety_checker=None)
    # Échantillonneur rapide : ~25-30 étapes suffisent (au lieu de 50 avec PNDM)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


class LoraSwitcher:
    """Charge plusieurs LoRA une seule fois et passe de l'une à l'autre sans recharger SD."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.loaded = {}

    def use(self, lora_path: str | None, scale: float):
        if lora_path is None or scale == 0:
            if self.loaded:
                self.pipe.disable_lora()
            return
        name = Path(lora_path).parent.name + "_" + Path(lora_path).name
        name = name.replace("-", "_").replace(".", "_")
        if name not in self.loaded:
            self.pipe.load_lora_weights(lora_path, adapter_name=name)
            self.loaded[name] = lora_path
        self.pipe.enable_lora()
        self.pipe.set_adapters([name], adapter_weights=[scale])


@torch.no_grad()
def generate(pipe, prompt, seed=0, steps=30, guidance=7.5, negative_prompt=None, n=1):
    g = torch.Generator("cuda").manual_seed(seed)
    t0 = time.time()
    images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                  guidance_scale=guidance, generator=g, num_images_per_prompt=n).images
    return images, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/samples")
    args = ap.parse_args()

    pipe = load_pipeline()
    sw = LoraSwitcher(pipe)
    sw.use(args.lora, args.scale)
    imgs, dt = generate(pipe, args.prompt, seed=args.seed, steps=args.steps, n=args.n,
                        negative_prompt="blurry, low quality, watermark, text")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in args.prompt)[:50]
    for i, im in enumerate(imgs):
        im.save(out / f"{slug}_s{args.scale}_seed{args.seed}_{i}.png")
    print(f"{len(imgs)} images en {dt:.1f}s -> {out}")


if __name__ == "__main__":
    main()
