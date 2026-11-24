"""Démo interactive : prompt + intensité de la LoRA, comparaison avant / après.

Lancement (Colab) : python demo/app.py --share
"""
import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generate import LoraSwitcher, generate, load_pipeline  # noqa: E402

NEG = "blurry, low quality, watermark, text, deformed"


def find_loras():
    found = sorted(str(p.parent) for p in (ROOT / "results" / "loras").glob("*/*/*.safetensors"))
    return {str(Path(p).relative_to(ROOT / "results" / "loras")): p for p in found}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Lien public (utile sur Colab)")
    args = ap.parse_args()

    loras = find_loras()
    if not loras:
        raise SystemExit("Aucune LoRA trouvée dans results/loras/ : entraîner d'abord.")
    pipe = load_pipeline()
    sw = LoraSwitcher(pipe)

    def run(prompt, lora_name, scale, seed, steps):
        sw.use(None, 0)
        before, t_b = generate(pipe, prompt, seed=int(seed), steps=int(steps), negative_prompt=NEG)
        sw.use(loras[lora_name], scale)
        after, t_a = generate(pipe, prompt, seed=int(seed), steps=int(steps), negative_prompt=NEG)
        info = f"Avant : {t_b:.1f}s · Après : {t_a:.1f}s · même seed ({int(seed)})"
        return before[0], after[0], info

    with gr.Blocks(title="LoRA portes marocaines") as demo:
        gr.Markdown("## LoRA texte-image : portes marocaines\n"
                    "Utilisez le mot **mrcdoor** dans le prompt pour activer le concept appris.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(value="a photo of a mrcdoor with blue zellige, sunny day",
                                    label="Prompt")
                lora = gr.Dropdown(list(loras), value=list(loras)[-1], label="LoRA")
                scale = gr.Slider(0, 1.5, value=1.0, step=0.1, label="Intensité de la LoRA")
                seed = gr.Number(value=0, precision=0, label="Seed")
                steps = gr.Slider(10, 50, value=30, step=1, label="Étapes de débruitage")
                btn = gr.Button("Générer", variant="primary")
            with gr.Column():
                with gr.Row():
                    out_b = gr.Image(label="Modèle de base")
                    out_a = gr.Image(label="Avec LoRA")
                info = gr.Markdown()
        btn.click(run, [prompt, lora, scale, seed, steps], [out_b, out_a, info])
        gr.Examples([["a mrcdoor in the snow"], ["a watercolor painting of a mrcdoor"],
                     ["a mrcdoor on a modern glass building"]], [prompt])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
