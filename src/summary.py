"""Résumé des résultats en Markdown (pour le README et le rapport).

Usage : python -m src.summary   ->  results/resume.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    out = ["# Résumé des résultats\n"]

    trains = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / "loras").glob("*/metrics_train.json"))]
    if trains:
        t = pd.DataFrame(trains)[["experiment", "rank", "steps", "n_train_images", "trainable_params",
                                  "trainable_percent", "train_time_min", "peak_vram_gb", "lora_size_mb"]]
        out += ["## Coût d'entraînement\n", md(t), ""]
        out.append(f"Temps GPU total : {t.train_time_min.sum():.0f} min. "
                   f"VRAM max : {t.peak_vram_gb.max()} Go (limite du sujet : 12 Go).\n")

    m = ROOT / "results" / "eval" / "metrics.csv"
    if m.exists():
        df = pd.read_csv(m)
        cols = ["lora", "scale", "clip_score", "domain_sim", "train_nn_sim", "copy_rate_%",
                "diversity", "leakage_sim", "sec_per_image"]
        for g, title in [("domain", "Prompts du domaine"), ("generalization", "Prompts de généralisation")]:
            sub = df[df.group == g][[c for c in cols if c in df]].sort_values(["lora", "scale"])
            out += [f"## {title}\n", md(sub), ""]

        gen = df[(df.group == "generalization") & (df.lora != "base")]
        if not gen.empty:
            # Critère simple : proximité au domaine élevée SANS recopier les images d'entraînement
            gen = gen.assign(score=gen.domain_sim - gen["copy_rate_%"] / 100)
            best = gen.sort_values("score", ascending=False).iloc[0]
            out += ["## Meilleure configuration (critère : domain_sim − taux de copie)\n",
                    f"`{best.lora}` à l'échelle {best.scale} : domain_sim = {best.domain_sim}, "
                    f"copies = {best['copy_rate_%']} %, CLIPScore = {best.clip_score}.\n"]

    path = ROOT / "results" / "resume.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Écrit : {path}")


if __name__ == "__main__":
    main()
