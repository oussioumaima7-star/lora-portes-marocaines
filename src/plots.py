"""Figures pour le rapport à partir de results/eval/metrics.csv et des train_log.csv.

Usage : python -m src.plots
Sorties : results/figures/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def loss_curves():
    plt.figure(figsize=(7, 4))
    for log in sorted((ROOT / "results" / "loras").glob("*/train_log.csv")):
        df = pd.read_csv(log)
        plt.plot(df["step"], df["loss"].ewm(alpha=0.05).mean(), label=log.parent.name)
    plt.xlabel("étape")
    plt.ylabel("perte MSE (lissée)")
    plt.title("Courbes d'entraînement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "loss_curves.png", dpi=150)
    plt.close()


def by_rank(df):
    d = df[df["lora"].str.startswith("rank") & df["lora"].str.endswith("/final")
           & ~df["lora"].str.contains("steps") & (df["scale"] == 1.0)].copy()
    if d.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col, title in zip(axes, ["clip_score", "domain_sim", "copy_rate_%", "lora_size_mb"],
                              ["Respect du texte (CLIPScore)", "Proximité au domaine",
                               "Quasi-copies (%)", "Taille LoRA (Mo)"]):
        for g, sub in d.groupby("group"):
            if col == "lora_size_mb" and g != "domain":
                continue
            sub = sub.sort_values("rank")
            ax.plot(sub["rank"], sub[col], marker="o", label=g)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rang r")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "comparaison_rangs.png", dpi=150)
    plt.close()


def by_steps(df):
    d = df[df["lora"].str.contains("steps") & (df["scale"] == 1.0)
           & (df["group"] == "generalization")].copy()
    if d.empty or "steps" not in d:
        return
    d = d.sort_values("steps")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in zip(axes, ["domain_sim", "diversity", "train_nn_sim"],
                              ["Proximité au domaine ↑", "Diversité ↑",
                               "Similarité au plus proche voisin d'entraînement"]):
        ax.plot(d["steps"], d[col], marker="o")
        ax.set_xlabel("étapes")
        ax.set_title(title, fontsize=10)
    plt.suptitle("Surapprentissage : prompts de généralisation, rang 8")
    plt.tight_layout()
    plt.savefig(FIG / "surapprentissage_etapes.png", dpi=150)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    loss_curves()
    m = ROOT / "results" / "eval" / "metrics.csv"
    if m.exists():
        df = pd.read_csv(m)
        by_rank(df)
        by_steps(df)
    print(f"Figures dans {FIG}")


if __name__ == "__main__":
    main()
