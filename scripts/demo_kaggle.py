"""Lance la démo Gradio sur Kaggle avec les LoRA déjà entraînées (aucun réentraînement).

Prérequis : la sortie du notebook d'entraînement (version réussie) ajoutée en Input.
Lancé par le notebook :  python scripts/demo_kaggle.py
Le lien public "https://....gradio.live" s'affiche après ~2 min : l'ouvrir dans le navigateur.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("/kaggle/input")
DST = ROOT / "results" / "loras"


def restore():
    # 1) dossier "loras" dans la sortie du notebook d'entraînement
    for d in INPUT.rglob("loras"):
        if d.is_dir() and any(d.glob("*/final/*.safetensors")):
            for f in d.glob("*/final"):
                shutil.copytree(f, DST / f.parent.name / "final", dirs_exist_ok=True)
            return f"dossier {d}"
    # 2) sinon, resultats.zip
    for z in INPUT.rglob("resultats.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.startswith("results/loras/") and "/final/" in n and not n.endswith("/"):
                    zf.extract(n, ROOT)
        if any(DST.glob("*/final/*.safetensors")):
            return f"archive {z}"
    sys.exit("LoRA introuvables : ajoute en Input la sortie du notebook d'entraînement "
             "(+ Add Input > Your Work > Notebooks).")


if __name__ == "__main__":
    src = restore()
    print(f"LoRA reprises depuis {src} : {sorted(p.name for p in DST.iterdir())}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, str(ROOT / "demo" / "app.py"), "--share"], check=True)
