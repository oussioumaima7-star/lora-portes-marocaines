"""Tout le pipeline Kaggle en un seul script : données -> 5 entraînements -> évaluation -> zip.

Lancé par le notebook Kaggle :  python scripts/run_kaggle.py
Robuste : si un entraînement échoue, on continue avec les autres, et resultats.zip
est TOUJOURS créé à la fin (même partiel) pour pouvoir analyser ce qui s'est passé.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT_ZIP = WORK / "resultats.zip"
LOG = ROOT / "results" / "run_log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, check=True):
    log("$ " + " ".join(args))
    r = subprocess.run([sys.executable] + args)
    if check and r.returncode != 0:
        raise RuntimeError(f"échec ({r.returncode}) : {' '.join(args)}")
    return r.returncode == 0


def find_data():
    """Trouve images + CSV n'importe où dans /kaggle/input (zip décompressé ou non)."""
    extra = WORK / "ds"
    for z in INPUT.rglob("*.zip"):
        if z.name == "resultats.zip":
            continue
        log(f"Décompression de {z}")
        zipfile.ZipFile(z).extractall(extra)
    roots = [INPUT, extra]
    raw = ROOT / "data" / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    n = 0
    for r in roots:
        for f in r.rglob("*"):
            # on ignore les images produites par un run précédent (figures, grilles...)
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and not EXCL & set(f.parts):
                shutil.copy(f, raw / f.name)
                n += 1
    for name in ["sources.csv", "captions.csv"]:
        hits = [h for r in roots for h in r.rglob(name)]
        if hits:
            shutil.copy(hits[0], ROOT / "data" / name)
        log(f"{name} : {hits[0] if hits else 'INTROUVABLE'}")
    n = len(list(raw.iterdir()))
    log(f"{n} images copiées dans data/raw")
    if n < 20:
        raise RuntimeError(f"Seulement {n} images : vérifier le dataset en Input")


EXCL = {"processed", "results", "figures", "eval", "eval_etapes"}


def restore_loras():
    """Reprend les LoRA d'un run précédent (sortie d'une version ajoutée en Input) :
    évite de refaire 2 h 40 d'entraînement si seule l'évaluation a échoué."""
    dst = ROOT / "results" / "loras"
    for d in INPUT.rglob("loras"):
        if d.is_dir() and any(d.glob("*/final/*.safetensors")):
            shutil.copytree(d, dst, dirs_exist_ok=True)
            log(f"LoRA reprises depuis {d} : {sorted(p.name for p in dst.iterdir())}")
            return
    log("Aucune LoRA précédente trouvée : entraînement complet")


def make_zip():
    files = []
    for pattern in ["results/run_log.txt", "results/resume.md", "results/figures/*",
                    "results/eval/*.csv", "results/eval/*.jpg", "results/eval_etapes/*.csv",
                    "results/eval_etapes/*.jpg", "results/loras/*/*.json", "results/loras/*/*.csv",
                    "results/loras/*/final/*", "data/captions.csv", "data/prepare_report.txt"]:
        files += glob.glob(pattern)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f)
    log(f"{OUT_ZIP} : {len(files)} fichiers, {OUT_ZIP.stat().st_size / 1e6:.1f} Mo")


def main():
    t0 = time.time()
    try:
        # Kaggle préinstalle torchao 0.10, que les versions récentes de peft refusent
        # (« only versions above 0.16.0 are supported »). On ne s'en sert pas : on le retire.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
        r = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available as f; print(f())"],
                           capture_output=True, text=True)
        log(f"torchao retiré, peft le voit : {r.stdout.strip() or r.stderr.strip()[-200:]}")
        import torch
        log(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUN'}")
        if not torch.cuda.is_available():
            raise RuntimeError("Pas de GPU : Settings > Accelerator > GPU T4 x2")

        find_data()
        restore_loras()
        run(["data/prepare.py", "--heldout", "8", "--mode", "crop"])
        import pandas as pd
        cap = pd.read_csv("data/captions.csv")
        todo = cap[cap.caption.str.contains("TODO")]
        if not todo.empty:
            raise RuntimeError(f"{len(todo)} légendes contiennent TODO")
        log(f"{len(cap)} légendes OK")

        runs = {"rank8": "rank8", "rank4": "rank4", "rank16": "rank16", "rank32": "rank32",
                "steps_long": "rank8_steps2000"}
        for cfg, name in runs.items():
            if (ROOT / "results" / "loras" / name / "final").exists():
                log(f"{cfg} déjà fait")
                continue
            log(f"===== Entraînement {cfg}")
            if not run(["-m", "src.train_lora", "--config", f"configs/{cfg}.yaml"], check=False):
                log(f"!! {cfg} a échoué, on continue")

        finals = [f"results/loras/{r}/final" for r in ["rank4", "rank8", "rank16", "rank32"]
                  if Path(f"results/loras/{r}/final").exists()]
        if finals:
            log("===== Évaluation des rangs")
            run(["-m", "src.evaluate", "--loras", *finals], check=False)
            if Path("results/eval/metrics.csv").exists():
                shutil.copy("results/eval/metrics.csv", "results/eval/metrics_rangs.csv")
        ckpts = sorted(glob.glob("results/loras/rank8_steps2000/checkpoint-*")) + \
            glob.glob("results/loras/rank8_steps2000/final")
        if ckpts:
            log("===== Évaluation au fil des étapes")
            run(["-m", "src.evaluate", "--quick", "--out", "results/eval_etapes", "--loras", *ckpts],
                check=False)
        if Path("results/eval/metrics_rangs.csv").exists():
            a = pd.read_csv("results/eval/metrics_rangs.csv")
            if Path("results/eval_etapes/metrics.csv").exists():
                b = pd.read_csv("results/eval_etapes/metrics.csv")
                a = pd.concat([a, b[b.lora != "base"]])
            a.to_csv("results/eval/metrics.csv", index=False)
        run(["-m", "src.plots"], check=False)
        run(["-m", "src.summary"], check=False)
    except Exception as e:
        log(f"ERREUR : {e!r}")
    finally:
        log(f"Durée totale : {(time.time() - t0) / 60:.0f} min")
        make_zip()
        # libère l'espace : les images générées une par une ne sont pas utiles dans Output
        for d in glob.glob("results/eval/*/") + glob.glob("results/eval_etapes/*/"):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
