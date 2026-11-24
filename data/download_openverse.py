"""Télécharge des images sous licence libre depuis Openverse (source citée par le sujet).

Openverse indexe Flickr, Wikimedia et d'autres sources ; on IGNORE ici Wikimedia
(déjà exploité, et ses limites de requêtes nous bloquent). Les licences et auteurs
sont écrits dans data/sources.csv, comme pour download_commons.py.

Usage :
  python data/download_openverse.py --max 60
"""
import argparse
import csv
import time

from download_commons import FIELDS, HEADERS, RAW_DIR, REJECTED, SOURCES_CSV, fetch

API = "https://api.openverse.org/v1/images/"
QUERIES = ["moroccan door", "morocco door", "medina door morocco", "fes door",
           "chefchaouen door", "marrakech door", "rabat door", "tetouan door", "essaouira door"]


def license_label(lic: str, version: str) -> str:
    lic = (lic or "").lower()
    if lic == "cc0":
        return "CC0 1.0"
    if lic == "pdm":
        return "Public domain"
    return f"CC {lic.upper()} {version or ''}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=60, help="Nombre max de NOUVELLES images")
    ap.add_argument("--pages", type=int, default=3, help="Pages de 20 résultats par requête")
    ap.add_argument("--min-size", type=int, default=600)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    known = set()
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                known |= {r["source_url"], r["title"]}
    if REJECTED.exists():
        known |= set(REJECTED.read_text(encoding="utf-8").splitlines())
    idx = 1 + max([int(p.stem.split("_")[1]) for p in RAW_DIR.glob("openverse_*.jpg")], default=-1)

    added, skipped = 0, {"wikimedia": 0, "taille": 0, "déjà vu": 0, "erreur": 0}
    for q in QUERIES:
        for page in range(1, args.pages + 1):
            if added >= args.max:
                break
            params = {"q": q, "license": "by,by-sa,cc0,pdm", "page_size": 20, "page": page,
                      "extension": "jpg"}
            try:
                data = fetch(API, params=params).json()
            except Exception as e:
                print(f"  ! recherche '{q}' p{page} : {e}")
                break
            results = data.get("results", [])
            for r in results:
                if added >= args.max:
                    break
                if "wikimedia" in (r.get("source") or r.get("provider") or "") or "wikimedia" in (r.get("url") or ""):
                    skipped["wikimedia"] += 1
                    continue
                landing = r.get("foreign_landing_url") or r.get("url")
                if landing in known or r.get("id") in known:
                    skipped["déjà vu"] += 1
                    continue
                w, h = r.get("width") or 0, r.get("height") or 0
                if w and h and min(w, h) < args.min_size:
                    skipped["taille"] += 1
                    continue
                name = f"openverse_{idx:04d}.jpg"
                try:
                    img = fetch(r["url"], tries=3)
                except Exception as e:
                    print(f"  ! ignorée ({e})")
                    skipped["erreur"] += 1
                    continue
                (RAW_DIR / name).write_bytes(img.content)
                row = {"file_name": name, "title": (r.get("title") or r.get("id"))[:200],
                       "author": (r.get("creator") or "inconnu")[:200],
                       "license": license_label(r.get("license"), r.get("license_version")),
                       "license_url": r.get("license_url") or "", "source_url": landing}
                write_header = not SOURCES_CSV.exists()
                with open(SOURCES_CSV, "a", newline="", encoding="utf-8") as f:
                    w_ = csv.DictWriter(f, fieldnames=FIELDS)
                    if write_header:
                        w_.writeheader()
                    w_.writerow(row)
                known.add(landing)
                print(f"  + {name}  [{row['license']}]  {row['title'][:60]}  ({r.get('source')})")
                idx += 1
                added += 1
                time.sleep(args.delay)
            if len(results) < 20:
                break
    print(f"\n{added} images ajoutées. Ignorées : {skipped}")


if __name__ == "__main__":
    main()
