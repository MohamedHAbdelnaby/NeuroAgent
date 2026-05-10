
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = REPO_ROOT / "eval" / "ground_truth" / "adhd_labels.csv"
FASTSURFER_ROOT = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "ADHD-200"
OUT_PATH = REPO_ROOT / "data" / "adhd200_structural_norms.json"

ICV_REF = 1_400_000.0

STRUCTURES = [
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-Thalamus", "Right-Thalamus",
    "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
]

def parse_stats(stats_path: Path) -> tuple[dict[str, float], float]:
    volumes: dict[str, float] = {}
    icv = 0.0
    if not stats_path.exists():
        return volumes, icv
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Measure"):
                parts = line.split(",")
                if len(parts) >= 4:
                    name = parts[0].split()[-1].strip()
                    if name in ("Mask", "MaskVol"):
                        try:
                            icv = float(parts[3].strip())
                        except ValueError:
                            pass
                continue
            if line.startswith("#") or not line:
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            try:
                volumes[cols[4]] = float(cols[3])
            except (ValueError, IndexError):
                pass
    return volumes, icv

def find_stats_file(subj_dir: Path) -> Path | None:
    for name in ("aseg+DKT.VINN.stats", "aseg.VINN.stats", "aseg.stats"):
        p = subj_dir / "stats" / name
        if p.exists():
            return p
    return None

def main() -> None:
    if not LABELS_CSV.exists():
        raise SystemExit(f"Missing labels CSV: {LABELS_CSV}")

    by_site: dict[str, dict[str, list[float]]] = {}
    global_acc: dict[str, list[float]] = {s: [] for s in STRUCTURES}

    n_total = n_skipped = 0
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if int(row["label"]) != 0:
                continue
            sid = row["subject_id"]
            site = row["site"]
            subj_dir = FASTSURFER_ROOT / sid
            stats_path = find_stats_file(subj_dir)
            if stats_path is None:
                n_skipped += 1
                continue
            volumes, icv = parse_stats(stats_path)
            if not volumes or icv <= 0:
                n_skipped += 1
                continue
            ratio = ICV_REF / icv
            site_acc = by_site.setdefault(site, {s: [] for s in STRUCTURES})
            for s in STRUCTURES:
                v = volumes.get(s, 0.0)
                if v > 0:
                    corrected = v * ratio
                    site_acc[s].append(corrected)
                    global_acc[s].append(corrected)

    norms: dict[str, dict[str, dict[str, float]]] = {}
    for site, acc in sorted(by_site.items()):
        norms[site] = {}
        for s, vals in acc.items():
            if len(vals) >= 5:
                norms[site][s] = {
                    "mean": round(statistics.mean(vals), 2),
                    "sd": round(statistics.stdev(vals), 2),
                    "n": len(vals),
                }
    norms["GLOBAL"] = {
        s: {
            "mean": round(statistics.mean(v), 2),
            "sd": round(statistics.stdev(v), 2),
            "n": len(v),
        }
        for s, v in global_acc.items()
        if len(v) >= 5
    }

    out = {
        "icv_ref_mm3": ICV_REF,
        "structures": STRUCTURES,
        "norms": norms,
        "n_subjects_seen": n_total,
        "n_subjects_skipped": n_skipped,
        "source": "ADHD-200 control subjects (label==0), FastSurfer aseg+DKT.VINN.stats, ICV-corrected to ICV_REF",
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    for site, by_s in norms.items():
        n = next(iter(by_s.values()))["n"] if by_s else 0
        print(f"  {site:12s} n={n:3d}  Left-Caudate={by_s.get('Left-Caudate', {})}")

if __name__ == "__main__":
    main()
