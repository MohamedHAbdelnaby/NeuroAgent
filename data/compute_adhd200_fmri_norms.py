
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import statistics
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = REPO_ROOT / "eval" / "ground_truth" / "adhd_labels.csv"
FMRIPREP_ROOT = REPO_ROOT / "preprocessing" / "fmriprep_output"
OUT_PATH = REPO_ROOT / "data" / "adhd200_fmri_norms.json"

SEEDS = {
    "PCC":       (0.0,  -52.0,  26.0),
    "mPFC":      (0.0,   54.0,  -6.0),
    "Precuneus": (0.0,  -56.0,  38.0),
    "L.ANG":     (-46.0, -66.0, 32.0),
    "R.ANG":     (46.0,  -66.0, 32.0),
    "ACC":       (0.0,   28.0,  20.0),
    "R.INS":     (36.0,   0.0,   4.0),
    "L.INS":     (-36.0,  0.0,   4.0),
    "L.DLPFC":   (-44.0, 36.0,  28.0),
    "R.DLPFC":   (44.0,  36.0,  28.0),
    "L.CAU":     (-14.0,  8.0,  14.0),
    "R.CAU":     (14.0,   8.0,  14.0),
    "R.IFG":     (52.0,  18.0,  10.0),
}

PAIRS = [
    ("PCC",    "mPFC",    "PCC_mPFC"),
    ("PCC",    "Precuneus","PCC_Precuneus"),
    ("ACC",    "R.INS",   "ACC_R.INS"),
    ("L.DLPFC","R.DLPFC", "L.DLPFC_R.DLPFC"),
    ("L.DLPFC","L.CAU",   "L.DLPFC_L.CAU"),
    ("R.DLPFC","R.CAU",   "R.DLPFC_R.CAU"),
    ("R.IFG",  "R.CAU",   "R.IFG_R.CAU"),
]

CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    "white_matter", "csf",
]

def load_confounds(tsv: Path) -> np.ndarray | None:
    try:
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t")
        cols = [c for c in CONFOUND_COLS if c in df.columns]
        if not cols:
            return None
        return df[cols].fillna(0).values.astype(float)
    except Exception:
        return None

def find_bold_and_confounds(site: str, sid: str) -> tuple[Path | None, Path | None]:
    sub = sid.split("_", 1)[1] if "_" in sid else sid
    sub_dir = FMRIPREP_ROOT / site / sub
    if not sub_dir.exists():
        return None, None
    bolds = sorted(sub_dir.rglob(
        "*MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    ))
    if not bolds:
        return None, None
    bold = bolds[0]
    confs = sorted(sub_dir.rglob("*confounds_timeseries.tsv"))
    return bold, (confs[0] if confs else None)

def extract_seed_ts(bold: Path, confs: Path | None) -> dict[str, np.ndarray] | None:
    try:
        from nilearn.maskers import NiftiSpheresMasker
        seeds = list(SEEDS.values())
        names = list(SEEDS.keys())
        t_r = 2.0
        sidecar = bold.with_suffix("").with_suffix(".json")
        if sidecar.exists():
            try:
                t_r = float(json.loads(sidecar.read_text()).get("RepetitionTime", t_r))
            except Exception:
                pass
        masker = NiftiSpheresMasker(
            seeds=seeds, radius=8, allow_overlap=True,
            detrend=True, standardize=True,
            low_pass=0.1, high_pass=0.01, t_r=t_r,
            smoothing_fwhm=6.0, verbose=0,
        )
        confounds = load_confounds(confs) if confs else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = masker.fit_transform(str(bold), confounds=confounds) if confounds is not None \
                 else masker.fit_transform(str(bold))
        return {n: ts[:, i] for i, n in enumerate(names)}
    except Exception:
        return None

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def process_subject(args: tuple[str, str]) -> tuple[str, dict[str, float]] | None:
    site, sid = args
    bold, confs = find_bold_and_confounds(site, sid)
    if bold is None:
        return None
    ts = extract_seed_ts(bold, confs)
    if ts is None:
        return None
    rs: dict[str, float] = {}
    for a, b, key in PAIRS:
        if a in ts and b in ts:
            rs[key] = pearson_r(ts[a], ts[b])
    return site, rs

def main() -> None:
    work: list[tuple[str, str]] = []
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["label"]) != 0:
                continue
            work.append((row["site"], row["subject_id"]))
    print(f"Processing {len(work)} control subjects...")

    import os, signal
    env_workers = os.environ.get("N_WORKERS")
    if env_workers:
        n_workers = max(1, int(env_workers))
    else:
        n_workers = max(1, min(mp.cpu_count() // 2, 8))
    print(f"Using {n_workers} workers", flush=True)

    per_subject_timeout = int(os.environ.get("PER_SUBJECT_TIMEOUT", "120"))

    by_site: dict[str, dict[str, list[float]]] = {}
    n_ok = 0
    n_timeout = 0
    pool = mp.Pool(n_workers)
    try:
        async_results = [(args, pool.apply_async(process_subject, (args,))) for args in work]
        for i, (args, ar) in enumerate(async_results):
            try:
                result = ar.get(timeout=per_subject_timeout)
            except mp.TimeoutError:
                n_timeout += 1
                if (i + 1) % 25 == 0 or i + 1 == len(work):
                    print(f"  {i+1}/{len(work)} processed, {n_ok} OK, {n_timeout} timeout", flush=True)
                continue
            except Exception as e:
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(work)} processed, {n_ok} OK [error on {args[1]}: {type(e).__name__}]", flush=True)
                continue
            if (i + 1) % 25 == 0 or i + 1 == len(work):
                print(f"  {i+1}/{len(work)} processed, {n_ok} OK, {n_timeout} timeout", flush=True)
            if result is None:
                continue
            site, rs = result
            n_ok += 1
            site_acc = by_site.setdefault(site, {key: [] for _, _, key in PAIRS})
            for key, v in rs.items():
                site_acc[key].append(v)
    finally:
        pool.terminate()
        pool.join()

    print(f"\n{n_ok}/{len(work)} control subjects produced fMRI norms ({n_timeout} timeouts)")

    norms: dict[str, dict[str, dict[str, float]]] = {}
    global_acc: dict[str, list[float]] = {key: [] for _, _, key in PAIRS}
    for site, acc in sorted(by_site.items()):
        norms[site] = {}
        for key, vals in acc.items():
            if len(vals) >= 5:
                norms[site][key] = {
                    "mean": round(statistics.mean(vals), 4),
                    "sd":   round(statistics.stdev(vals), 4),
                    "n":    len(vals),
                }
            global_acc[key].extend(vals)
    norms["GLOBAL"] = {
        key: {
            "mean": round(statistics.mean(v), 4),
            "sd":   round(statistics.stdev(v), 4),
            "n":    len(v),
        }
        for key, v in global_acc.items() if len(v) >= 5
    }

    out = {
        "seeds_mni": SEEDS,
        "pairs": [p[2] for p in PAIRS],
        "norms": norms,
        "n_subjects_attempted": len(work),
        "n_subjects_ok": n_ok,
        "source": "ADHD-200 control subjects (label==0), fMRIPrep MNI res-2 preproc_bold, "
                  "8mm sphere seed extraction with 24-motion + WM/CSF + bandpass 0.01-0.1Hz",
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    for site, by_p in norms.items():
        if "PCC_mPFC" in by_p:
            d = by_p["PCC_mPFC"]
            print(f"  {site:12s} PCC_mPFC mean={d['mean']:+.3f} sd={d['sd']:.3f} n={d['n']}")

if __name__ == "__main__":
    main()
