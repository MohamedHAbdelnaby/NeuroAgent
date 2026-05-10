
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LABELS_CSV = REPO_ROOT / "eval" / "ground_truth" / "adhd_labels.csv"
FASTSURFER = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "ADHD-200"
FMRIPREP   = REPO_ROOT / "preprocessing" / "fmriprep_output"
OUT_PATH   = REPO_ROOT / "data" / "adhd200_loso_logreg.json"

ICV_REF = 1_400_000.0

ADHD_STRUCT_FEATS = [
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-Thalamus", "Right-Thalamus",
    "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
]

ADHD_FMRI_FEATS = [
    "PCC_mPFC", "PCC_Precuneus", "ACC_R.INS",
    "L.DLPFC_R.DLPFC", "L.DLPFC_L.CAU", "R.DLPFC_R.CAU", "R.IFG_R.CAU",
]

SEEDS = {
    "PCC": (0.0, -52.0, 26.0), "mPFC": (0.0, 54.0, -6.0),
    "Precuneus": (0.0, -56.0, 38.0),
    "ACC": (0.0, 28.0, 20.0), "R.INS": (36.0, 0.0, 4.0),
    "L.DLPFC": (-44.0, 36.0, 28.0), "R.DLPFC": (44.0, 36.0, 28.0),
    "L.CAU": (-14.0, 8.0, 14.0), "R.CAU": (14.0, 8.0, 14.0),
    "R.IFG": (52.0, 18.0, 10.0),
}
PAIRS = [("PCC","mPFC","PCC_mPFC"), ("PCC","Precuneus","PCC_Precuneus"),
          ("ACC","R.INS","ACC_R.INS"),
          ("L.DLPFC","R.DLPFC","L.DLPFC_R.DLPFC"),
          ("L.DLPFC","L.CAU","L.DLPFC_L.CAU"),
          ("R.DLPFC","R.CAU","R.DLPFC_R.CAU"),
          ("R.IFG","R.CAU","R.IFG_R.CAU")]
CONF_COLS = [
    "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
    "trans_x_derivative1","trans_y_derivative1","trans_z_derivative1",
    "rot_x_derivative1","rot_y_derivative1","rot_z_derivative1",
    "trans_x_power2","trans_y_power2","trans_z_power2",
    "rot_x_power2","rot_y_power2","rot_z_power2",
    "trans_x_derivative1_power2","trans_y_derivative1_power2","trans_z_derivative1_power2",
    "rot_x_derivative1_power2","rot_y_derivative1_power2","rot_z_derivative1_power2",
    "white_matter","csf",
]

def parse_stats(stats_path: Path) -> tuple[dict[str, float], float]:
    volumes: dict[str, float] = {}
    icv = 0.0
    if not stats_path.exists():
        return volumes, icv
    for line in stats_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("# Measure"):
            parts = line.split(",")
            if len(parts) >= 4 and parts[0].split()[-1].strip() in ("Mask", "MaskVol"):
                try: icv = float(parts[3].strip())
                except: pass
            continue
        if line.startswith("#") or not line: continue
        cols = line.split()
        if len(cols) < 5: continue
        try: volumes[cols[4]] = float(cols[3])
        except: pass
    return volumes, icv

def find_stats_file(subj_dir: Path) -> Path | None:
    for name in ("aseg+DKT.VINN.stats", "aseg.VINN.stats", "aseg.stats"):
        p = subj_dir / "stats" / name
        if p.exists(): return p
    return None

def extract_raw_features(args):
    sid, site, label, age, gender = args
    subj_dir = FASTSURFER / sid
    stats = find_stats_file(subj_dir)
    if stats is None:
        return sid, site, label, age, gender, None, None
    volumes, icv = parse_stats(stats)
    if icv <= 0:
        return sid, site, label, age, gender, None, None
    icv_ratio = ICV_REF / icv
    raw_struct = {f: volumes.get(f, 0.0) * icv_ratio for f in ADHD_STRUCT_FEATS
                   if volumes.get(f, 0.0) > 0}

    raw_fmri = {}
    sub = sid.split("_", 1)[1] if "_" in sid else sid
    sub_dir = FMRIPREP / site / sub
    if sub_dir.exists():
        bolds = sorted(sub_dir.rglob("*MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))
        if bolds:
            try:
                from nilearn.maskers import NiftiSpheresMasker
                import pandas as pd
                bold = bolds[0]
                tr = 2.0
                sidecar = bold.with_suffix("").with_suffix(".json")
                if sidecar.exists():
                    try: tr = float(json.loads(sidecar.read_text()).get("RepetitionTime", tr))
                    except: pass
                conf_arr = None
                confs = sorted(sub_dir.rglob("*confounds_timeseries.tsv"))
                if confs:
                    try:
                        df = pd.read_csv(confs[0], sep="\t")
                        cols = [c for c in CONF_COLS if c in df.columns]
                        if cols:
                            conf_arr = df[cols].fillna(0).values.astype(float)
                    except: pass
                masker = NiftiSpheresMasker(
                    seeds=list(SEEDS.values()), radius=8, allow_overlap=True,
                    detrend=True, standardize=True, low_pass=0.1, high_pass=0.01,
                    t_r=tr, smoothing_fwhm=6.0, verbose=0,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = masker.fit_transform(str(bold), confounds=conf_arr) if conf_arr is not None \
                         else masker.fit_transform(str(bold))
                seed_ts = {n: ts[:, i] for i, n in enumerate(SEEDS.keys())}
                for a, b, key in PAIRS:
                    if a in seed_ts and b in seed_ts:
                        ta, tb = seed_ts[a], seed_ts[b]
                        if len(ta) >= 10 and ta.std() > 1e-6 and tb.std() > 1e-6:
                            raw_fmri[key] = float(np.corrcoef(ta, tb)[0, 1])
            except Exception:
                pass

    return sid, site, label, age, gender, raw_struct, raw_fmri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    rows: list[tuple] = []
    with open(LABELS_CSV) as f:
        for r in csv.DictReader(f):
            try:
                age = float(r["age"]) if r.get("age") else None
                gender = int(round(float(r["gender"]))) if r.get("gender") else None
            except: age, gender = None, None
            rows.append((r["subject_id"], r["site"], int(r["label"]), age, gender))
    print(f"Extracting raw features for {len(rows)} subjects...", flush=True)

    import os
    n_workers = int(os.environ.get("N_WORKERS", args.workers))
    print(f"Using {n_workers} workers (per-subject 120s timeout)", flush=True)

    raw_data: list[dict] = []
    with mp.Pool(n_workers) as pool:
        futures = [(args, pool.apply_async(extract_raw_features, (args,))) for args in rows]
        for i, (args_, ar) in enumerate(futures):
            try:
                sid, site, label, age, gender, raw_struct, raw_fmri = ar.get(timeout=120)
            except mp.TimeoutError:
                continue
            except Exception:
                continue
            if raw_struct is None: continue
            raw_data.append({"sid": sid, "site": site, "label": label,
                              "age": age, "gender": gender,
                              "raw_struct": raw_struct, "raw_fmri": raw_fmri or {}})
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  {i+1}/{len(rows)} processed, {len(raw_data)} OK", flush=True)
        pool.terminate(); pool.join()

    print(f"\nFeature extraction done: {len(raw_data)} subjects with usable data")

    raw_data = [d for d in raw_data if d["age"] is not None and d["gender"] is not None]
    sites = sorted(set(d["site"] for d in raw_data))
    feature_names = (["age", "sex_male"]
                     + [f"struct__{s}" for s in ADHD_STRUCT_FEATS]
                     + [f"fmri__{p}" for p in ADHD_FMRI_FEATS])

    print(f"Sites: {sites}")
    print(f"Feature schema (z-scored inside LOSO): {len(feature_names)} features")
    print(f"Label distribution: {np.bincount([d['label'] for d in raw_data])}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score

    folds: dict[str, dict] = {}
    oof_pred = np.full(len(raw_data), -1, dtype=int)
    oof_proba = np.full(len(raw_data), 0.5)
    sids = [d["sid"] for d in raw_data]

    for held_out_site in sites:
        train_idx = [i for i, d in enumerate(raw_data) if d["site"] != held_out_site]
        test_idx  = [i for i, d in enumerate(raw_data) if d["site"] == held_out_site]
        if len(train_idx) < 10 or len(test_idx) < 5:
            print(f"  {held_out_site}: too few subjects, skipping"); continue

        norms_per_site_struct: dict[str, dict[str, tuple[float, float]]] = {}
        norms_per_site_fmri:   dict[str, dict[str, tuple[float, float]]] = {}
        global_struct_acc: dict[str, list[float]] = {f: [] for f in ADHD_STRUCT_FEATS}
        global_fmri_acc:   dict[str, list[float]] = {p: [] for p in ADHD_FMRI_FEATS}
        per_site_struct: dict[str, dict[str, list[float]]] = {}
        per_site_fmri:   dict[str, dict[str, list[float]]] = {}
        for i in train_idx:
            d = raw_data[i]
            if d["label"] != 0: continue
            ps = per_site_struct.setdefault(d["site"], {f: [] for f in ADHD_STRUCT_FEATS})
            pf = per_site_fmri.setdefault(d["site"], {p: [] for p in ADHD_FMRI_FEATS})
            for f in ADHD_STRUCT_FEATS:
                v = d["raw_struct"].get(f)
                if v is not None and v > 0:
                    ps[f].append(v); global_struct_acc[f].append(v)
            for p in ADHD_FMRI_FEATS:
                v = d["raw_fmri"].get(p)
                if v is not None:
                    pf[p].append(v); global_fmri_acc[p].append(v)

        def _stats(values: list[float]) -> tuple[float, float] | None:
            if len(values) < 5: return None
            mu = float(np.mean(values)); sd = float(np.std(values, ddof=1))
            if sd <= 0: return None
            return mu, sd

        global_struct = {f: _stats(v) for f, v in global_struct_acc.items()}
        global_fmri   = {p: _stats(v) for p, v in global_fmri_acc.items()}
        for site, by_f in per_site_struct.items():
            norms_per_site_struct[site] = {f: _stats(v) for f, v in by_f.items()}
        for site, by_p in per_site_fmri.items():
            norms_per_site_fmri[site] = {p: _stats(v) for p, v in by_p.items()}

        def _z(value: float | None, site: str, feat: str, kind: str) -> float:
            if value is None: return 0.0
            site_norms = (norms_per_site_struct if kind == "struct"
                           else norms_per_site_fmri).get(site, {})
            ms = site_norms.get(feat)
            if ms is None:
                ms = (global_struct if kind == "struct" else global_fmri).get(feat)
            if ms is None: return 0.0
            mu, sd = ms
            return (value - mu) / sd

        def featurize(idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
            rows = []
            ys = []
            for i in idx:
                d = raw_data[i]
                row = [float(d["age"]), 1.0 if d["gender"] == 1 else 0.0]
                for f in ADHD_STRUCT_FEATS:
                    row.append(_z(d["raw_struct"].get(f), d["site"], f, "struct"))
                for p in ADHD_FMRI_FEATS:
                    row.append(_z(d["raw_fmri"].get(p), d["site"], p, "fmri"))
                rows.append(row); ys.append(d["label"])
            return np.array(rows), np.array(ys)

        Xtr, ytr = featurize(train_idx)
        Xte, yte = featurize(test_idx)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  class_weight="balanced", random_state=0)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        for arr_idx, i in enumerate(test_idx):
            oof_pred[i] = int(pred[arr_idx])
            oof_proba[i] = float(proba[arr_idx])

        bal = balanced_accuracy_score(yte, pred)
        try: auc = roc_auc_score(yte, proba)
        except ValueError: auc = float("nan")
        print(f"  hold-out {held_out_site:11s} n_train={len(train_idx):4d}  "
              f"n_test={len(test_idx):4d}  balacc={bal:.3f}  auc={auc:.3f}", flush=True)

        folds[held_out_site] = {
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "struct_norms": {site: {f: ms for f, ms in by_f.items() if ms is not None}
                              for site, by_f in norms_per_site_struct.items()},
            "fmri_norms":   {site: {p: ms for p, ms in by_p.items() if ms is not None}
                              for site, by_p in norms_per_site_fmri.items()},
            "global_struct_norms": {f: ms for f, ms in global_struct.items() if ms is not None},
            "global_fmri_norms":   {p: ms for p, ms in global_fmri.items() if ms is not None},
            "n_train": len(train_idx),
            "n_test":  len(test_idx),
            "balacc":  float(bal),
            "auc":     float(auc) if not np.isnan(auc) else None,
        }

    valid = oof_pred != -1
    y_arr = np.array([d["label"] for d in raw_data])
    if valid.sum() > 0:
        pooled_bal = balanced_accuracy_score(y_arr[valid], oof_pred[valid])
        pooled_acc = accuracy_score(y_arr[valid], oof_pred[valid])
        pooled_auc = roc_auc_score(y_arr[valid], oof_proba[valid])
        print(f"\nPOOLED LOSO: n={valid.sum()}  acc={pooled_acc:.4f}  "
              f"balacc={pooled_bal:.4f}  auc={pooled_auc:.4f}")

    out = {
        "feature_names": feature_names,
        "struct_features": ADHD_STRUCT_FEATS,
        "fmri_features": ADHD_FMRI_FEATS,
        "icv_ref_mm3": ICV_REF,
        "folds": folds,
        "pooled_loso_balacc": float(pooled_bal) if valid.sum() else None,
        "pooled_loso_auc":    float(pooled_auc) if valid.sum() else None,
        "pooled_loso_acc":    float(pooled_acc) if valid.sum() else None,
        "n_subjects_total":   int(valid.sum()),
        "n_features":         len(feature_names),
        "method": ("Per-fold z-scoring uses training-fold controls only "
                    "(no test-set leakage). sklearn LogisticRegression "
                    "(C=1.0, class_weight='balanced'). Leave-one-site-out."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
