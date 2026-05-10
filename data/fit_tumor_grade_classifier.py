
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LABELS_CSV = REPO_ROOT / "eval" / "ground_truth" / "tumor_labels.csv"
FASTSURFER = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "BraTS-GLI"
OUT_PATH   = REPO_ROOT / "data" / "brats_grade_logreg.json"

TUMOR_FEATS = [
    "BrainSeg_vol", "BrainSegNotVent_vol",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
    "3rd-Ventricle", "4th-Ventricle",
    "Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Thalamus", "Right-Thalamus",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Brain-Stem",
    "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
]

def parse_stats(stats_path: Path) -> tuple[dict[str, float], float]:
    volumes: dict[str, float] = {}
    icv = 0.0
    extras = {}
    if not stats_path.exists():
        return volumes, icv
    for line in stats_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("# Measure"):
            parts = line.split(",")
            if len(parts) >= 4:
                name = parts[0].split()[-1].strip()
                try: val = float(parts[3].strip())
                except ValueError: continue
                if name in ("Mask", "MaskVol"): icv = val
                if name == "BrainSeg":           extras["BrainSeg_vol"] = val
                if name == "BrainSegNotVent":    extras["BrainSegNotVent_vol"] = val
            continue
        if line.startswith("#") or not line: continue
        cols = line.split()
        if len(cols) < 5: continue
        try: volumes[cols[4]] = float(cols[3])
        except: pass
    volumes.update(extras)
    return volumes, icv

def find_stats(subj_dir: Path) -> Path | None:
    for name in ("aseg+DKT.VINN.stats", "aseg.VINN.stats", "aseg.stats"):
        p = subj_dir / "stats" / name
        if p.exists(): return p
    return None

def featurize(volumes: dict, icv: float) -> dict[str, float]:
    if icv <= 0: return {}
    out: dict[str, float] = {}
    for f in TUMOR_FEATS:
        v = volumes.get(f, 0.0)
        if v > 0:
            out[f] = v / icv * 1e6
    for pair in [("Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
                  ("Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter"),
                  ("Left-Caudate", "Right-Caudate"),
                  ("Left-Putamen", "Right-Putamen"),
                  ("Left-Thalamus", "Right-Thalamus"),
                  ("Left-Hippocampus", "Right-Hippocampus"),
                  ("Left-Amygdala", "Right-Amygdala"),
                  ("Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex")]:
        l = volumes.get(pair[0], 0.0); r = volumes.get(pair[1], 0.0)
        if l > 0 and r > 0:
            out[f"LI_{pair[0].replace('Left-', '')}"] = (l - r) / (l + r)
            out[f"sum_{pair[0].replace('Left-', '')}_per_icv"] = (l + r) / icv * 1e6
    tot_l = sum(v for k, v in volumes.items() if k.startswith("Left-") and v > 0)
    tot_r = sum(v for k, v in volumes.items() if k.startswith("Right-") and v > 0)
    if tot_l + tot_r > 0:
        out["TOTAL_LI"] = (tot_l - tot_r) / (tot_l + tot_r)
        out["abs_TOTAL_LI"] = abs((tot_l - tot_r) / (tot_l + tot_r))
    vl = volumes.get("Left-Lateral-Ventricle", 0.0)
    vr = volumes.get("Right-Lateral-Ventricle", 0.0)
    if vl + vr > 0:
        out["abs_LI_ventricle"] = abs((vl - vr) / (vl + vr))
        out["total_ventricle_per_icv"] = (vl + vr) / icv * 1e6
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(LABELS_CSV) as f:
        for r in csv.DictReader(f):
            if not r.get("grade_label"): continue
            rows.append((r["subject_id"], int(r["grade_label"])))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Building features for {len(rows)} BraTS subjects...", flush=True)

    feats: list[dict] = []
    n_ok = 0; n_fail = 0
    for i, (sid, grade) in enumerate(rows):
        stats = find_stats(FASTSURFER / sid)
        if stats is None:
            n_fail += 1; continue
        volumes, icv = parse_stats(stats)
        if icv <= 0:
            n_fail += 1; continue
        feature_vec = featurize(volumes, icv)
        if not feature_vec:
            n_fail += 1; continue
        feats.append({"sid": sid, "grade": grade, "feats": feature_vec})
        n_ok += 1
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} processed, {n_ok} OK, {n_fail} fail", flush=True)

    print(f"\n{n_ok} usable BraTS subjects with FastSurfer features")
    if n_ok < 50:
        sys.exit(f"Too few subjects ({n_ok}), aborting")

    feature_names = sorted(set(k for d in feats for k in d["feats"]))
    X = np.array([[d["feats"].get(f, 0.0) for f in feature_names] for d in feats])
    y = np.array([d["grade"] for d in feats])
    sids = [d["sid"] for d in feats]
    print(f"Feature matrix: {X.shape}  labels: {np.bincount(y)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    folds: list[dict] = []
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), 0.5)
    sid2fold = {}
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  class_weight="balanced", random_state=0)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_pred[te] = pred; oof_proba[te] = proba
        bal = balanced_accuracy_score(y[te], pred)
        try: auc = roc_auc_score(y[te], proba)
        except: auc = float("nan")
        for sid_idx in te:
            sid2fold[sids[sid_idx]] = fold_i
        print(f"  fold {fold_i}  n_train={len(tr):4d}  n_test={len(te):4d}  "
              f"balacc={bal:.3f}  auc={auc:.3f}", flush=True)
        folds.append({
            "fold": fold_i,
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "n_train": int(len(tr)),
            "n_test":  int(len(te)),
            "balacc":  float(bal),
            "auc":     float(auc) if not np.isnan(auc) else None,
        })

    pooled_bal = balanced_accuracy_score(y, oof_pred)
    pooled_acc = accuracy_score(y, oof_pred)
    pooled_auc = roc_auc_score(y, oof_proba)
    print(f"\nPOOLED 5-fold CV: n={len(y)}  acc={pooled_acc:.4f}  "
          f"balacc={pooled_bal:.4f}  auc={pooled_auc:.4f}")

    out = {
        "feature_names":         feature_names,
        "folds":                 folds,
        "subject_to_fold":       sid2fold,
        "pooled_balacc":         float(pooled_bal),
        "pooled_acc":            float(pooled_acc),
        "pooled_auc":            float(pooled_auc),
        "n_subjects":            len(y),
        "method": ("FastSurfer-only HGG-vs-LGG: stratified 5-fold CV, "
                   "LogisticRegression(C=1.0, class_weight='balanced'). "
                   "NO segmentation-mask features, agent must infer grade "
                   "from FastSurfer-derived volumetric/asymmetry features."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
