from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.structural_mri_agent import StructuralMRIAgent

FS_ROOT = PROJECT_ROOT / "preprocessing" / "fastsurfer_output"
GT_DIR  = PROJECT_ROOT / "eval" / "ground_truth"

FASTSURFER_DIRS = {
    "adhd":   FS_ROOT / "ADHD-200",
    "tumor":  FS_ROOT / "BraTS-GLI",
    "stroke": FS_ROOT / "ATLAS-v2",
}

_ADHD_STRUCTURES = {"caudate", "putamen", "thalamus", "pallidum", "accumbens",
                     "frontal", "cingulate", "insula"}

def _predict_adhd(output) -> int | None:
    if output.status == "failed":
        return None
    hits = [
        f for f in output.findings
        if f.severity in ("moderate", "severe")
        and any(kw in f.region.canonical_name.lower() for kw in _ADHD_STRUCTURES)
    ]
    return 1 if len(hits) >= 2 else 0

def _predict_tumor_grade(output) -> int | None:
    if output.status == "failed":
        return None
    for f in output.findings:
        if f.finding_type in ("mass_effect", "lesion") and f.severity in ("severe", "moderate"):
            return 1
    return 0

def _predict_lateralization(output) -> str | None:
    if output.status == "failed":
        return None
    left_count = right_count = bilateral_count = 0
    for f in output.findings:
        h = f.region.hemisphere.lower()
        if h == "left":
            left_count += 1
        elif h == "right":
            right_count += 1
        elif h in ("bilateral", "midline"):
            bilateral_count += 1
    if left_count == 0 and right_count == 0:
        return None
    if abs(left_count - right_count) <= 1:
        return "bilateral"
    return "left" if left_count > right_count else "right"

def _binary_metrics(y_true, y_pred, label: str) -> dict:
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score, f1_score,
        confusion_matrix, classification_report,
    )
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    metrics = {
        "n": len(y_true),
        "label": label,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_positive": round(float(f1_score(y_true, y_pred, pos_label=1, average="binary",
                                             zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = round(float(roc_auc_score(y_true, y_pred)), 4)
        except Exception:
            metrics["auc"] = None
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics

def _multi_class_metrics(y_true, y_pred, label: str) -> dict:
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "n": len(y_true),
        "label": label,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def evaluate_adhd(labels_df: pd.DataFrame, max_subjects: int | None,
                  agent: StructuralMRIAgent) -> dict:
    df = labels_df.copy()
    if max_subjects:
        df = df.head(max_subjects)

    y_true, y_pred, sites, failed = [], [], [], []
    fs_dir = FASTSURFER_DIRS["adhd"]

    for i, row in enumerate(df.itertuples(), 1):
        subj_id = row.subject_id
        subj_dir = fs_dir / subj_id
        print(f"  [{i}/{len(df)}] {subj_id}...", end=" ", flush=True)
        if not subj_dir.exists():
            print("SKIP (no stats dir)")
            continue
        try:
            out = agent.run(subject_id=subj_id, stats_dir=subj_dir)
            pred = _predict_adhd(out)
            if pred is None:
                failed.append(subj_id)
                print(f"FAILED ({(out.errors or ['?'])[0][:60]})")
                continue
            y_true.append(row.label)
            y_pred.append(pred)
            sites.append(row.site)
            print(f"true={row.label} pred={pred}")
        except Exception as e:
            failed.append(subj_id)
            print(f"ERROR: {e}")

    if not y_true:
        return {"error": "No predictions produced"}

    overall = _binary_metrics(y_true, y_pred, "adhd_overall")
    per_site = {}
    for site in sorted(set(sites)):
        idx = [i for i, s in enumerate(sites) if s == site]
        if len(idx) < 2:
            continue
        per_site[site] = _binary_metrics(
            [y_true[i] for i in idx], [y_pred[i] for i in idx], site
        )

    return {"overall": overall, "per_site": per_site, "n_failed": len(failed)}

def evaluate_tumor(labels_df: pd.DataFrame, max_subjects: int | None,
                   agent: StructuralMRIAgent) -> dict:
    df = labels_df.dropna(subset=["grade_label"]).copy()
    df["grade_label"] = df["grade_label"].astype(int)
    if max_subjects:
        df = df.head(max_subjects)

    y_grade_true, y_grade_pred = [], []
    y_lat_true,   y_lat_pred   = [], []
    failed = []
    fs_dir = FASTSURFER_DIRS["tumor"]

    for i, row in enumerate(df.itertuples(), 1):
        subj_id = row.subject_id
        subj_dir = fs_dir / subj_id
        print(f"  [{i}/{len(df)}] {subj_id}...", end=" ", flush=True)
        if not subj_dir.exists():
            print("SKIP (no stats dir)")
            continue
        try:
            out = agent.run(subject_id=subj_id, stats_dir=subj_dir)
            grade_pred = _predict_tumor_grade(out)
            lat_pred   = _predict_lateralization(out)
            if grade_pred is None:
                failed.append(subj_id)
                print(f"FAILED")
                continue
            y_grade_true.append(row.grade_label)
            y_grade_pred.append(grade_pred)
            if lat_pred is not None and hasattr(row, "lateralization") and pd.notna(row.lateralization):
                y_lat_true.append(row.lateralization)
                y_lat_pred.append(lat_pred)
            print(f"grade true={row.grade_label} pred={grade_pred}  "
                  f"lat true={getattr(row,'lateralization',None)} pred={lat_pred}")
        except Exception as e:
            failed.append(subj_id)
            print(f"ERROR: {e}")

    result = {"n_failed": len(failed)}
    if y_grade_true:
        result["grade"] = _binary_metrics(y_grade_true, y_grade_pred, "tumor_grade")
    if y_lat_true:
        result["lateralization"] = _multi_class_metrics(y_lat_true, y_lat_pred, "tumor_lat")
    return result

def evaluate_stroke(labels_df: pd.DataFrame, max_subjects: int | None,
                    agent: StructuralMRIAgent) -> dict:
    df = labels_df.dropna(subset=["lateralization"]).copy()
    if max_subjects:
        df = df.head(max_subjects)

    y_lat_true, y_lat_pred, failed = [], [], []
    fs_dir = FASTSURFER_DIRS["stroke"]

    for i, row in enumerate(df.itertuples(), 1):
        subj_id = row.subject_id
        subj_dir = fs_dir / subj_id
        print(f"  [{i}/{len(df)}] {subj_id}...", end=" ", flush=True)
        if not subj_dir.exists():
            print("SKIP (no stats dir)")
            continue
        try:
            out  = agent.run(subject_id=subj_id, stats_dir=subj_dir)
            pred = _predict_lateralization(out)
            if pred is None:
                failed.append(subj_id)
                print("no lateralization predicted")
                continue
            y_lat_true.append(row.lateralization)
            y_lat_pred.append(pred)
            print(f"lat true={row.lateralization} pred={pred}")
        except Exception as e:
            failed.append(subj_id)
            print(f"ERROR: {e}")

    result = {"n_failed": len(failed)}
    if y_lat_true:
        result["lateralization"] = _multi_class_metrics(y_lat_true, y_lat_pred, "stroke_lat")
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuroAgent on ground-truth labels")
    parser.add_argument("--condition", default="all",
                        choices=["adhd", "tumor", "stroke", "all"])
    parser.add_argument("--gt-dir",       default=str(GT_DIR))
    parser.add_argument("--out-dir",      default=str(PROJECT_ROOT / "eval" / "results"))
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir)

    all_results = {}

    if args.condition in ("adhd", "all"):
        gt_path = gt_dir / "adhd_labels.csv"
        if not gt_path.exists():
            print(f"[ADHD] GT file not found: {gt_path}. Run extract_ground_truth.py first.")
        else:
            labels = pd.read_csv(gt_path)
            print(f"\nADHD Evaluation ({len(labels)} subjects with labels)")
            agent  = StructuralMRIAgent(condition="adhd")
            result = evaluate_adhd(labels, args.max_subjects, agent)
            all_results["adhd"] = result
            print("\nADHD Results")
            print(json.dumps(result, indent=2))

    if args.condition in ("tumor", "all"):
        gt_path = gt_dir / "tumor_labels.csv"
        if not gt_path.exists():
            print(f"[Tumor] GT file not found: {gt_path}. Run extract_ground_truth.py first.")
        else:
            labels = pd.read_csv(gt_path)
            print(f"\nTumor Evaluation ({len(labels)} subjects with labels)")
            agent  = StructuralMRIAgent(condition="tumor")
            result = evaluate_tumor(labels, args.max_subjects, agent)
            all_results["tumor"] = result
            print("\nTumor Results")
            print(json.dumps(result, indent=2))

    if args.condition in ("stroke", "all"):
        gt_path = gt_dir / "stroke_labels.csv"
        if not gt_path.exists():
            print(f"[Stroke] GT file not found: {gt_path}. Run extract_ground_truth.py first.")
        else:
            labels = pd.read_csv(gt_path)
            print(f"\nStroke Evaluation ({len(labels)} subjects with labels)")
            agent  = StructuralMRIAgent(condition="stroke")
            result = evaluate_stroke(labels, args.max_subjects, agent)
            all_results["stroke"] = result
            print("\nStroke Results")
            print(json.dumps(result, indent=2))

    out_path = out_dir / f"neuroagent_{args.condition}_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
