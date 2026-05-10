from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import NeuroAgentOrchestrator

FS_ROOT = PROJECT_ROOT / "preprocessing" / "fastsurfer_output"
GT_DIR  = PROJECT_ROOT / "eval"  / "ground_truth"
OUT_DIR = PROJECT_ROOT / "eval"  / "results"

FASTSURFER_DIRS = {
    "adhd":   FS_ROOT / "ADHD-200",
    "tumor":  FS_ROOT / "BraTS-GLI",
    "stroke": FS_ROOT / "ATLAS-v2",
}

PRICES: dict[str, dict] = {
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "claude-sonnet-4-6":{"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":  {"input": 15.00, "output": 75.00},
}

def _estimate_cost(usage: dict, model: str) -> float:
    p = PRICES.get(model, PRICES["gpt-4o-mini"])
    return (usage["prompt_tokens"] / 1e6 * p["input"]
          + usage["completion_tokens"] / 1e6 * p["output"])

def _predict_from_summary(condition: str, diagnostic_summary: str,
                          confidence: float) -> int | None:
    s = (diagnostic_summary or "").lower()

    if condition == "adhd":
                                                                                
        adhd_markers = ["adhd", "attention deficit", "consistent with adhd",
                        "supports adhd", "indicative of adhd"]
        ctrl_markers = ["typically developing", "no findings", "normal",
                        "not consistent with adhd", "control", "no evidence"]
        adhd_hit = any(m in s for m in adhd_markers)
        ctrl_hit = any(m in s for m in ctrl_markers)
        if adhd_hit and not ctrl_hit:
            return 1
        if ctrl_hit and not adhd_hit:
            return 0
                                           
        return 1 if confidence >= 0.5 else 0

    if condition == "tumor":
                                                                    
        high_markers = ["high-grade", "high grade", "glioblastoma", "gbm",
                        "grade iv", "grade 4", "anaplastic", "grade iii",
                        "ring enhancement", "necrosis", "severe mass effect"]
        low_markers  = ["low-grade", "low grade", "grade ii", "grade 2",
                        "non-enhancing", "indolent", "minimal mass effect"]
        if any(m in s for m in high_markers):
            return 1
        if any(m in s for m in low_markers):
            return 0
        return None

    return None

def run_eval(condition: str, max_subjects: int | None, model: str,
             mock: bool, resume: bool, out_dir: Path) -> dict:
    label_files = {
        "adhd":   GT_DIR / "adhd_labels.csv",
        "tumor":  GT_DIR / "tumor_labels.csv",
    }
    if condition not in label_files:
        raise ValueError(f"Unsupported condition: {condition}")
    if not label_files[condition].exists():
        raise FileNotFoundError(f"Run extract_ground_truth.py first: {label_files[condition]}")

    labels = pd.read_csv(label_files[condition])
    if condition == "tumor":
        labels = labels.dropna(subset=["grade_label"])
    if max_subjects:
        labels = labels.head(max_subjects)

    results_dir = out_dir / "orchestrator" / condition
    results_dir.mkdir(parents=True, exist_ok=True)

    fs_dir = FASTSURFER_DIRS[condition]
    orch   = NeuroAgentOrchestrator(condition=condition, model=model, mock=mock)

    cum_input, cum_output, n_done = 0, 0, 0
    rows = []

    for i, row in enumerate(labels.itertuples(), 1):
        sid       = row.subject_id
        result_fp = results_dir / f"{sid}.json"

        if resume and result_fp.exists():
            saved = json.loads(result_fp.read_text())
            rows.append(saved)
            cum_input  += saved.get("prompt_tokens", 0)
            cum_output += saved.get("completion_tokens", 0)
            n_done     += 1
            print(f"  [{i}/{len(labels)}] {sid} (cached)")
            continue

        subj_dir = fs_dir / sid
        if not subj_dir.exists():
            print(f"  [{i}/{len(labels)}] {sid} SKIP (no stats dir)")
            continue

        if condition == "adhd":
            true_label = int(row.label)
        else:
            true_label = int(row.grade_label)

        print(f"  [{i}/{len(labels)}] {sid}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = orch.run(subject_id=sid, stats_dir=subj_dir, output_dir=None)
            usage  = orch.last_usage.copy()
            elapsed = time.time() - t0

            pred = _predict_from_summary(condition, result.diagnostic_summary,
                                          result.confidence)

            entry = {
                "subject_id":           sid,
                "true":                 true_label,
                "pred":                 pred,
                "confidence":           result.confidence,
                "diagnostic_summary":   result.diagnostic_summary,
                "elapsed_s":            round(elapsed, 1),
                "prompt_tokens":        usage["prompt_tokens"],
                "completion_tokens":    usage["completion_tokens"],
                "n_api_calls":          usage["n_calls"],
            }
            result_fp.write_text(json.dumps(entry, indent=2))
            rows.append(entry)

            cum_input  += usage["prompt_tokens"]
            cum_output += usage["completion_tokens"]
            n_done     += 1

            cost_so_far = _estimate_cost({"prompt_tokens": cum_input,
                                          "completion_tokens": cum_output}, model)
            print(f"true={true_label} pred={pred}  "
                  f"({usage['prompt_tokens']}+{usage['completion_tokens']} tok, "
                  f"{elapsed:.0f}s, ${cost_so_far:.3f} so far)")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)

    valid = [r for r in rows if r.get("pred") in (0, 1)]
    metrics: dict = {
        "condition": condition, "model": model, "mock": mock,
        "n_total": len(rows), "n_valid": len(valid),
        "tokens_input": cum_input, "tokens_output": cum_output,
        "estimated_cost_usd": round(_estimate_cost(
            {"prompt_tokens": cum_input, "completion_tokens": cum_output}, model), 4),
    }
    if valid:
        from sklearn.metrics import (
            balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,
        )
        y_true = np.array([r["true"] for r in valid])
        y_pred = np.array([r["pred"] for r in valid])
        metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
        metrics["f1_macro"]          = round(float(f1_score(y_true, y_pred, average="macro",
                                                             zero_division=0)), 4)
        metrics["confusion_matrix"]  = confusion_matrix(y_true, y_pred).tolist()
        if len(np.unique(y_true)) > 1:
            try:
                metrics["auc"] = round(float(roc_auc_score(y_true, y_pred)), 4)
            except Exception:
                pass

    summary_fp = out_dir / f"orchestrator_{condition}_summary.json"
    summary_fp.write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Full NeuroAgent orchestrator eval with cost tracking")
    parser.add_argument("--condition",    required=True, choices=["adhd", "tumor"])
    parser.add_argument("--model",        default="gpt-4o-mini")
    parser.add_argument("--max-subjects", type=int, default=3)
    parser.add_argument("--mock",         action="store_true")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--out-dir",      default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nNeuroAgent Orchestrator Eval")
    print(f"  Condition:    {args.condition}")
    print(f"  Model:        {args.model}")
    print(f"  Mock:         {args.mock}")
    print(f"  Max subjects: {args.max_subjects}")
    if not args.mock and args.model in PRICES:
        p = PRICES[args.model]
        print(f"  Pricing:      ${p['input']}/M input, ${p['output']}/M output")
    print()

    metrics = run_eval(args.condition, args.max_subjects, args.model,
                       args.mock, args.resume, out_dir)

    print("\nFinal Metrics")
    print(json.dumps(metrics, indent=2))
    print(f"\nTotal estimated cost: ${metrics['estimated_cost_usd']}")
    if metrics.get("n_valid"):
        per_subj = metrics["estimated_cost_usd"] / metrics["n_valid"]
        print(f"   Per subject: ${per_subj:.4f}")
        print(f"   Extrapolated to 584 ADHD subjects:  ${per_subj*584:.2f}")
        print(f"   Extrapolated to 1621 tumor subjects: ${per_subj*1621:.2f}")

if __name__ == "__main__":
    main()
