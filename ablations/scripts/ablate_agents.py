from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ABLATIONS = {
    "baseline":           {},
    "no_structural":      {"NEUROAGENT_DISABLE_STRUCTURAL": "1"},
    "no_functional":      {"NEUROAGENT_DISABLE_FUNCTIONAL": "1"},
    "no_rag":             {"NEUROAGENT_DISABLE_RAG": "1"},
    "no_atlas":           {"NEUROAGENT_DISABLE_ATLAS": "1"},
    "no_struct_no_func":  {"NEUROAGENT_DISABLE_STRUCTURAL": "1",
                            "NEUROAGENT_DISABLE_FUNCTIONAL": "1"},
    "rule_only":          {"NEUROAGENT_RULE_ONLY": "1"},
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS))
    parser.add_argument("--model",    required=True)
    parser.add_argument("--tasks",    nargs="+",
                        default=["adhd_binary", "tumor_lat", "stroke_lat", "stroke_detection"])
    parser.add_argument("--mode",     default="neuroagent",
                        choices=["neuroagent", "direct"])
    parser.add_argument("--max-subjects", type=int, default=9999)
    parser.add_argument("--parallel",     type=int, default=4)
    parser.add_argument("--sleep",        type=float, default=0.4)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--out-root",     default=str(REPO_ROOT / "ablations" / "results"))
    args = parser.parse_args()

    for k, v in ABLATIONS[args.ablation].items():
        os.environ[k] = v
    print(f"[ablate] ablation={args.ablation}  flags={ABLATIONS[args.ablation]}")

    out_dir = Path(args.out_root) / args.ablation
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STUDY_DIR_OVERRIDE"] = str(out_dir)

    cs = importlib.import_module("eval.comprehensive_study")
    cs.STUDY_DIR = out_dir

    all_results = []
    for task in args.tasks:
        print(f"\nAblation: {args.ablation}  Model: {args.model}  Task: {task}")
        try:
            m = cs.run_experiment(args.model, task, args.max_subjects,
                                   args.resume, args.sleep, mode=args.mode,
                                   parallel=args.parallel, balanced=False)
            m["mode"] = args.mode
            m["ablation"] = args.ablation
            all_results.append(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({"model": args.model, "task": task,
                                 "mode": args.mode, "ablation": args.ablation,
                                 "error": str(e)})

    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = out_dir / f"{args.model}__{args.mode}__summary.csv"
    df.to_csv(summary, index=False)
    print(f"\nSaved: {summary}")
    cols = [c for c in ("model", "task", "mode", "ablation", "balanced_accuracy",
                         "f1_macro", "auc_roc", "sensitivity", "specificity", "n_samples")
            if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
