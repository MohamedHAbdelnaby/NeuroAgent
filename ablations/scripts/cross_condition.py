from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from agents.orchestrator import NeuroAgentOrchestrator

    ep_file = REPO_ROOT / "baselines" / "vllm_endpoints" / f"{args.model}.json"
    if not ep_file.exists():
        sys.exit(f"No vLLM endpoint for {args.model}; spin one up first.")
    ep = json.loads(ep_file.read_text())

    GT = REPO_ROOT / "eval" / "ground_truth"
    FS_ADHD   = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "ADHD-200"
    FS_TUMOR  = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "BraTS-GLI"
    FS_STROKE = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "ATLAS-v2"

    import random, pandas as pd
    random.seed(7)
    subjects_by_task: dict[str, list[dict]] = {}
    df = pd.read_csv(GT / "adhd_labels.csv")
    sample = df.sample(min(args.n, len(df)), random_state=7)
    subjects_by_task["adhd_binary"] = [
        {"sid": r.subject_id, "true": int(r.label),
         "fs": str(FS_ADHD / r.subject_id), "task_cond": "adhd"}
        for r in sample.itertuples()
    ]
    df = pd.read_csv(GT / "tumor_labels.csv")
    df = df[df["lateralization"].isin(["left", "right"])]
    df["label"] = (df["lateralization"] == "right").astype(int)
    sample = df.sample(min(args.n, len(df)), random_state=7)
    subjects_by_task["tumor_lat"] = [
        {"sid": r.subject_id, "true": int(r.label),
         "fs": str(FS_TUMOR / r.subject_id),
         "seg": getattr(r, "seg_path", None),
         "task_cond": "tumor"}
        for r in sample.itertuples()
    ]
    df = pd.read_csv(GT / "stroke_labels.csv")
    df = df[df["lateralization"].isin(["left", "right"])]
    df["label"] = (df["lateralization"] == "right").astype(int)
    sample = df.sample(min(args.n, len(df)), random_state=7)
    subjects_by_task["stroke_lat"] = [
        {"sid": r.subject_id, "true": int(r.label),
         "fs": str(FS_STROKE / r.subject_id), "task_cond": "stroke"}
        for r in sample.itertuples()
    ]

    rows: list[dict] = []
    for task, subjects in subjects_by_task.items():
        task_cond = subjects[0]["task_cond"]
        for orch_cond in ("adhd", "tumor", "stroke"):
            print(f"\n[cross] task={task}  orchestrator condition={orch_cond}")
            orch = NeuroAgentOrchestrator(condition=orch_cond, model=args.model,
                                           base_url=ep["url"], api_key=ep["api_key"])
            n_correct = 0; n_total = 0; t0 = time.time()
            for s in subjects:
                try:
                    res = orch.run(
                        subject_id=s["sid"], stats_dir=s["fs"],
                        seg_path=s.get("seg"),
                        task=task,
                    )
                    if res.predicted_label == s["true"]:
                        n_correct += 1
                    n_total += 1
                except Exception as e:
                    print(f"  err {s['sid']}: {e}")
            acc = n_correct / max(1, n_total)
            elapsed = time.time() - t0
            rows.append({
                "task": task, "task_native_condition": task_cond,
                "orchestrator_condition": orch_cond,
                "n": n_total, "accuracy": round(acc, 4),
                "transfer": "native" if orch_cond == task_cond else f"{orch_cond}->{task_cond}",
                "elapsed_s": round(elapsed, 1),
            })
            print(f"  acc={acc:.3f}  n={n_total}  ({elapsed:.0f}s)")

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "ablations" / "results" / "cross_condition.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
