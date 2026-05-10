from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
_STUDY_NAME = os.environ.get("STUDY_NAME", "study_balanced_200")
STUDY_DIR = PROJECT_ROOT / "eval" / "results" / _STUDY_NAME

PRICING_USD_PER_1M = {
    "gpt-oss-120b":           {"in": 0.10,  "out": 0.50},
    "gpt-oss-20b":            {"in": 0.05,  "out": 0.20},
    "Llama-3.1-8B-Instruct":  {"in": 0.06,  "out": 0.06},
    "Mistral-7B-v0.3":        {"in": 0.05,  "out": 0.05},
    "Qwen3-14B":              {"in": 0.07,  "out": 0.07},
    "Gemma-3-12b-it":         {"in": 0.07,  "out": 0.07},
    "Kimi-K2.6":              {"in": 0.40,  "out": 2.00},
    "MiniMax-M2.7":           {"in": 0.20,  "out": 1.00},
}

def collect_rows():
    rows = []
    for d in sorted(STUDY_DIR.iterdir() if STUDY_DIR.exists() else []):
        if not d.is_dir() or d.name in ("plots", "_pre_fmri", "_excluded_kimi", "_stale_pre_fix"):
            continue
        is_na = "__neuroagent" in d.name
        model = d.name.replace("__neuroagent", "")
        mode = "neuroagent" if is_na else "direct"
        for fp in sorted(d.glob("*_per_subject.jsonl")):
            task = fp.stem.replace("_per_subject", "")
            for line in fp.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                p_tok = rec.get("prompt_tokens")
                c_tok = rec.get("completion_tokens")
                total = rec.get("tokens", 0) or 0
                if p_tok is None or c_tok is None:
                    p_tok = c_tok = None
                rows.append({
                    "model": model,
                    "mode": mode,
                    "task": task,
                    "subject_id": rec.get("subject_id"),
                    "elapsed_s": rec.get("elapsed_s", 0),
                    "n_calls": rec.get("n_calls"),
                    "tokens": int(total),
                    "prompt_tokens": int(p_tok) if p_tok is not None else None,
                    "completion_tokens": int(c_tok) if c_tok is not None else None,
                    "pred_valid": rec.get("pred") in (0, 1),
                })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["model", "mode", "task"], as_index=False).agg(
        n_subjects=("subject_id", "count"),
        n_valid=("pred_valid", "sum"),
        total_tokens=("tokens", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        n_calls_total=("n_calls", "sum"),
        elapsed_total_s=("elapsed_s", "sum"),
    )
    g["mean_tokens_per_subject"] = (g["total_tokens"] / g["n_subjects"]).round(0)
    g["mean_calls_per_subject"]  = (g["n_calls_total"] / g["n_subjects"]).round(2)
    return g

DEFAULT_PROMPT_FRAC = 0.70

def add_cost(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    in_cost, out_cost, total_cost, est_flag = [], [], [], []
    for _, r in g.iterrows():
        price = PRICING_USD_PER_1M.get(r["model"])
        p_tok = r.get("prompt_tokens") or 0
        c_tok = r.get("completion_tokens") or 0
        total = r.get("total_tokens", 0) or 0
        if (not p_tok and not c_tok) and total:
            p_tok = total * DEFAULT_PROMPT_FRAC
            c_tok = total * (1 - DEFAULT_PROMPT_FRAC)
            est = True
        else:
            est = False
        if price is None or total == 0:
            in_cost.append(None); out_cost.append(None); total_cost.append(None)
            est_flag.append(False); continue
        ic = (p_tok / 1_000_000) * price["in"]
        oc = (c_tok / 1_000_000) * price["out"]
        in_cost.append(round(ic, 4)); out_cost.append(round(oc, 4))
        total_cost.append(round(ic + oc, 4))
        est_flag.append(est)
    g["cost_in_usd"]    = in_cost
    g["cost_out_usd"]   = out_cost
    g["cost_total_usd"] = total_cost
    g["cost_estimated"] = est_flag
    return g

def main():
    df = collect_rows()
    if df.empty:
        print(f"No data found under {STUDY_DIR}")
        sys.exit(1)
    g = summarize(df)
    g = add_cost(g)

    out_csv = STUDY_DIR / "cost_analysis.csv"
    g.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print()

    show_cols = ["model", "mode", "task", "n_subjects",
                 "total_tokens", "prompt_tokens", "completion_tokens",
                 "n_calls_total", "mean_tokens_per_subject", "mean_calls_per_subject",
                 "cost_total_usd"]
    show_cols = [c for c in show_cols if c in g.columns]
    print(g[show_cols].to_string(index=False))
    print()

    print("Totals per (model, mode):")
    rollup = df.groupby(["model", "mode"], as_index=False).agg(
        n_subjects=("subject_id", "count"),
        total_tokens=("tokens", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        elapsed_total_s=("elapsed_s", "sum"),
    )
    rollup["mean_tokens_per_subject"] = (rollup["total_tokens"] / rollup["n_subjects"]).round(0)
    rollup["wall_clock_h"]            = (rollup["elapsed_total_s"] / 3600).round(2)
    rollup = add_cost(rollup)
    print(rollup.to_string(index=False))

    grand = pd.DataFrame([{
        "n_rows":            len(df),
        "total_tokens":      int(df["tokens"].sum()),
        "total_prompt":      int(df["prompt_tokens"].sum(skipna=True)) if df["prompt_tokens"].notna().any() else None,
        "total_completion":  int(df["completion_tokens"].sum(skipna=True)) if df["completion_tokens"].notna().any() else None,
        "total_elapsed_h":   round(df["elapsed_s"].sum() / 3600, 2),
    }])
    print()
    print("Grand totals across study:")
    print(grand.to_string(index=False))

if __name__ == "__main__":
    main()
