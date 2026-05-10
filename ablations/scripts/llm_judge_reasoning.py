from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

JUDGE_SYSTEM = """You are a neurologist scoring AI-generated diagnostic reasoning traces.

Score each trace on FOUR dimensions:

1. ATLAS_GROUNDING (1-5):
   1 = region names are vague ("frontal area") or invented ("posterior nucleus")
   3 = some regions cite recognizable atlas labels (Left-Caudate, R.IFG, AAL #21)
   5 = every region claim is grounded in a known atlas (AAL, DKT, Desikan, FreeSurfer)

2. CLINICAL_RELEVANCE (1-5):
   1 = cited literature is irrelevant or absent
   3 = some references support the finding
   5 = every clinical claim is backed by an appropriate citation (PMID, study name, criteria)

3. LOGICAL_REASONING (1-5):
   1 = conclusion contradicts the evidence
   3 = conclusion partially follows from evidence
   5 = conclusion is fully supported by the cited measurements

4. HALLUCINATION_COUNT (integer >= 0):
   Count specific factual claims (region names, citations, measurements) that
   appear made up or cannot be verified from the trace itself.

Respond with ONLY a JSON object:
  {"atlas_grounding": int, "clinical_relevance": int, "logical_reasoning": int, "hallucination_count": int, "comment": "one short sentence"}
No preamble, no markdown fences."""

def call_judge(client, model: str, system: str, user: str,
               max_retries: int = 3) -> dict | None:
    for retry in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=400,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            import re
            text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{[^}]*\}", text, re.DOTALL)
                if m:
                    try: return json.loads(m.group())
                    except: pass
        except Exception as e:
            if retry == max_retries - 1:
                print(f"    judge error after {max_retries} tries: {e}")
            time.sleep(5 * (retry + 1))
    return None

def load_jsonl(p: Path, max_n: int = 0) -> list[dict]:
    out = []
    if not p.exists(): return out
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try: out.append(json.loads(line))
        except: pass
        if max_n and len(out) >= max_n: break
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="gpt-oss-20b")
    parser.add_argument("--judge-url", default=None)
    parser.add_argument("--judge-key", default="local-vllm-key")
    parser.add_argument("--ablations-dir", default=str(REPO_ROOT / "ablations" / "results"))
    parser.add_argument("--n-per-cell", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_ROOT / "ablations" / "results" / "llm_judge_scores.csv"))
    parser.add_argument("--target-model", default="Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    if args.judge_url:
        url, key = args.judge_url, args.judge_key
    else:
        ep = REPO_ROOT / "baselines" / "vllm_endpoints" / f"{args.judge_model}.json"
        if not ep.exists():
            sys.exit(f"No endpoint file for {args.judge_model}; pass --judge-url or start vLLM first.")
        d = json.loads(ep.read_text())
        url, key = d["url"], d.get("api_key", "local-vllm-key")
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key=key)
    print(f"Judge: {args.judge_model} @ {url}", flush=True)

    rng = random.Random(args.seed)

    rows: list[dict] = []
    abl_root = Path(args.ablations_dir)
    if not abl_root.exists():
        sys.exit(f"Ablations dir not found: {abl_root}")

    for ablation_dir in sorted(abl_root.iterdir()):
        if not ablation_dir.is_dir(): continue
        ablation = ablation_dir.name
        if ablation in ("llm_judge_scores.csv",): continue
        model_dir = ablation_dir / f"{args.target_model}__neuroagent"
        if not model_dir.exists(): continue
        for task in ("adhd_binary", "tumor_grade", "stroke_lat"):
            f = model_dir / f"{task}_per_subject.jsonl"
            if not f.exists(): continue
            subjects = load_jsonl(f)
            if not subjects: continue
            sample = rng.sample(subjects, min(args.n_per_cell, len(subjects)))
            print(f"\n[{ablation}/{task}] judging {len(sample)} subjects", flush=True)
            for j, rec in enumerate(sample, 1):
                trace = rec.get("summary", "") or ""
                if not trace.strip(): continue
                user = (f"Subject ID: {rec.get('subject_id')}\n"
                        f"Task: {task}\n"
                        f"Predicted label: {rec.get('pred')}, true label: {rec.get('true')}\n"
                        f"Reasoning trace excerpt:\n\n{trace[:3000]}")
                scores = call_judge(client, args.judge_model, JUDGE_SYSTEM, user)
                if scores is None: continue
                rows.append({
                    "ablation": ablation, "task": task,
                    "subject_id": rec.get("subject_id"),
                    "true": rec.get("true"), "pred": rec.get("pred"),
                    "atlas_grounding": scores.get("atlas_grounding"),
                    "clinical_relevance": scores.get("clinical_relevance"),
                    "logical_reasoning": scores.get("logical_reasoning"),
                    "hallucination_count": scores.get("hallucination_count"),
                    "judge_comment": scores.get("comment", ""),
                })
                if j % 5 == 0:
                    print(f"    {j}/{len(sample)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} judged traces)")

    if not df.empty:
        agg = df.groupby(["ablation", "task"]).agg(
            n=("subject_id", "count"),
            atlas_grounding=("atlas_grounding", "mean"),
            clinical_relevance=("clinical_relevance", "mean"),
            logical_reasoning=("logical_reasoning", "mean"),
            hallucination_count=("hallucination_count", "mean"),
        ).round(3)
        print("\naggregate scores (mean)")
        print(agg.to_string())

if __name__ == "__main__":
    main()
