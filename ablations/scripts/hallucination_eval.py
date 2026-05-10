from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_REGION_PATTERNS = [
    r"caudate", r"putamen", r"pallidum", r"thalamus", r"accumbens",
    r"hippocampus", r"amygdala", r"cerebellum", r"insula",
    r"precuneus", r"cingulate", r"posterior cingulate", r"anterior cingulate",
    r"angular gyrus", r"supramarginal", r"dlpfc", r"vlpfc",
    r"medial prefrontal", r"orbitofrontal",
    r"inferior frontal", r"middle frontal", r"superior frontal",
    r"inferior temporal", r"middle temporal", r"superior temporal",
    r"inferior parietal", r"superior parietal",
    r"occipital", r"lingual", r"fusiform",
    r"corpus callosum", r"brainstem", r"basal ganglia",
    r"default mode", r"salience network", r"frontostriatal",
    r"motor cortex", r"sensorimotor", r"executive network",
]
_KNOWN_RE = re.compile(r"|".join(rf"\b{p}\b" for p in _REGION_PATTERNS),
                        re.IGNORECASE)

_SUSPECT_RE = re.compile(
    r"\b(?:left|right)?\s*"
    r"(?:Brodmann\s*area\s*\d+|"
    r"BA\d+|"
    r"area\s+\d+[a-z]?|"
    r"V\d|"
    r"M\d|"
    r"prefrontal\s+gyrus|"
    r"caudate\s+nucleus\s+head|"
    r"posterior\s+nucleus|"
    r"upper\s+brainstem)\b",
    re.IGNORECASE,
)

def extract_text(record: dict) -> str:
    parts = [record.get("summary", "") or "",
             record.get("extr_reason", "") or "",
             record.get("raw", "") or ""]
    return "\n".join(parts)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="study_postfix2")
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    study_dir = REPO_ROOT / "eval" / "results" / args.study
    if not study_dir.exists():
        sys.exit(f"Study not found: {study_dir}")

    rows: list[dict] = []
    for model_dir in sorted(study_dir.iterdir()):
        if not model_dir.is_dir(): continue
        if "__neuroagent" not in model_dir.name: continue
        model = model_dir.name.replace("__neuroagent", "")
        for jf in sorted(model_dir.glob("*_per_subject.jsonl")):
            task = jf.stem.replace("_per_subject", "")
            n_subj = n_grounded = n_suspect = 0
            grounded_counter = Counter()
            suspect_counter = Counter()
            for line in jf.read_text().splitlines():
                if not line.strip(): continue
                try: rec = json.loads(line)
                except: continue
                txt = extract_text(rec)
                if not txt: continue
                n_subj += 1
                grs = _KNOWN_RE.findall(txt.lower())
                sus = _SUSPECT_RE.findall(txt)
                n_grounded += len(grs)
                n_suspect  += len(sus)
                grounded_counter.update(g.lower() for g in grs)
                suspect_counter.update(s.lower() for s in sus)
            if n_subj > 0:
                rows.append({
                    "model": model, "task": task, "n_subjects": n_subj,
                    "grounded_claims_total": n_grounded,
                    "grounded_per_subject": round(n_grounded / n_subj, 3),
                    "suspect_claims_total": n_suspect,
                    "suspect_per_subject": round(n_suspect / n_subj, 3),
                    "hallucination_rate": (
                        round(n_suspect / max(1, n_grounded + n_suspect), 4)),
                    "top_suspect": str(suspect_counter.most_common(5)),
                })

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "ablations" / "results" / f"hallucination_{args.study}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    cols = ["model", "task", "n_subjects", "grounded_per_subject",
            "suspect_per_subject", "hallucination_rate"]
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
