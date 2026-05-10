from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_pred_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out

def load_pheno(adhd_csv: Path) -> dict[str, dict]:
    import csv as _csv
    out: dict[str, dict] = {}
    if not adhd_csv.exists(): return out
    with open(adhd_csv) as f:
        for row in _csv.DictReader(f):
            sid = row["subject_id"]
            try: age = float(row.get("age", "")) if row.get("age") else None
            except: age = None
            try: gender = int(round(float(row.get("gender", "")))) if row.get("gender") else None
            except: gender = None
            out[sid] = {"site": row.get("site"), "age": age, "gender": gender,
                        "label": int(row["label"]) if row.get("label") else None}
    return out

def analyze_adhd(records: list[dict], pheno: dict, lines: list[str]) -> None:
    fps = [r for r in records if r["true"] == 0 and r["pred"] == 1]
    fns = [r for r in records if r["true"] == 1 and r["pred"] == 0]
    lines.append(f"\n### ADHD failure breakdown ({len(records)} total: "
                  f"{len(fps)} FP, {len(fns)} FN)\n")

    def by_site(rs):
        c = Counter(pheno.get(r["subject_id"], {}).get("site") for r in rs)
        return ", ".join(f"{s}={n}" for s, n in c.most_common())
    def by_gender(rs):
        c = Counter(pheno.get(r["subject_id"], {}).get("gender") for r in rs)
        return ", ".join(f"{'male' if k==1 else 'female' if k==0 else '?'}={v}"
                          for k,v in c.items())
    def age_stats(rs):
        ages = [pheno.get(r["subject_id"], {}).get("age") for r in rs]
        ages = [a for a in ages if a is not None]
        if not ages: return "n/a"
        return f"mean={sum(ages)/len(ages):.1f}  range={min(ages):.1f}-{max(ages):.1f}"

    lines.append(f"- FP sites: {by_site(fps)}")
    lines.append(f"- FN sites: {by_site(fns)}")
    lines.append(f"- FP gender: {by_gender(fps)}")
    lines.append(f"- FN gender: {by_gender(fns)}")
    lines.append(f"- FP age:    {age_stats(fps)}")
    lines.append(f"- FN age:    {age_stats(fns)}")

    fps_sorted = sorted(fps, key=lambda r: r.get("conf", 0) or 0, reverse=True)[:5]
    fns_sorted = sorted(fns, key=lambda r: r.get("conf", 0) or 0, reverse=True)[:5]
    lines.append("\n**Top-5 high-confidence FPs:**")
    for r in fps_sorted:
        sid = r["subject_id"]; ph = pheno.get(sid, {})
        lines.append(f"- `{sid}` (conf={r.get('conf')}, site={ph.get('site')}, "
                      f"age={ph.get('age')}, sex={'M' if ph.get('gender')==1 else 'F'})  "
                      f"reason: {r.get('extr_reason','')[:160]}")
    lines.append("\n**Top-5 high-confidence FNs:**")
    for r in fns_sorted:
        sid = r["subject_id"]; ph = pheno.get(sid, {})
        lines.append(f"- `{sid}` (conf={r.get('conf')}, site={ph.get('site')}, "
                      f"age={ph.get('age')}, sex={'M' if ph.get('gender')==1 else 'F'})  "
                      f"reason: {r.get('extr_reason','')[:160]}")

def analyze_simple(task: str, records: list[dict], lines: list[str]) -> None:
    fps = [r for r in records if r["true"] == 0 and r["pred"] == 1]
    fns = [r for r in records if r["true"] == 1 and r["pred"] == 0]
    invalids = [r for r in records if r.get("pred") is None]
    lines.append(f"\n### {task} ({len(records)} total: {len(fps)} FP, {len(fns)} FN, "
                  f"{len(invalids)} invalid)\n")
    for label, group in (("Top-5 FPs", fps), ("Top-5 FNs", fns)):
        sample = sorted(group, key=lambda r: r.get("conf", 0) or 0, reverse=True)[:5]
        lines.append(f"\n**{label}:**")
        for r in sample:
            lines.append(f"- `{r['subject_id']}` (conf={r.get('conf')}) "
                          f"reason: {r.get('extr_reason','')[:160]}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="study_postfix2")
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    study_dir = REPO_ROOT / "eval" / "results" / args.study
    if not study_dir.exists():
        sys.exit(f"Study folder not found: {study_dir}")

    pheno = load_pheno(REPO_ROOT / "eval" / "ground_truth" / "adhd_labels.csv")

    out_lines: list[str] = [f"# Failure-mode analysis {args.study}\n"]
    for model_dir in sorted(study_dir.iterdir()):
        if not model_dir.is_dir(): continue
        model_name = model_dir.name
        out_lines.append(f"\n## {model_name}\n")
        for jf in sorted(model_dir.glob("*_per_subject.jsonl")):
            task = jf.stem.replace("_per_subject", "")
            records = load_pred_jsonl(jf)
            if not records: continue
            if task == "adhd_binary":
                analyze_adhd(records, pheno, out_lines)
            else:
                analyze_simple(task, records, out_lines)

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "ablations" / "results" / f"failure_modes_{args.study}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
