
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score,
)
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GT_DIR   = PROJECT_ROOT / "eval" / "ground_truth"
FS_ROOT  = PROJECT_ROOT / "preprocessing" / "fastsurfer_output"
OUT_DIR  = PROJECT_ROOT / "baselines" / "results"

FASTSURFER_DIRS = {
    "adhd":   FS_ROOT / "ADHD-200",
    "tumor":  FS_ROOT / "BraTS-GLI",
    "stroke": FS_ROOT / "ATLAS-v2",
}

STATS_FILES = [
    "stats/aseg+DKT.VINN.stats",
    "stats/aseg.VINN.stats",
]

def _parse_volumes(stats_dir: Path) -> dict[str, float]:
    volumes: dict[str, float] = {}
    for rel_path in STATS_FILES:
        fpath = stats_dir / rel_path
        if not fpath.exists():
            continue
        for line in fpath.read_text(errors="replace").splitlines():
            if line.startswith("#"):
                if line.startswith("# Measure"):
                    parts = line.split(",")
                    if len(parts) >= 4:
                        try:
                            name = parts[0].split()[-1].strip()
                            vol  = float(parts[3].strip())
                            volumes[name] = vol
                        except (ValueError, IndexError):
                            pass
                continue
            cols = line.split()
            if len(cols) >= 5:
                try:
                    name = cols[4]
                    vol  = float(cols[3])
                    volumes[name] = vol
                except (ValueError, IndexError):
                    pass
    return volumes

def _asymmetry_summary(volumes: dict[str, float]) -> str:
    left_vols:  dict[str, float] = {}
    right_vols: dict[str, float] = {}
    for name, vol in volumes.items():
        lower = name.lower()
        if lower.startswith("left-") or lower.startswith("left_"):
            key = name[5:]
            left_vols[key] = vol
        elif lower.startswith("right-") or lower.startswith("right_"):
            key = name[6:]
            right_vols[key] = vol
        elif lower.startswith("lh-") or lower.startswith("lh_"):
            key = name[3:]
            left_vols[key] = vol
        elif lower.startswith("rh-") or lower.startswith("rh_"):
            key = name[3:]
            right_vols[key] = vol
        elif lower.startswith("ctx-lh-"):
            key = name[7:]
            left_vols[key] = vol
        elif lower.startswith("ctx-rh-"):
            key = name[7:]
            right_vols[key] = vol

    pairs = [(k, left_vols[k], right_vols[k])
             for k in left_vols if k in right_vols]
    if not pairs:
        return "No matched left/right pairs found."

    rows = []
    for structure, lv, rv in pairs:
        denom = lv + rv
        li    = (lv - rv) / denom if denom > 0 else 0.0
        rows.append((structure, lv, rv, li))

    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    top = rows[:20]

    lines = [
        "Left/Right Asymmetry (LI = (L-R)/(L+R); positive = left larger, negative = right larger):",
        f"{'Structure':<35} {'Left_mm3':>10} {'Right_mm3':>10} {'LI':>8}",
        "-" * 67,
    ]
    for structure, lv, rv, li in top:
        lines.append(f"{structure:<35} {lv:>10.0f} {rv:>10.0f} {li:>+8.3f}")

    total_l = sum(v for _, v, _, _ in rows)
    total_r = sum(v for _, _, v, _ in rows)
    total_li = (total_l - total_r) / (total_l + total_r) if (total_l + total_r) > 0 else 0
    lines.append("-" * 67)
    lines.append(f"{'TOTAL (all paired structures)':<35} {total_l:>10.0f} {total_r:>10.0f} {total_li:>+8.3f}")
    return "\n".join(lines)

_NORMATIVE: dict[str, dict] = {}

def _build_normative_buckets(controls_df: pd.DataFrame, fs_root: Path
                              ) -> tuple[dict[str, dict[str, float]], set[str]]:
    from collections import defaultdict
    buckets: dict[str, dict[str, float]] = defaultdict(dict)
    loaded: set[str] = set()
    for sid in controls_df["subject_id"]:
        subj_dir = fs_root / sid
        if not subj_dir.exists():
            continue
        for name, vol in _parse_volumes(subj_dir).items():
            buckets[name][sid] = vol
        loaded.add(sid)
    return dict(buckets), loaded

def _normative_from_buckets(buckets: dict[str, dict[str, float]],
                             exclude_sid: str | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, sid_to_vol in buckets.items():
        if exclude_sid and exclude_sid in sid_to_vol:
            vols = [v for s, v in sid_to_vol.items() if s != exclude_sid]
        else:
            vols = list(sid_to_vol.values())
        if len(vols) < 10:
            continue
        std = float(np.std(vols))
        if std <= 1:
            continue
        out[name] = {"mean": float(np.mean(vols)), "std": std}
    return out

def _build_normative_stats(controls_df: pd.DataFrame, fs_root: Path) -> dict[str, dict]:
    buckets, loaded = _build_normative_buckets(controls_df, fs_root)
    print(f"  [Normative] Built from {len(loaded)} controls, {len(buckets)} structures")
    return _normative_from_buckets(buckets)

ADHD_STRUCTURES = {
    "caudate", "putamen", "thalamus", "pallidum", "accumbens",
    "hippocampus", "amygdala", "frontal", "cingulate", "insula",
}

def _zscore_summary(volumes: dict[str, float], normative: dict[str, dict]) -> str:
    rows = []
    for name, vol in volumes.items():
        if not any(kw in name.lower() for kw in ADHD_STRUCTURES):
            continue
        if name not in normative:
            continue
        mean = normative[name]["mean"]
        std  = normative[name]["std"]
        z    = (vol - mean) / std
        rows.append((name, vol, mean, z))

    if not rows:
        return "No ADHD-relevant structures found in normative data."

    rows.sort(key=lambda x: x[3])
    lines = [
        "Z-scores vs age-matched controls (negative = smaller than control mean):",
        f"{'Structure':<40} {'Volume_mm3':>11} {'Ctrl_mean':>11} {'Z':>7}",
        "-" * 73,
    ]
    for name, vol, mean, z in rows[:25]:
        flag = " <<<" if z < -1.5 else (" >>" if z > 1.5 else "")
        lines.append(f"{name:<40} {vol:>11.0f} {mean:>11.0f} {z:>+7.2f}{flag}")
    return "\n".join(lines)

def _read_stats(stats_dir: Path, condition: str,
                normative: dict[str, dict] | None = None) -> str:
    volumes = _parse_volumes(stats_dir)
    if condition in ("tumor", "stroke"):
        return _asymmetry_summary(volumes)
    if normative:
        return _zscore_summary(volumes, normative)
    return "No normative data available."

def _make_adhd_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a neuroimaging expert. You are given a Z-score table for brain structures "
        "compared to age-matched controls from the same dataset.\n\n"
        "Z < -1.5 means the structure is significantly SMALLER than controls (marked <<<).\n"
        "Z > +1.5 means the structure is significantly LARGER than controls (marked >>).\n"
        "Z near 0 means the structure is within the normal range.\n\n"
        "Classify the subject:\n"
        "  0 = Typically developing control (most structures near z=0)\n"
        "  1 = ADHD (caudate, putamen, and/or thalamus z < -1.5)\n\n"
        "Base your decision ONLY on the z-scores shown. If no structures are z < -1.5, "
        "classify as control (0). Only classify as ADHD (1) if key subcortical structures "
        "(caudate, putamen, thalamus) are clearly reduced (z < -1.5).\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing specific z-scores\"}"
    )
    user = f"Z-score table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_tumor_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a neuro-oncology expert. A brain tumor is confirmed in this subject. "
        "You are given a pre-computed left/right asymmetry table from FastSurfer.\n\n"
        "LI = (Left - Right) / (Left + Right). "
        "Brain tumors ADD volume (FastSurfer counts tumor mass as tissue), so the hemisphere "
        "with the tumor will appear LARGER, not smaller.\n\n"
        "Determine which hemisphere contains the tumor:\n"
        "  0 = Left hemisphere tumor  (left structures are larger; positive TOTAL LI)\n"
        "  1 = Right hemisphere tumor (right structures are larger; negative TOTAL LI)\n\n"
        "Use the TOTAL LI at the bottom of the table as your primary signal. "
        "If TOTAL LI > 0, answer 0 (left). If TOTAL LI < 0, answer 1 (right).\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing the total LI value\"}"
    )
    user = f"Asymmetry table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_adhd_matched_prompt(features_text: str) -> list[dict]:
    system = (
        "You are a neuroimaging expert classifying an ADHD-200 subject.\n\n"
        "You are given the SAME features used by a trained leave-one-site-out "
        "logistic-regression classifier:\n"
        "  - age (years)\n"
        "  - sex_male (1=male, 0=female)\n"
        "  - 16 ICV-corrected subcortical volume z-scores vs same-site controls\n"
        "  - 7 fMRI seed-based connectivity z-scores vs same-site controls\n\n"
        "z-score sign and magnitude:\n"
        "  z near 0 = within control range\n"
        "  |z| >= 1.5 = 5th/95th percentile deviation (chance rate ~13% in controls)\n"
        "  ADHD signal is small (Hoogman 2017 d = -0.11 to -0.19).\n\n"
        "Demographics: ~26% prevalence overall (38% male, 12% female).\n\n"
        "Classify:\n"
        "  0 = typically developing control\n"
        "  1 = ADHD\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing specific z-scores and demographics\"}"
    )
    user = f"Features:\n\n{features_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_tumor_grade_matched_prompt(features_text: str) -> list[dict]:
    system = (
        "You are a neuro-oncology expert classifying a glioma's WHO grade.\n\n"
        "You are given FastSurfer-derived structural features only, NO "
        "tumor segmentation mask. The grade signal is INDIRECT: HGG tumors "
        "(grade IV / GBM) tend to be larger and produce greater mass "
        "effect (asymmetry, ventricular compression, displaced subcortical "
        "structures) than LGG (grade II-III).\n\n"
        "Feature naming:\n"
        "  <Structure>            : ICV-corrected volume (per-million ICV)\n"
        "  LI_<Structure>          : (L-R)/(L+R) lateralization index\n"
        "  abs_LI_<Structure>      : magnitude of asymmetry\n"
        "  TOTAL_LI                : whole-brain hemispheric asymmetry\n"
        "  abs_TOTAL_LI            : magnitude of total asymmetry\n"
        "  abs_LI_ventricle        : ventricular asymmetry magnitude\n"
        "  total_ventricle_per_icv : combined ventricular volume per million ICV\n\n"
        "Predict:\n"
        "  0 = LGG (low-grade glioma, less mass effect)\n"
        "  1 = HGG (high-grade glioma / GBM, more mass effect, larger volume)\n\n"
        "The signal is weak (FastSurfer only), chance-level performance "
        "is expected if no clear asymmetry is present.\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing specific features\"}"
    )
    user = f"FastSurfer features:\n\n{features_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_stroke_lat_matched_prompt(features_text: str) -> list[dict]:
    system = (
        "You are a stroke neurologist classifying lesion lateralization.\n\n"
        "You are given ONLY FastSurfer-derived hemispheric volume "
        "lateralization indices, NO lesion mask. Strokes cause "
        "ipsilateral tissue loss, so the affected hemisphere appears "
        "SMALLER on FastSurfer.\n\n"
        "TOTAL_LI = (L_volume - R_volume) / (L_volume + R_volume)\n"
        "  TOTAL_LI < 0  =>  LEFT smaller  =>  LEFT-hemisphere stroke (label 0)\n"
        "  TOTAL_LI > 0  =>  RIGHT smaller =>  RIGHT-hemisphere stroke (label 1)\n\n"
        "Per-structure LI_<Name> values can corroborate. Use the sign of "
        "TOTAL_LI as the primary signal.\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing TOTAL_LI value\"}"
    )
    user = f"FastSurfer LIs:\n\n{features_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_tumor_grade_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a neuro-oncology expert. A brain tumor is confirmed in this subject. "
        "You are given a left/right asymmetry table from FastSurfer.\n\n"
        "Predict the glioma's WHO grade:\n"
        "  0 = LOW-GRADE GLIOMA (LGG, WHO grade II-III), typically smaller, less mass effect\n"
        "  1 = HIGH-GRADE GLIOMA (HGG / glioblastoma, WHO grade IV), larger volume, "
        "    enhancing tumor + necrosis, more mass effect (asymmetry, ventricular compression)\n\n"
        "Cues from FastSurfer alone are weak, the segmentation mask volumes (which you "
        "do not have access to) carry most of the grade signal. Use the magnitude of "
        "asymmetry and ventricular compression as your primary signals.\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence\"}"
    )
    user = f"Asymmetry table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_stroke_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a stroke neurologist. A stroke lesion is confirmed in this subject. "
        "You are given a pre-computed left/right asymmetry table from FastSurfer.\n\n"
        "LI = (Left - Right) / (Left + Right). "
        "Strokes cause tissue LOSS, so the hemisphere with the stroke will appear SMALLER.\n\n"
        "Determine which hemisphere contains the stroke lesion:\n"
        "  0 = Left hemisphere stroke  (left structures are smaller; negative TOTAL LI)\n"
        "  1 = Right hemisphere stroke (right structures are smaller; positive TOTAL LI)\n\n"
        "Use the TOTAL LI at the bottom of the table as your primary signal. "
        "If TOTAL LI < 0, answer 0 (left). If TOTAL LI > 0, answer 1 (right).\n\n"
        "Respond with ONLY a JSON object: {\"label\": 0 or 1, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"one sentence citing the total LI value\"}"
    )
    user = f"Asymmetry table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

PROMPT_BUILDERS = {
    "adhd":   _make_adhd_prompt,
    "tumor":  _make_tumor_prompt,
    "stroke": _make_stroke_prompt,
}

LAT_LABEL_MAP = {"left": 0, "right": 1}

def _get_true_label(row, condition: str) -> int | None:
    if condition == "adhd":
        return int(row.label)
    elif condition == "tumor":
        lat = str(row.lateralization) if pd.notna(getattr(row, "lateralization", None)) else None
        return LAT_LABEL_MAP.get(lat, None)
    elif condition == "stroke":
        lat = str(row.lateralization) if pd.notna(getattr(row, "lateralization", None)) else None
        return LAT_LABEL_MAP.get(lat, None)
    return None

def _parse_response(content: str) -> tuple[int | None, float]:
    cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()
    try:
        obj = json.loads(cleaned)
        label = int(obj.get("label", -1))
        conf  = float(obj.get("confidence", 0.5))
        if label in (0, 1):
            return label, conf
    except Exception:
        pass
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            label = int(obj.get("label", -1))
            conf  = float(obj.get("confidence", 0.5))
            if label in (0, 1):
                return label, conf
        except Exception:
            pass
    m = re.search(r'"label"\s*:\s*([01])', content)
    if m:
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
        return int(m.group(1)), float(c.group(1)) if c else 0.5
    return None, 0.5

def run_baseline(condition: str, max_subjects: int | None, model: str,
                 out_dir: Path, sleep_between: float = 0.5,
                 debug: bool = False) -> dict:
    client = OpenAI()
    fs_dir = FASTSURFER_DIRS[condition]

    label_files = {
        "adhd":   GT_DIR / "adhd_labels.csv",
        "tumor":  GT_DIR / "tumor_labels.csv",
        "stroke": GT_DIR / "stroke_labels.csv",
    }
    if not label_files[condition].exists():
        print(f"GT file not found: {label_files[condition]}. Run extract_ground_truth.py first.")
        return {}
    labels = pd.read_csv(label_files[condition])

    if condition in ("stroke", "tumor"):
        labels = labels[labels["lateralization"].isin(["left", "right"])]

    normative: dict[str, dict] = {}
    if condition == "adhd":
        controls = labels[labels["label"] == 0]
        print(f"  Building normative stats from {len(controls)} control subjects...")
        normative = _build_normative_stats(controls, fs_dir)

    if max_subjects:
        labels = labels.head(max_subjects)

    prompt_builder = PROMPT_BUILDERS[condition]
    y_true, y_pred_label, y_pred_prob = [], [], []
    failed = []

    for i, row in enumerate(labels.itertuples(), 1):
        sid  = row.subject_id
        true_label = _get_true_label(row, condition)
        if true_label is None:
            continue

        subj_dir = fs_dir / sid
        if not subj_dir.exists():
            print(f"  [{i}/{len(labels)}] {sid} SKIP (no stats dir)")
            continue

        stats_text = _read_stats(subj_dir, condition, normative if condition == "adhd" else None)
        messages   = prompt_builder(stats_text)

        print(f"  [{i}/{len(labels)}] {sid}...", end=" ", flush=True)
        try:
            resp  = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=150,
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            if debug:
                print(f"\n    [RAW] {content!r}")
            pred, conf = _parse_response(content)
            if pred is None:
                print(f"PARSE FAIL: {content[:120]}")
                failed.append(sid)
                continue
            y_true.append(true_label)
            y_pred_label.append(pred)
            y_pred_prob.append([1 - conf, conf] if pred == 1 else [conf, 1 - conf])
            print(f"true={true_label} pred={pred} conf={conf:.2f}")
        except Exception as e:
            print(f"API ERROR: {e}")
            failed.append(sid)
        if sleep_between:
            time.sleep(sleep_between)

    if not y_true:
        return {"error": "No predictions produced"}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_label)
    y_prob = np.array(y_pred_prob)
    result = {
        "n": len(y_true),
        "n_failed": len(failed),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        try:
            result["auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
        except Exception:
            pass
    return result

def main():
    parser = argparse.ArgumentParser(description="Direct LLM baseline for NeuroAgent comparison")
    parser.add_argument("--condition",    required=True, choices=["adhd", "tumor", "stroke", "all"])
    parser.add_argument("--model",        default="gpt-4o")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-dir",      default=str(OUT_DIR))
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = ["adhd"] if args.condition == "all" else [args.condition]
    if args.condition == "stroke":
        print("Stroke lateralization is all-left in our ATLAS-v2 subset. Skipping.")
        return
    if args.condition == "tumor":
        print("Tumor grade requires T1c contrast (not in FastSurfer stats). "
              "LLM will see asymmetry table only.")
    all_results = {}

    for cond in conds:
        print(f"\nDirect LLM baseline: {cond.upper()} (model={args.model})")
        result = run_baseline(cond, args.max_subjects, args.model, out_dir,
                              debug=args.debug)
        all_results[cond] = result
        print(json.dumps(result, indent=2))

        out_path = out_dir / f"{cond}_llm_baseline_{args.model.replace('/','_')}_results.json"
        out_path.write_text(json.dumps({"condition": cond, "model": args.model, **result}, indent=2))
        print(f"Saved: {out_path}")

    if len(conds) > 1:
        all_path = out_dir / "all_llm_baseline_results.json"
        all_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nAll results saved: {all_path}")

if __name__ == "__main__":
    main()
