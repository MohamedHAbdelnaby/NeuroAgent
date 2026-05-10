from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)
from openai import OpenAI
from openai import (
    RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
)

_RETRYABLE_API_ERRORS = (
    RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.direct_llm_baseline import (
    _parse_response, _read_stats, _build_normative_stats,
    _build_normative_buckets, _normative_from_buckets,
    _make_adhd_prompt, _make_tumor_prompt, _make_tumor_grade_prompt, _get_true_label, FASTSURFER_DIRS,
    _make_adhd_matched_prompt, _make_tumor_grade_matched_prompt, _make_stroke_lat_matched_prompt,
)

FMRIPREP_ROOT = PROJECT_ROOT / "preprocessing" / "fmriprep_output"

def _resolve_fmri_paths(subject_id: str) -> tuple[str | None, str | None]:
    if "_sub-" not in subject_id:
        return None, None
    site, sub = subject_id.split("_", 1)
    sub_dir = FMRIPREP_ROOT / site / sub
    if not sub_dir.exists():
        return None, None
    bolds = sorted(sub_dir.glob(
        "ses-*/func/*MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))
    if not bolds:
        return None, None
    bold = bolds[0]
    base = bold.name.replace(
        "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz", "")
    conf = bold.parent / f"{base}_desc-confounds_timeseries.tsv"
    return str(bold), (str(conf) if conf.exists() else None)

GT_DIR        = PROJECT_ROOT / "eval" / "ground_truth"
                                                                               
_STUDY_NAME   = os.environ.get("STUDY_NAME", "study")
STUDY_DIR     = PROJECT_ROOT / "eval" / "results" / _STUDY_NAME
VLLM_ENDPOINTS_DIR = PROJECT_ROOT / "baselines" / "vllm_endpoints"

ARC_API_MODELS = [
    "gpt-oss-120b",
    "Kimi-K2.6",
    "kimi-k26-non-thinking",
    "MiniMax-M2.7",
]

VLLM_HOSTED_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-14B",
    "Mistral-7B-v0.3",
    "gpt-oss-20b",
    "Gemma-3-12b-it",
]

DEFAULT_MODELS = ["gpt-oss-120b"] + VLLM_HOSTED_MODELS
DEFAULT_TASKS  = ["adhd_binary", "tumor_grade", "stroke_lat"]

DEFAULT_MODES = ["direct", "direct_matched", "neuroagent", "neuroagent_llm"]

def _make_stroke_detection_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a neuroimaging researcher working on an academic brain morphology "
        "study. You are given a pre-computed left/right asymmetry table from FastSurfer "
        "for one subject from a public research dataset.\n\n"
        "LI = (Left - Right) / (Left + Right). |LI| near 0 means the brain is symmetric. "
        "Chronic stroke causes tissue LOSS on the affected side, producing measurable "
        "hemispheric asymmetry. On this dataset, control subjects have |TOTAL LI| ≈ 0.007 "
        "(median), while stroke subjects have |TOTAL LI| ≈ 0.016 (median, with a long "
        "right tail up to ~0.13).\n\n"
        "Classify:\n"
        "  0 = No stroke (brain is roughly symmetric, |TOTAL LI| < 0.012 AND no single "
        "structure with extreme |LI|)\n"
        "  1 = Stroke present (|TOTAL LI| >= 0.012 OR a clearly asymmetric subcortical "
        "structure consistent with focal lesion)\n\n"
        "Use BOTH the TOTAL LI at the bottom of the table AND the top-asymmetric "
        "structures. Small focal strokes can leave TOTAL LI low but produce one outlier.\n\n"
        "Respond with ONLY valid JSON: "
        "{\"label\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}"
    )
    user = f"Asymmetry table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _make_stroke_lat_prompt(stats_text: str) -> list[dict]:
    system = (
        "You are a stroke neurologist. A chronic stroke lesion is confirmed in this subject. "
        "You are given a pre-computed left/right asymmetry table from FastSurfer.\n\n"
        "LI = (Left - Right) / (Left + Right). Stroke causes tissue LOSS on the lesioned "
        "side, so the side with the lesion will appear SMALLER (lower volume).\n\n"
        "  0 = LEFT hemisphere stroke (left structures are smaller, TOTAL LI is negative)\n"
        "  1 = RIGHT hemisphere stroke (right structures are smaller, TOTAL LI is positive)\n\n"
        "Use the sign of TOTAL LI as the primary cue. If TOTAL LI < 0, answer 0 (left). "
        "If TOTAL LI > 0, answer 1 (right). Top-asymmetric structures can confirm.\n\n"
        "Respond with ONLY valid JSON: "
        "{\"label\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"one sentence citing the TOTAL LI value\"}"
    )
    user = f"Asymmetry table:\n\n{stats_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _build_matched_features_text(task: str, sid: str, fs_path: str,
                                   seg_p: str | None,
                                   lesion_p: str | None,
                                   bold_p: str | None,
                                   conf_p: str | None) -> str:
    from agents.structural_mri_agent import StructuralMRIAgent
    from agents.functional_fmri_agent import FunctionalFMRIAgent
    cond = ("adhd" if task == "adhd_binary"
             else "tumor" if task in ("tumor_grade", "tumor_lat")
             else "stroke")

    sa = StructuralMRIAgent(condition=cond)
    s_out = sa.run(sid, fs_path, seg_path=seg_p, lesion_mask_path=lesion_p)
    sm = s_out.global_metrics or {}

    if cond == "tumor" and task == "tumor_grade":
                                                                         
        feats = sm.get("tumor_features") or {}
        if not feats:
            return ""
        lines = ["  FastSurfer-derived features (volumes scaled per-million ICV; LIs are dimensionless):"]
        keys = sorted(feats.keys())
        for k in keys:
            lines.append(f"    {k:42s} = {feats[k]:+.4f}")
        return "\n".join(lines)
    if cond == "stroke" and task == "stroke_lat":
                                                          
        total_li = sm.get("total_li")
        lines = []
        if total_li is not None:
            lines.append(f"  TOTAL_LI = {total_li:+.4f}   "
                         f"(L-R)/(L+R); negative => LEFT smaller => LEFT stroke (label 0)")
        for k, v in sorted(sm.items()):
            if k.startswith("LI_") and isinstance(v, (int, float)):
                lines.append(f"  {k:30s} = {v:+.4f}")
        return "\n".join(lines) if lines else ""
    if cond == "adhd":
                                                                            
        from agents.orchestrator import _load_adhd_pheno
        ph = _load_adhd_pheno().get(sid, {})
        age = ph.get("age"); gender = ph.get("gender")
        sex_str = "male" if gender == 1 else "female" if gender == 0 else "?"
        lines: list[str] = [
            f"  age:       {age}",
            f"  sex:       {sex_str}",
            "",
            "  Subcortical ICV-corrected volume z-scores vs same-site controls:",
        ]
        STRUCTS = [
            "Left-Caudate", "Right-Caudate", "Left-Putamen", "Right-Putamen",
            "Left-Pallidum", "Right-Pallidum", "Left-Hippocampus", "Right-Hippocampus",
            "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area", "Right-Accumbens-area",
            "Left-Thalamus", "Right-Thalamus",
            "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
        ]
        for s in STRUCTS:
            v = sm.get(f"z_{s}")
            lines.append(f"    z_{s:30s} = {v:+.2f}" if isinstance(v, (int, float))
                          else f"    z_{s:30s} = n/a")

        if bold_p:
            try:
                fa = FunctionalFMRIAgent(condition="adhd")
                f_out = fa.run(sid, bold_p, conf_p)
                fm = f_out.global_metrics or {}
                lines.append("")
                lines.append("  fMRI seed-based connectivity z-scores vs same-site controls:")
                PAIRS = ["PCC_mPFC", "PCC_Precuneus", "ACC_R.INS",
                          "L.DLPFC_R.DLPFC", "L.DLPFC_L.CAU", "R.DLPFC_R.CAU",
                          "R.IFG_R.CAU"]
                for p in PAIRS:
                    v = fm.get(f"z_{p}")
                    lines.append(f"    z_{p:18s} = {v:+.2f}" if isinstance(v, (int, float))
                                  else f"    z_{p:18s} = n/a")
            except Exception:
                lines.append("  fMRI: extraction failed (using structural + demographics only)")
        else:
            lines.append("")
            lines.append("  fMRI: not available for this subject")
        return "\n".join(lines)
    return ""

def _extract_label_from_orchestrator(result, task: str) -> tuple[int | None, float]:
    conf = float(getattr(result, "confidence", 0.5) or 0.5)
    if hasattr(result, "predicted_label") and result.predicted_label in (0, 1):
        return int(result.predicted_label), conf
    return None, conf

def _resolve_client_for_model(model: str) -> OpenAI:
    ep_file = VLLM_ENDPOINTS_DIR / f"{model}.json"
    if ep_file.exists():
        ep = json.loads(ep_file.read_text())
        return OpenAI(base_url=ep["url"], api_key=ep.get("api_key", "local-vllm-key"))
    return OpenAI()

def compute_all_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "n_samples":         int(len(y_true)),
        "n_positive":        int((y_true == 1).sum()),
        "n_negative":        int((y_true == 0).sum()),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_macro":   round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro":      round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro":          round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_positive":       round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_negative":       round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "precision_positive":round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_positive":   round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_negative":round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_negative":   round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
    }

    metrics["sensitivity"] = metrics["recall_positive"]
    metrics["specificity"] = metrics["recall_negative"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"]   = int(cm[0, 0])
    metrics["false_positives"]  = int(cm[0, 1])
    metrics["false_negatives"]  = int(cm[1, 0])
    metrics["true_positives"]   = int(cm[1, 1])

    if len(np.unique(y_true)) > 1:
        if y_prob is not None:
            try:
                prob_pos = np.asarray(y_prob)[:, 1] if np.ndim(y_prob) == 2 else np.asarray(y_prob)
                metrics["auc_roc"]     = round(float(roc_auc_score(y_true, prob_pos)), 4)
                metrics["pr_auc"]      = round(float(average_precision_score(y_true, prob_pos)), 4)
            except Exception:
                metrics["auc_roc"] = None
                metrics["pr_auc"]  = None
        else:
            try:
                metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_pred)), 4)
            except Exception:
                pass

    return metrics

def run_experiment(model: str, task: str, max_subjects: int,
                   resume: bool, sleep: float = 0.4,
                   mode: str = "direct", parallel: int = 1,
                   balanced: bool = False) -> dict:

    folder_suffix = {
        "direct":           "",
        "direct_matched":   "__direct_matched",
        "neuroagent":       "__neuroagent",
        "neuroagent_llm":   "__neuroagent_llm",
        "rule_only":        "__rule_only",
    }.get(mode, f"__{mode}")
    folder_name = model + folder_suffix
    out_dir = STUDY_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject_log = out_dir / f"{task}_per_subject.jsonl"
    metrics_path    = out_dir / f"{task}_results.json"

    if task == "adhd_binary":
        condition = "adhd"
        full_labels = pd.read_csv(GT_DIR / "adhd_labels.csv")
        full_labels["fs_path"] = full_labels["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["adhd"] / s))
        prompt_fn = _make_adhd_prompt
    elif task == "tumor_lat":
        condition = "tumor"
        full_labels = pd.read_csv(GT_DIR / "tumor_labels.csv")
        full_labels = full_labels[full_labels["lateralization"].isin(["left", "right"])]
        full_labels["label"] = (full_labels["lateralization"] == "right").astype(int)
        full_labels["fs_path"] = full_labels["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["tumor"] / s))
        prompt_fn = _make_tumor_prompt
    elif task == "tumor_grade":
                                                                              
        condition = "tumor"
        full_labels = pd.read_csv(GT_DIR / "tumor_labels.csv")
        full_labels = full_labels.dropna(subset=["grade_label"]).copy()
        full_labels["label"] = full_labels["grade_label"].astype(int)
        full_labels["fs_path"] = full_labels["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["tumor"] / s))
        prompt_fn = _make_tumor_grade_prompt
    elif task == "stroke_lat":
        condition = "stroke"
        full_labels = pd.read_csv(GT_DIR / "stroke_labels.csv")
        full_labels = full_labels[full_labels["lateralization"].isin(["left", "right"])]
        full_labels["label"] = (full_labels["lateralization"] == "right").astype(int)
        full_labels["fs_path"] = full_labels["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["stroke"] / s))
        prompt_fn = _make_stroke_lat_prompt
    elif task == "stroke_detection":
        condition = "stroke"
        stroke_df = pd.read_csv(GT_DIR / "stroke_labels.csv")
        stroke_df = stroke_df.dropna(subset=["lateralization"]).copy()
        stroke_df["label"]   = 1
        stroke_df["fs_path"] = stroke_df["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["stroke"] / s))
        ctrl_df = pd.read_csv(GT_DIR / "adhd_labels.csv")
        ctrl_df = ctrl_df[ctrl_df["label"] == 0].copy()
        ctrl_df["fs_path"] = ctrl_df["subject_id"].apply(
            lambda s: str(FASTSURFER_DIRS["adhd"] / s))
        common_cols = ["subject_id", "label", "fs_path"]
        full_labels = pd.concat([stroke_df[common_cols], ctrl_df[common_cols]],
                                 ignore_index=True)
        full_labels = full_labels.sample(frac=1, random_state=42).reset_index(drop=True)
        prompt_fn = _make_stroke_detection_prompt
    else:
        raise ValueError(f"Unknown task: {task}")

    client = _resolve_client_for_model(model)

    if mode == "neuroagent_llm":
        os.environ["NEUROAGENT_LLM_DECIDES"] = "1"
    elif mode == "neuroagent":
        os.environ.pop("NEUROAGENT_LLM_DECIDES", None)

    matched_prompt_fn = None
    if task == "adhd_binary":
        matched_prompt_fn = _make_adhd_matched_prompt
    elif task == "tumor_grade":
        matched_prompt_fn = _make_tumor_grade_matched_prompt
    elif task == "stroke_lat":
        matched_prompt_fn = _make_stroke_lat_matched_prompt

    orch = None
    if mode in ("neuroagent", "neuroagent_llm"):
        from agents.orchestrator import NeuroAgentOrchestrator
        ep_file = VLLM_ENDPOINTS_DIR / f"{model}.json"
        if ep_file.exists():
            ep = json.loads(ep_file.read_text())
            orch = NeuroAgentOrchestrator(condition=condition, model=model,
                                           base_url=ep["url"],
                                           api_key=ep.get("api_key", "local-vllm-key"))
        else:
            orch = NeuroAgentOrchestrator(condition=condition, model=model)

    normative = {}
    normative_buckets: dict = {}
    normative_subjects: set = set()
    if condition == "adhd":
        controls = full_labels[full_labels["label"] == 0]
        print(f"  [{model}/{task}] Building normative buckets from {len(controls)} controls...")
        normative_buckets, normative_subjects = _build_normative_buckets(
            controls, FASTSURFER_DIRS["adhd"])
                                                                                      
        normative = _normative_from_buckets(normative_buckets)
        print(f"  [{model}/{task}] Loaded {len(normative_subjects)} controls, "
              f"{len(normative)} structures pass min-N+std filter")

    if balanced and max_subjects and "label" in full_labels.columns:
        n_per_class = max_subjects // 2
        per_class = []
        for lbl, grp in full_labels.groupby("label"):
            per_class.append(grp.sample(min(len(grp), n_per_class), random_state=42))
        labels = (pd.concat(per_class, ignore_index=True)
                  .sample(frac=1, random_state=42)
                  .reset_index(drop=True))
        print(f"  [{model}/{task}] Balanced subset: {labels['label'].value_counts().to_dict()}")
    else:
        labels = full_labels.head(max_subjects) if max_subjects else full_labels

    seen: set[str] = set()
    cached_preds = []
    if resume and per_subject_log.exists():
        n_kept, n_dropped = 0, 0
        kept_lines: list[str] = []
        for line in per_subject_log.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "extr_reason" not in d:
                n_dropped += 1
                continue
            seen.add(d["subject_id"])
            cached_preds.append(d)
            kept_lines.append(line)
            n_kept += 1
                                                                             
        if n_dropped > 0:
            per_subject_log.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            print(f"  [{model}/{task}] Resume: kept {n_kept} post-fix lines, dropped {n_dropped} pre-fix lines (will re-run)")
        else:
            print(f"  [{model}/{task}] Resuming from {n_kept} cached predictions")

    rows = list(cached_preds)
    log_fp = open(per_subject_log, "a")

    if mode in ("neuroagent", "neuroagent_llm") and parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from agents.orchestrator import NeuroAgentOrchestrator
        import threading
        log_lock = threading.Lock()

        def _process(row):
            sid = row.subject_id
            if sid in seen:
                return None
            tl = int(row.label) if (hasattr(row, "label") and row.label in (0, 1)) \
                                 else _get_true_label(row, condition)
            if tl is None:
                return None
            sd = Path(getattr(row, "fs_path", "")) if hasattr(row, "fs_path") else None
            if sd is None or not sd.exists():
                return None

            ep_file = VLLM_ENDPOINTS_DIR / f"{model}.json"
            if ep_file.exists():
                ep = json.loads(ep_file.read_text())
                local_orch = NeuroAgentOrchestrator(
                    condition=condition, model=model,
                    base_url=ep["url"],
                    api_key=ep.get("api_key", "local-vllm-key"))
            else:
                local_orch = NeuroAgentOrchestrator(condition=condition, model=model)

            t0 = time.time()
            bold_p, conf_p = (_resolve_fmri_paths(sid)
                              if condition == "adhd" else (None, None))
                                                                             
            seg_p = None
            lesion_p = None
            try:
                result = local_orch.run(
                    subject_id=sid, stats_dir=sd, output_dir=None,
                    bold_path=bold_p, confounds_path=conf_p,
                    seg_path=seg_p,
                    lesion_mask_path=lesion_p,
                    task=task,
                )
                pred, conf = _extract_label_from_orchestrator(result, task)
                u = local_orch.last_usage
                p_tok = int(u.get("prompt_tokens", 0))
                c_tok = int(u.get("completion_tokens", 0))
                n_call = int(u.get("n_calls", 0))
                tokens = p_tok + c_tok
                elapsed = time.time() - t0
                summary_excerpt = (getattr(result, "diagnostic_summary", "") or "")[:1500]
                extr_reason     = getattr(result, "extraction_reason", "")
                base = {"subject_id": sid, "true": tl, "pred": pred,
                        "conf": (round(float(conf), 3) if pred is not None else conf),
                        "elapsed_s": round(elapsed, 1), "tokens": tokens,
                        "prompt_tokens": p_tok, "completion_tokens": c_tok,
                        "n_calls": n_call,
                        "summary": summary_excerpt, "extr_reason": extr_reason}
                if pred is not None:
                    base["true"] = int(tl); base["pred"] = int(pred)
                rd = base
                with log_lock:
                    log_fp.write(json.dumps(rd) + "\n"); log_fp.flush()
                return rd
            except Exception as e:
                print(f"  {sid} ERROR: {type(e).__name__}: {e}")
                return None

        pending = [r for r in labels.itertuples() if r.subject_id not in seen]
        print(f"  [{model}/{task}] PARALLEL: {len(pending)} subjects with {parallel} workers")
        done_count = 0
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(_process, row): row.subject_id for row in pending}
            for fut in as_completed(futs):
                rd = fut.result()
                done_count += 1
                if rd:
                    rows.append(rd)
                    if done_count % 10 == 0 or done_count <= 5:
                        print(f"  [{done_count}/{len(pending)}] {rd.get('subject_id')} "
                              f"true={rd.get('true')} pred={rd.get('pred')} "
                              f"({rd.get('elapsed_s')}s)")
                                        
        log_fp.close()
        log_fp = None

    for i, row in enumerate(labels.itertuples(), 1):
        if log_fp is None:
            break                                               
        sid = row.subject_id
        if sid in seen:
            continue
                                                                           
        if hasattr(row, "label") and row.label in (0, 1):
            true_label = int(row.label)
        else:
            true_label = _get_true_label(row, condition)
        if true_label is None:
            continue

        subj_dir = Path(getattr(row, "fs_path", "")) if hasattr(row, "fs_path") else None
        if subj_dir is None or not subj_dir.exists():
            continue

        if mode == "direct":
                                                                               
            read_cond = "tumor" if task in ("stroke_detection", "stroke_lat") else condition
                                                                                    
            if condition == "adhd" and sid in normative_subjects:
                local_normative = _normative_from_buckets(
                    normative_buckets, exclude_sid=sid)
            else:
                local_normative = normative
            stats_text = _read_stats(subj_dir, read_cond,
                                     local_normative if condition == "adhd" else None)
            if not stats_text or "No " in stats_text[:25]:
                continue
            messages = prompt_fn(stats_text)
        elif mode == "direct_matched" and matched_prompt_fn is not None:
                                                                     
            seg_p_local    = None
            lesion_p_local = None
            bold_p_local, conf_p_local = (_resolve_fmri_paths(sid)
                                            if condition == "adhd" else (None, None))
            try:
                features_text = _build_matched_features_text(
                    task, sid, str(subj_dir), seg_p_local, lesion_p_local,
                    bold_p_local, conf_p_local,
                )
            except Exception as e:
                print(f"  {sid} matched features error: {type(e).__name__}: {e}")
                continue
            if not features_text:
                continue
            messages = matched_prompt_fn(features_text)
        elif mode == "rule_only":
                                                                        
            from agents.structural_mri_agent import StructuralMRIAgent
            from agents.functional_fmri_agent import FunctionalFMRIAgent
            from agents.orchestrator import NeuroAgentOrchestrator as _NA
            seg_p_local    = None                                    
            lesion_p_local = None
            bold_p_local, conf_p_local = (_resolve_fmri_paths(sid)
                                            if condition == "adhd" else (None, None))
            sa = StructuralMRIAgent(condition=condition)
            s_out = sa.run(sid, str(subj_dir), seg_path=seg_p_local,
                            lesion_mask_path=lesion_p_local)
            sm = dict(s_out.global_metrics or {}); sm["_subject_id"] = sid
            fm: dict = {"_subject_id": sid}
            if condition == "adhd" and bold_p_local:
                try:
                    fa = FunctionalFMRIAgent(condition="adhd")
                    f_out = fa.run(sid, bold_p_local, conf_p_local)
                    fm.update(dict(f_out.global_metrics or {}))
                    fm["_subject_id"] = sid
                except Exception:
                    pass
            t0 = time.time()
            label, reason = _NA._compute_decision(task, sm, fm)
            elapsed = time.time() - t0
            row_d = {
                "subject_id": sid, "true": int(true_label),
                "pred": (int(label) if label in (0, 1) else None),
                "conf": 1.0 if label in (0, 1) else 0.0,
                "elapsed_s": round(elapsed, 3), "tokens": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0,
                "summary": "", "extr_reason": f"rule_only: {reason}",
            }
            log_fp.write(json.dumps(row_d) + "\n"); log_fp.flush()
            rows.append(row_d)
            continue

        t0 = time.time()
        try:
            if mode in ("direct", "direct_matched"):
                                                                  
                max_tok = (2500 if "kimi" in model.lower() and "non-thinking" not in model.lower()
                           else 800)
                                                                                  
                resp = None
                for retry in range(8):
                    try:
                        resp = client.chat.completions.create(
                            model=model, messages=messages,
                            max_tokens=max_tok, temperature=0,
                        )
                    except _RETRYABLE_API_ERRORS as e:
                        resp = None
                        backoff = 60 * (1 + retry)
                        print(f"  [{i}/{len(labels)}] {sid} {type(e).__name__} - sleeping {backoff}s...")
                        time.sleep(backoff)
                        continue
                    if resp is not None and getattr(resp, "choices", None):
                        break
                                                                                                
                    backoff = 60 * (1 + retry)
                    print(f"  [{i}/{len(labels)}] {sid} null response - sleeping {backoff}s...")
                    time.sleep(backoff)
                if resp is None or not getattr(resp, "choices", None):
                    raise RuntimeError("API returned null after 8 retries - likely rate-limited or down")
                msg     = resp.choices[0].message
                content = msg.content or getattr(msg, "reasoning_content", "") or ""
                pred, conf = _parse_response(content)
                if resp.usage:
                    p_tok  = int(getattr(resp.usage, "prompt_tokens", 0))
                    c_tok  = int(getattr(resp.usage, "completion_tokens", 0))
                    tokens = int(getattr(resp.usage, "total_tokens", p_tok + c_tok))
                else:
                    p_tok = c_tok = tokens = 0
                n_call = 1
                raw_excerpt = content[:200]
            else:
                bold_p, conf_p = (_resolve_fmri_paths(sid)
                                  if condition == "adhd" else (None, None))
                seg_p = None                                         
                lesion_p = None
                result = orch.run(
                    subject_id=sid, stats_dir=subj_dir, output_dir=None,
                    bold_path=bold_p, confounds_path=conf_p,
                    seg_path=seg_p,
                    lesion_mask_path=lesion_p,
                    task=task,
                )
                pred, conf = _extract_label_from_orchestrator(result, task)
                u = orch.last_usage
                p_tok = int(u.get("prompt_tokens", 0))
                c_tok = int(u.get("completion_tokens", 0))
                n_call = int(u.get("n_calls", 0))
                tokens = p_tok + c_tok
                raw_excerpt = (getattr(result, "diagnostic_summary", "") or "")[:1500]
                _extr_reason = getattr(result, "extraction_reason", "")

            elapsed = time.time() - t0
            extr_reason = locals().get("_extr_reason", "direct_v2")
            row_d = {"subject_id": sid, "true": int(true_label),
                     "pred": (int(pred) if pred is not None else None),
                     "conf": (round(float(conf), 3) if pred is not None else conf),
                     "elapsed_s": round(elapsed, 1), "tokens": tokens,
                     "prompt_tokens": p_tok, "completion_tokens": c_tok,
                     "n_calls": n_call,
                     "summary": raw_excerpt, "extr_reason": extr_reason}
            if pred is None:
                row_d["raw"] = raw_excerpt
                log_fp.write(json.dumps(row_d) + "\n"); log_fp.flush()
                rows.append(row_d)
                print(f"  [{i}/{len(labels)}] {sid} PARSE_FAIL")
                continue
            log_fp.write(json.dumps(row_d) + "\n"); log_fp.flush()
            rows.append(row_d)
            print(f"  [{i}/{len(labels)}] {sid} true={true_label} pred={pred} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i}/{len(labels)}] {sid} ERROR: {type(e).__name__}: {e}")
        time.sleep(sleep)

    if log_fp is not None:
        log_fp.close()

    valid = [r for r in rows if r.get("pred") in (0, 1)]
    if not valid:
        result = {"model": model, "task": task, "error": "no valid predictions"}
        metrics_path.write_text(json.dumps(result, indent=2))
        return result

    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]
    y_prob = []
    for r in valid:
        c = r["conf"]
        y_prob.append([1 - c, c] if r["pred"] == 1 else [c, 1 - c])

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    total_tokens = sum(r.get("tokens", 0) for r in valid)
    metrics.update({
        "model": model, "task": task,
        "n_invalid": len(rows) - len(valid),
        "total_tokens": int(total_tokens),
        "mean_elapsed_s": round(float(np.mean([r["elapsed_s"] for r in valid])), 2),
    })
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  [{model}/{task}] DONE - AUC={metrics.get('auc_roc')}, "
          f"BalAcc={metrics['balanced_accuracy']}, F1={metrics['f1_macro']}")
    return metrics

def aggregate_all_results() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(STUDY_DIR.iterdir() if STUDY_DIR.exists() else []):
        if not model_dir.is_dir():
            continue
        for fp in model_dir.glob("*_results.json"):
            try:
                d = json.loads(fp.read_text())
                rows.append(d)
            except Exception:
                pass

    cnn_dir = PROJECT_ROOT / "baselines" / "results"
    for cond, task_name in [("adhd", "adhd_binary"), ("tumor", "tumor_grade")]:
        fp = cnn_dir / f"{cond}_cnn_results.json"
        if fp.exists():
            try:
                d = json.loads(fp.read_text())
                d.setdefault("model", "3D-CNN")
                d.setdefault("task", task_name)
                rows.append(d)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive LLM × NeuroAgent study")
    sub    = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run experiments")
    run_p.add_argument("--models",       nargs="+", default=DEFAULT_MODELS)
    run_p.add_argument("--tasks",        nargs="+", default=DEFAULT_TASKS)
    run_p.add_argument("--modes",        nargs="+", default=["direct"],
                       choices=["direct", "direct_matched", "neuroagent",
                                 "neuroagent_llm", "rule_only"],
                       help="Evaluation modes: direct (raw stats only) | "
                            "direct_matched (LLM with same features as agent) | "
                            "neuroagent (agent + rule decides) | "
                            "neuroagent_llm (agent + LLM decides) | "
                            "rule_only (agents + rule, no LLM)")
    run_p.add_argument("--max-subjects", type=int, default=50)
    run_p.add_argument("--resume",       action="store_true")
    run_p.add_argument("--sleep",        type=float, default=0.4)
    run_p.add_argument("--parallel",     type=int,   default=1,
                       help="In neuroagent mode, process N subjects concurrently "
                            "via ThreadPoolExecutor.  Only safe for vLLM-hosted "
                            "models (no rate limit).  Default: 1 (sequential).")
    run_p.add_argument("--balanced",     action="store_true",
                       help="Sample max-subjects with class balance (N/2 per class).")

    sum_p = sub.add_parser("summarize", help="Aggregate results into a CSV table")

    args = parser.parse_args()

    if args.command == "run":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set."); sys.exit(1)
        STUDY_DIR.mkdir(parents=True, exist_ok=True)

        all_results = []
        for mode in args.modes:
            for model in args.models:
                for task in args.tasks:
                    print(f"\n  Mode: {mode}  |  Model: {model}  |  Task: {task}  |  N: {args.max_subjects}")
                    try:
                        m = run_experiment(model, task, args.max_subjects,
                                           args.resume, args.sleep, mode=mode,
                                           parallel=args.parallel,
                                           balanced=args.balanced)
                        m["mode"] = mode
                        all_results.append(m)
                    except Exception as e:
                        print(f"  Failed: {e}")
                        all_results.append({"model": model, "task": task,
                                             "mode": mode, "error": str(e)})

        print("\nFINAL SUMMARY")
        new_df = pd.DataFrame(all_results)
        summary_path = STUDY_DIR / "summary.csv"
        if summary_path.exists():
            try:
                existing = pd.read_csv(summary_path)
                df = pd.concat([existing, new_df], ignore_index=True)
            except Exception as e:
                print(f"  [WARN] could not read existing summary.csv ({e}); starting fresh")
                df = new_df
        else:
            df = new_df
                                                             
        dedupe_keys = [c for c in ("model", "task", "mode") if c in df.columns]
        if dedupe_keys:
            df = df.drop_duplicates(subset=dedupe_keys, keep="last")
        cols = [c for c in ["model","task","mode","balanced_accuracy","f1_macro","auc_roc","sensitivity","specificity","n_samples"] if c in df.columns]
        if cols:
            print(df[cols].to_string(index=False))
        df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")

    elif args.command == "summarize":
        df = aggregate_all_results()
        if df.empty:
            print("No results found yet."); return
        df.to_csv(STUDY_DIR / "summary.csv", index=False)
        cols = [c for c in ["model","task","balanced_accuracy","f1_macro","auc_roc","sensitivity","specificity","n_samples"] if c in df.columns]
        print(df[cols].to_string(index=False))
        print(f"\nSaved: {STUDY_DIR / 'summary.csv'}")

if __name__ == "__main__":
    main()
