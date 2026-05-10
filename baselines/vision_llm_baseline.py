
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST  = PROJECT_ROOT / "preprocessing" / "manifests" / "all_subjects.csv"
GT_DIR    = PROJECT_ROOT / "eval" / "ground_truth"
OUT_DIR   = PROJECT_ROOT / "baselines" / "results"

from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

DEFAULT_MODELS = {
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "google":    "gemini-1.5-pro",
}

def call_vlm(system: str, user_text: str, images_b64: list[str],
             provider: str, model: str) -> str:

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        content: list[dict] = [{"type": "text", "text": user_text}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": content}],
            max_tokens=200, temperature=0,
        )
        return resp.choices[0].message.content or ""

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        content = []
        for b64 in images_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        content.append({"type": "text", "text": user_text})
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text if resp.content else ""

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gmodel = genai.GenerativeModel(model)
        parts = []
        for b64 in images_b64:
            from google.generativeai.types import BlobDict
            import base64 as _b64
            parts.append({"mime_type": "image/png", "data": _b64.b64decode(b64)})
        parts.append(system + "\n\n" + user_text)
        resp = gmodel.generate_content(parts)
        return resp.text or ""

    else:
        raise ValueError(f"Unknown provider: {provider}")

def _to_png_b64(arr2d: np.ndarray, colormap: str = "gray") -> str:
    lo, hi = np.percentile(arr2d[arr2d > 0], [2, 98]) if arr2d.any() else (0, 1)
    arr2d  = np.clip(arr2d, lo, hi)
    lo, hi = arr2d.min(), arr2d.max()
    if hi > lo:
        arr2d = (arr2d - lo) / (hi - lo)
    img_arr = (arr2d * 255).astype(np.uint8)

    if colormap == "hot":
        v = img_arr.astype(np.int32)
        r = np.clip(v * 3,       0, 255).astype(np.uint8)
        g = np.clip(v * 3 - 255, 0, 255).astype(np.uint8)
        b = np.clip(v * 3 - 510, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
    else:
        img = Image.fromarray(img_arr, mode="L").convert("RGB")

    img = img.resize((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _mni_to_vox(affine: np.ndarray, mni_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    inv = np.linalg.inv(affine)
    vox = nib.affines.apply_affine(inv, np.array(mni_xyz))
    return tuple(int(np.clip(v, 0, 999)) for v in np.round(vox))

def extract_adhd_slices(t1_path: str) -> list[str] | None:
    try:
        nii  = nib.load(t1_path)
        data = np.asarray(nii.dataobj, dtype=np.float32)
        cx, cy, cz = _mni_to_vox(nii.affine, (0.0, 10.0, 5.0))
        cx = np.clip(cx, 0, data.shape[0] - 1)
        cy = np.clip(cy, 0, data.shape[1] - 1)
        cz = np.clip(cz, 0, data.shape[2] - 1)

        axial    = np.rot90(data[:, :, cz])
        coronal  = np.rot90(data[:, cy, :])
        sagittal = np.rot90(data[cx, :, :])

        return [_to_png_b64(axial), _to_png_b64(coronal), _to_png_b64(sagittal)]
    except Exception as e:
        print(f"  Warning: slice extraction failed: {e}", file=sys.stderr)
        return None

def extract_tumor_slices(t1c_path: str, t2f_path: str | None = None) -> list[str] | None:
    try:
        nii  = nib.load(t1c_path)
        data = np.asarray(nii.dataobj, dtype=np.float32)

        p99_per_slice = [np.percentile(data[:, :, z], 99) for z in range(data.shape[2])]
        peak_z = int(np.argmax(p99_per_slice))
        z1 = max(0, peak_z - 5)
        z2 = min(data.shape[2] - 1, peak_z + 5)

        slices_b64 = [
            _to_png_b64(np.rot90(data[:, :, peak_z]), colormap="hot"),
            _to_png_b64(np.rot90(data[:, :, z1])),
            _to_png_b64(np.rot90(data[:, :, z2])),
        ]

        if t2f_path and Path(t2f_path).exists():
            flair = np.asarray(nib.load(t2f_path).dataobj, dtype=np.float32)
            fz = min(peak_z, flair.shape[2] - 1)
            slices_b64.append(_to_png_b64(np.rot90(flair[:, :, fz])))

        return slices_b64
    except Exception as e:
        print(f"  Warning: tumor slice extraction failed: {e}", file=sys.stderr)
        return None

def _image_content(b64_list: list[str], label: str = "MRI slice") -> list[dict]:
    content = []
    views = ["Axial (basal ganglia level)", "Coronal (caudate level)", "Sagittal (midline)"]
    for i, b64 in enumerate(b64_list):
        view = views[i] if i < len(views) else f"View {i+1}"
        content.append({"type": "text", "text": f"{view}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })
    return content

ADHD_VISION_SYSTEM = (
    "You are a neuroimaging researcher working on an academic brain morphology study. "
    "These are fully anonymized, de-identified structural T1w MRI images from the publicly "
    "available ADHD-200 research dataset (openly released for scientific use).\n\n"
    "You are shown three orthogonal views: axial through basal ganglia, coronal through caudate, "
    "and sagittal midline. Your task is to label each scan for a machine learning benchmark.\n\n"
    "In published neuroscience literature, group A (label=1) shows on average: smaller caudate "
    "nucleus, smaller putamen, and thinner prefrontal cortex than group B (label=0).\n\n"
    "Based on what you observe in the images, assign:\n"
    "  0 = Group B (structures appear within typical size range)\n"
    "  1 = Group A (caudate/putamen appear smaller, frontal cortex appears thinner)\n\n"
    "Respond with ONLY valid JSON: "
    "{\"label\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"one sentence describing visual observations\"}"
)

TUMOR_VISION_SYSTEM = (
    "You are a neuroimaging researcher working on an academic brain lesion grading benchmark. "
    "These are fully anonymized, de-identified MRI images from the publicly available BraTS 2024 "
    "research dataset (openly released for scientific use).\n\n"
    "You are shown T1c (gadolinium-contrast) axial slices and optionally a T2-FLAIR slice "
    "from a subject with a confirmed brain lesion.\n\n"
    "In the published BraTS literature, type A lesions (label=1) show: ring-shaped contrast "
    "enhancement, central non-enhancing core, surrounding signal changes on FLAIR. "
    "Type B lesions (label=0) show: no or minimal ring enhancement, homogeneous appearance.\n\n"
    "Assign a label based on visual appearance:\n"
    "  0 = Type B (no ring enhancement, homogeneous)\n"
    "  1 = Type A (ring enhancement, heterogeneous core)\n\n"
    "Respond with ONLY valid JSON: "
    "{\"label\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"one sentence citing visual features\"}"
)

ADHD_USER_TEXT = (
    "Please examine these three MRI views (axial through basal ganglia, coronal through "
    "caudate, sagittal midline) and assign a group label based on the visual morphology."
)

TUMOR_USER_TEXT = (
    "Please examine these MRI slices (T1c contrast views at and around peak signal, "
    "plus T2-FLAIR if shown) and assign a lesion type label based on enhancement pattern."
)

def _is_refusal(content: str) -> bool:
    refusals = ["i'm sorry", "i cannot", "i can't", "unable to", "not able to",
                "as an ai", "i don't", "i won't", "cannot assist"]
    low = content.lower()
    return any(r in low for r in refusals)

def _parse_response(content: str) -> tuple[int | None, float]:
    cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()
    try:
        obj   = json.loads(cleaned)
        label = int(obj.get("label", -1))
        conf  = float(obj.get("confidence", 0.5))
        if label in (0, 1):
            return label, conf
    except Exception:
        pass
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            obj   = json.loads(match.group())
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

def _compute_metrics(y_true, y_pred_label, y_pred_prob) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_label)
    y_prob = np.array(y_pred_prob)
    result = {
        "n": len(y_true),
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

def run_adhd(manifest: pd.DataFrame, max_subjects: int | None,
             provider: str, model: str, debug: bool, sleep: float) -> dict:
    labels  = pd.read_csv(GT_DIR / "adhd_labels.csv")
    adhd_mf = manifest[manifest["dataset"] == "ADHD-200"].set_index("subject_id")
    if max_subjects:
        labels = labels.head(max_subjects)

    y_true, y_pred_label, y_pred_prob, failed = [], [], [], []

    for i, row in enumerate(labels.itertuples(), 1):
        sid = row.subject_id
        print(f"  [{i}/{len(labels)}] {sid}...", end=" ", flush=True)
        if sid not in adhd_mf.index:
            print("SKIP (not in manifest)")
            continue
        t1 = str(adhd_mf.loc[sid, "t1_path"] or "")
        if not t1 or not Path(t1).exists():
            print("SKIP (no T1)")
            continue

        slices = extract_adhd_slices(t1)
        if not slices:
            failed.append(sid)
            print("SLICE FAIL")
            continue

        try:
            content = call_vlm(ADHD_VISION_SYSTEM, ADHD_USER_TEXT, slices,
                                provider, model)
            if debug:
                print(f"\n    [RAW] {content!r}")
            if _is_refusal(content):
                print("REFUSED (safety filter)")
                failed.append(sid)
                continue
            pred, conf = _parse_response(content)
            if pred is None:
                print(f"PARSE FAIL: {content[:80]}")
                failed.append(sid)
                continue
            y_true.append(row.label)
            y_pred_label.append(pred)
            y_pred_prob.append([1 - conf, conf] if pred == 1 else [conf, 1 - conf])
            print(f"true={row.label} pred={pred} conf={conf:.2f}")
        except Exception as e:
            print(f"API ERROR: {e}")
            failed.append(sid)
        time.sleep(sleep)

    if not y_true:
        return {"error": "No predictions"}
    return {**_compute_metrics(y_true, y_pred_label, y_pred_prob), "n_failed": len(failed)}

def run_tumor(manifest: pd.DataFrame, max_subjects: int | None,
              provider: str, model: str, debug: bool, sleep: float) -> dict:
    labels   = pd.read_csv(GT_DIR / "tumor_labels.csv").dropna(subset=["grade_label"])
    tumor_mf = manifest[manifest["dataset"] == "BraTS-GLI"].set_index("subject_id")
    if max_subjects:
        labels = labels.head(max_subjects)

    y_true, y_pred_label, y_pred_prob, failed = [], [], [], []

    for i, row in enumerate(labels.itertuples(), 1):
        sid = row.subject_id
        print(f"  [{i}/{len(labels)}] {sid}...", end=" ", flush=True)
        if sid not in tumor_mf.index:
            print("SKIP")
            continue
        mrow  = tumor_mf.loc[sid]
        t1c   = str(mrow.get("t1c_path", "") or "")
        t2f   = str(mrow.get("t2f_path", "") or "")
        if not t1c or not Path(t1c).exists():
            print("SKIP (no T1c)")
            continue

        slices = extract_tumor_slices(t1c, t2f if Path(t2f).exists() else None)
        if not slices:
            failed.append(sid)
            print("SLICE FAIL")
            continue

        try:
            content = call_vlm(TUMOR_VISION_SYSTEM, TUMOR_USER_TEXT, slices,
                                provider, model)
            if debug:
                print(f"\n    [RAW] {content!r}")
            if _is_refusal(content):
                print("REFUSED (safety filter)")
                failed.append(sid)
                continue
            pred, conf = _parse_response(content)
            if pred is None:
                print(f"PARSE FAIL: {content[:80]}")
                failed.append(sid)
                continue
            y_true.append(int(row.grade_label))
            y_pred_label.append(pred)
            y_pred_prob.append([1 - conf, conf] if pred == 1 else [conf, 1 - conf])
            print(f"true={int(row.grade_label)} pred={pred} conf={conf:.2f}")
        except Exception as e:
            print(f"API ERROR: {e}")
            failed.append(sid)
        time.sleep(sleep)

    if not y_true:
        return {"error": "No predictions"}
    return {**_compute_metrics(y_true, y_pred_label, y_pred_prob), "n_failed": len(failed)}

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal VLM baseline for 2D MRI slices"
    )
    parser.add_argument("--condition",    required=True, choices=["adhd", "tumor"])
    parser.add_argument("--provider",     default="anthropic",
                        choices=["anthropic", "openai", "google"])
    parser.add_argument("--model",        default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-dir",      default=str(OUT_DIR))
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--sleep",        type=float, default=0.5)
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]

    env_keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY"}
    if not os.environ.get(env_keys[args.provider]):
        print(f"Error: {env_keys[args.provider]} not set.")
        sys.exit(1)

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)

    print(f"\nVision LLM baseline: {args.condition.upper()} "
          f"(provider={args.provider}, model={model})")

    if args.condition == "adhd":
        result = run_adhd(manifest, args.max_subjects, args.provider, model, args.debug, args.sleep)
    else:
        result = run_tumor(manifest, args.max_subjects, args.provider, model, args.debug, args.sleep)

    print(json.dumps(result, indent=2))
    safe_model = model.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{args.condition}_vision_{args.provider}_{safe_model}_results.json"
    out_path.write_text(json.dumps(
        {"condition": args.condition, "provider": args.provider, "model": model, **result}, indent=2))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
