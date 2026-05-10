from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import os as _os
PROJECT_ROOT = Path(__file__).parent.parent
_STUDY_NAME  = _os.environ.get("STUDY_NAME", "study")
STUDY_DIR    = PROJECT_ROOT / "eval" / "results" / _STUDY_NAME
PLOTS_DIR    = STUDY_DIR / "plots"
REPORT_HTML  = STUDY_DIR / "REPORT.html"
REPORT_PDF   = STUDY_DIR / "REPORT.pdf"

def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()

def _load_results() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(STUDY_DIR.iterdir() if STUDY_DIR.exists() else []):
        if not model_dir.is_dir() or model_dir.name == "plots":
            continue
                                                          
        if "kimi" in model_dir.name.lower():
            continue
        for fp in model_dir.glob("*_results.json"):
            try:
                d = json.loads(fp.read_text())
                if "error" in d:
                    continue
                d["model_name"] = model_dir.name
                rows.append(d)
            except Exception:
                pass

    return pd.DataFrame(rows)

def _df_to_html_table(df: pd.DataFrame, columns: list[str]) -> str:
    df = df[[c for c in columns if c in df.columns]].copy()
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"<td>{v:.3f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"

CSS = """
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #1a1a2e;
    background: #f7f7fb;
    margin: 0;
    line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 28px 80px; }
header {
    background: #1a1a2e;
    color: #fff;
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
header h1 { font-size: 26px; margin: 0 0 6px; font-weight: 600; }
header .sub { color: #a8a8c2; font-size: 13px; }
section { margin-bottom: 36px; }
section h2 {
    border-left: 4px solid #4361ee;
    padding-left: 12px;
    font-size: 19px;
    margin: 0 0 16px;
    color: #1a1a2e;
}
section h3 { font-size: 15px; color: #2c2c54; margin-top: 24px; }
.card {
    background: #fff;
    border-radius: 8px;
    padding: 20px 24px;
    border: 1px solid #e6e6f0;
    margin-bottom: 16px;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }
th { background: #f0f0f7; font-weight: 600; color: #2c2c54; }
tr:nth-child(even) td { background: #fafafd; }
img { max-width: 100%; height: auto; display: block; margin: 8px auto; border-radius: 4px; }
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.metric-box {
    flex: 1;
    min-width: 140px;
    background: #f0f0f7;
    border-radius: 6px;
    padding: 10px 14px;
    border-left: 3px solid #4361ee;
}
.metric-box .label { font-size: 11px; color: #6c6c8a; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-box .value { font-size: 20px; font-weight: 600; color: #1a1a2e; }
.note { font-size: 12px; color: #6c6c8a; font-style: italic; }
footer { text-align: center; color: #6c6c8a; font-size: 12px; margin-top: 32px; }
@media print {
    body { background: #fff; }
    .wrap { max-width: none; padding: 0; }
    section { page-break-inside: avoid; }
    img { page-break-inside: avoid; max-height: 80vh; }
}
"""

METHODS_TEXT = """
<p>This study evaluates multiple large language models (LLMs) and a 3D CNN baseline
on two neuroimaging classification tasks, all running against ground-truth labels
extracted from the source datasets.</p>

<h3>Tasks</h3>
<ul>
  <li><b>adhd_binary</b> - ADHD vs typically developing control, derived from
      ADHD-200 phenotypic CSVs (DX 0/1).</li>
  <li><b>tumor_lat</b> - Left vs right hemisphere tumor lateralization, derived
      from BraTS-GLI segmentation mask centroids.</li>
  <li><b>tumor_grade</b> (CNN only) - Low- vs high-grade glioma, derived from
      enhancing-tumor (label 3) presence in the segmentation mask.</li>
</ul>

<h3>Direct LLM baselines</h3>
<p>For each LLM, we send a structured prompt containing:
  <b>(ADHD)</b> a z-score table of subcortical/frontal volumes computed against
  in-dataset controls, or <b>(tumor)</b> a left/right asymmetry table (LI = (L-R)/(L+R))
  derived from FastSurfer parcellation. The model responds with JSON
  <code>{label, confidence, reasoning}</code>. No tool calls or external knowledge
  are used. Each model is evaluated identically.</p>

<h3>3D CNN baseline</h3>
<p>A 4-stage residual 3D CNN (~2M parameters) trained on raw NIfTI volumes
  (T1w for ADHD; T1+T1c+T2w+T2-FLAIR for tumor grade). 70/15/15 train/val/test
  split, AdamW + cosine schedule, class-balanced cross-entropy.</p>

<h3>Metrics</h3>
<p>For each (model, task) we report: accuracy, balanced accuracy, precision and
  recall (macro and per-class), F1 (macro and per-class), AUC-ROC, PR-AUC,
  sensitivity, specificity, and the confusion matrix.</p>
"""

def _build_html(df: pd.DataFrame) -> str:
    n_models = df["model_name"].nunique()
    n_tasks  = df["task"].nunique()
    n_eval   = len(df)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    cols_main = ["model_name", "n_samples", "accuracy", "balanced_accuracy",
                 "f1_macro", "auc_roc", "pr_auc", "sensitivity", "specificity"]

    sections = []

    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].sort_values("balanced_accuracy", ascending=False)
        if sub.empty:
            continue

        bars  = _img_b64(PLOTS_DIR / f"metric_bars_{task}.png")
        cmat  = _img_b64(PLOTS_DIR / f"confusion_grid_{task}.png")
        lead  = _img_b64(PLOTS_DIR / f"leaderboard_{task}.png")

        best = sub.iloc[0]
        metric_boxes = "".join([
            f'<div class="metric-box"><div class="label">{lbl}</div>'
            f'<div class="value">{(best.get(k) if pd.notna(best.get(k)) else "-")}</div></div>'
            for lbl, k in [("Best model", "model_name"), ("Bal Acc", "balanced_accuracy"),
                           ("AUC-ROC", "auc_roc"), ("F1 macro", "f1_macro"),
                           ("Sens", "sensitivity"), ("Spec", "specificity")]
        ])

        section_html = f"""
        <section>
          <h2>Task: {task}</h2>
          <div class="card">
            <p>Tested on {n_models} models with up to <b>n={int(sub['n_samples'].dropna().max() if sub['n_samples'].notna().any() else 0)}</b> subjects each.
            The best-performing model is shown below.</p>
            <div class="metric-row">{metric_boxes}</div>
          </div>
          <div class="card">
            <h3>Per-model metrics</h3>
            {_df_to_html_table(sub, cols_main)}
          </div>
          {_img_section('Leaderboard', lead)}
          {_img_section('Detailed metrics comparison', bars)}
          {_img_section('Confusion matrices', cmat)}
        </section>
        """
        sections.append(section_html)

    heatmap = _img_b64(PLOTS_DIR / "heatmap_all.png")
    summary_section = ""
    if heatmap:
        summary_section = f"""
        <section>
          <h2>Cross-task summary</h2>
          <div class="card">{_img_section('Balanced accuracy heatmap', heatmap)}</div>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>NeuroAgent Comprehensive Study</title>
<style>{CSS}</style></head><body>
<div class="wrap">
  <header>
    <h1>NeuroAgent Comprehensive Study</h1>
    <div class="sub">
      LLM and 3D CNN baselines vs. NeuroAgent framework - {timestamp}<br>
      {n_models} models &nbsp;·&nbsp; {n_tasks} tasks &nbsp;·&nbsp; {n_eval} (model, task) results
    </div>
  </header>

  <section>
    <h2>Methods</h2>
    <div class="card">{METHODS_TEXT}</div>
  </section>

  {summary_section}

  {''.join(sections)}

  <footer>
    Generated by eval/generate_study_report.py · NeuroAgent · {timestamp}
  </footer>
</div></body></html>
"""
    return html

def _img_section(title: str, b64: str) -> str:
    if not b64:
        return ""
    return (f'<div class="card"><h3>{title}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>')

def main():
    df = _load_results()
    if df.empty:
        print("No results found. Run comprehensive_study.py first.")
        return
    print(f"Loaded {len(df)} results across {df['model_name'].nunique()} models")

    html = _build_html(df)
    REPORT_HTML.write_text(html)
    print(f"Saved HTML report: {REPORT_HTML}")

    try:
        from weasyprint import HTML
        HTML(string=html, base_url=str(STUDY_DIR)).write_pdf(str(REPORT_PDF))
        print(f"Saved PDF report:  {REPORT_PDF}")
    except Exception as e:
        print(f"PDF export skipped: {e}")

if __name__ == "__main__":
    main()
