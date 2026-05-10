from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os as _os
PROJECT_ROOT = Path(__file__).parent.parent
_STUDY_NAME  = _os.environ.get("STUDY_NAME", "study")
STUDY_DIR    = PROJECT_ROOT / "eval" / "results" / _STUDY_NAME
PLOTS_DIR    = STUDY_DIR / "plots"

plt.rcParams.update({
    "figure.dpi":   120,
    "savefig.dpi":  150,
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":   True,
    "grid.alpha":  0.3,
    "grid.linestyle": "--",
})

HEADLINE_METRICS = ["balanced_accuracy", "f1_macro", "auc_roc",
                    "sensitivity", "specificity"]
PRETTY_METRIC = {
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro":          "F1 (macro)",
    "auc_roc":           "AUC-ROC",
    "sensitivity":       "Sensitivity",
    "specificity":       "Specificity",
    "accuracy":          "Accuracy",
    "pr_auc":            "PR-AUC",
}

def load_all_results() -> pd.DataFrame:
    rows = []
    if not STUDY_DIR.exists():
        return pd.DataFrame()
    for model_dir in sorted(STUDY_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "plots":
            continue
                                                             
        if "kimi" in model_dir.name.lower():
            continue
        for fp in model_dir.glob("*_results.json"):
            try:
                d = json.loads(fp.read_text())
                if "error" in d:
                    continue
                                                                             
                folder = model_dir.name
                if "__neuroagent" in folder:
                    d["model_name"] = folder.replace("__neuroagent", "")
                    d["mode"]       = "neuroagent"
                else:
                    d["model_name"] = folder
                    d["mode"]       = "direct"
                rows.append(d)
            except Exception:
                pass

    return pd.DataFrame(rows)

def _sort_key_grouped(model_name: str) -> tuple[str, int]:
    base = model_name.replace(" +NA", "").strip()
    is_na = 1 if model_name.endswith("+NA") else 0
    return (base.lower(), is_na)

def _df_sorted_grouped(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["_sort"] = sub["model_name"].apply(_sort_key_grouped)
    return sub.sort_values("_sort").drop(columns=["_sort"])

def plot_metric_bars(df: pd.DataFrame, task: str, out_dir: Path):
    sub = df[df["task"] == task].copy()
    if sub.empty:
        return
    sub = _df_sorted_grouped(sub)
    metrics = [m for m in HEADLINE_METRICS if m in sub.columns]
    n_models  = len(sub)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * n_models), 5.5))
    x  = np.arange(n_models)
    w  = 0.8 / n_metrics
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_metrics))

    for i, m in enumerate(metrics):
        vals = sub[m].fillna(0).to_numpy(dtype=float)
        ax.bar(x + (i - n_metrics/2 + 0.5) * w, vals, width=w,
               color=cmap[i], label=PRETTY_METRIC.get(m, m), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(sub["model_name"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison - {task}  (n={int(sub['n_samples'].dropna().max() if sub['n_samples'].notna().any() else 0)})")
    ax.legend(loc="upper right", framealpha=0.95, ncol=len(metrics)//2 + 1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
               label="chance (0.5)")
    plt.tight_layout()
    out_path = out_dir / f"metric_bars_{task}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

def plot_confusion_grid(df: pd.DataFrame, task: str, out_dir: Path):
    sub = df[df["task"] == task].copy()
    sub = sub[sub["confusion_matrix"].notna()]
    if sub.empty:
        return
    sub = _df_sorted_grouped(sub)
    n   = len(sub)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows),
                              squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes.flat, sub.iterrows()):
        cm = np.array(row["confusion_matrix"])
        ax.axis("on")
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        bal = row.get("balanced_accuracy", "?")
        auc = row.get("auc_roc", None)
        title = f"{row['model_name']}\nBal={bal}"
        if auc:
            title += f"  AUC={auc}"
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"Confusion Matrices - {task}", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"confusion_grid_{task}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

def plot_heatmap(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    metrics = [m for m in HEADLINE_METRICS if m in df.columns]
    pivot = df.pivot_table(index="model_name", columns="task",
                           values="balanced_accuracy", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(pivot.columns)),
                                     max(4, 0.5 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Balanced Accuracy")
    ax.set_title("Cross-task balanced accuracy heatmap")
    plt.tight_layout()
    out_path = out_dir / "heatmap_all.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

def _load_per_subject(model_name: str, task: str, mode: str) -> dict[str, tuple[int, int]]:
    folder = f"{model_name}__neuroagent" if mode == "neuroagent" else model_name
    fp = STUDY_DIR / folder / f"{task}_per_subject.jsonl"
    out: dict[str, tuple[int, int]] = {}
    if not fp.exists():
        return out
    for line in fp.read_text().splitlines():
        try:
            d = json.loads(line)
            if d.get("pred") in (0, 1):
                out[d["subject_id"]] = (int(d["true"]), int(d["pred"]))
        except Exception:
            pass
    return out

def _balanced_acc_on_intersection(direct_p, na_p):
    from sklearn.metrics import balanced_accuracy_score
    common = sorted(set(direct_p) & set(na_p))
    if len(common) < 2:
        return None, None, 0
    dy = [direct_p[s][0] for s in common]; dp = [direct_p[s][1] for s in common]
    ny = [na_p[s][0]     for s in common]; np_ = [na_p[s][1]     for s in common]
    try:
        d_acc = float(balanced_accuracy_score(dy, dp))
        n_acc = float(balanced_accuracy_score(ny, np_))
        return d_acc, n_acc, len(common)
    except Exception:
        return None, None, len(common)

def plot_direct_vs_neuroagent(df: pd.DataFrame, task: str, out_dir: Path):
    sub = df[df["task"] == task].copy()
    if sub.empty or "mode" not in sub.columns:
        return

    pivot = sub.pivot_table(index="model_name", columns="mode",
                             values="balanced_accuracy", aggfunc="first")
    if "direct" not in pivot.columns or "neuroagent" not in pivot.columns:
        return
    paired_models = pivot.dropna(subset=["direct", "neuroagent"], how="any").index.tolist()
    if not paired_models:
        return

    rows = []
    for m in paired_models:
        d_preds = _load_per_subject(m, task, "direct")
        n_preds = _load_per_subject(m, task, "neuroagent")
        d_acc, n_acc, n_common = _balanced_acc_on_intersection(d_preds, n_preds)
        if d_acc is None or n_acc is None:
            continue
        if n_common < MIN_N_SUBJECTS:
            print(f"  Filter: skip {m} on {task} - matched n={n_common} < {MIN_N_SUBJECTS}")
            continue
        rows.append((m, d_acc, n_acc, n_common))

    if not rows:
        return
    rows.sort(key=lambda r: -r[2])                       
    models  = [r[0] for r in rows]
    d_vals  = [r[1] for r in rows]
    n_vals  = [r[2] for r in rows]
    Ns      = [r[3] for r in rows]
    n_models = len(models)
    matched_N = min(Ns)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_models), 5.4))
    x = np.arange(n_models)
    w = 0.38
    ax.bar(x - w/2, d_vals, width=w, label="Direct LLM (baseline)",
           color="#94a3b8", edgecolor="white")
    ax.bar(x + w/2, n_vals, width=w, label="LLM + NeuroAgent (ours)",
           color="#4361ee", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={N})" for m, N in zip(models, Ns)],
                       rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"{task} - direct vs NeuroAgent on matched subjects "
                 f"(min n={matched_N}, paired models only)")
    ax.legend()
    for i, (d, n_) in enumerate(zip(d_vals, n_vals)):
        ax.text(i - w/2, d + 0.01, f"{d:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, n_ + 0.01, f"{n_:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / f"direct_vs_neuroagent_{task}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}  (paired models: {len(models)}, matched min n: {matched_N})")

def plot_leaderboard(df: pd.DataFrame, task: str, out_dir: Path):
    sub = df[df["task"] == task].copy()
    if sub.empty:
        return
    metric = "auc_roc" if "auc_roc" in sub.columns and sub["auc_roc"].notna().any() else "balanced_accuracy"
                                                                            
    sub = _df_sorted_grouped(sub).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(sub))))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sub)))
    bars = ax.barh(sub["model_name"], sub[metric].fillna(0), color=colors)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(PRETTY_METRIC.get(metric, metric))
    ax.set_title(f"{task} leaderboard - {PRETTY_METRIC.get(metric, metric)}")
    for bar, val in zip(bars, sub[metric].fillna(0)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / f"leaderboard_{task}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

MIN_N_SUBJECTS = 450                                                 

def _apply_strict_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()

    if "n_samples" in work.columns:
        before = len(work)
        keep_mask = work["n_samples"].isna() | (work["n_samples"] >= MIN_N_SUBJECTS)
        work = work[keep_mask].copy()
        dropped = before - len(work)
        if dropped:
            print(f"  Filter: dropped {dropped} rows with n_samples < {MIN_N_SUBJECTS}")

    if "mode" in work.columns and "model_name" in work.columns:
        keep_idx = []
        for task, grp in work.groupby("task"):
            llm_grp = grp[grp["mode"].isin(["direct", "neuroagent"])]
            cnn_grp = grp[~grp["mode"].isin(["direct", "neuroagent"])]
                                           
            keep_idx.extend(cnn_grp.index.tolist())
                                                                                      
            for model, mgrp in llm_grp.groupby("model_name"):
                modes_present = set(mgrp["mode"].unique())
                if modes_present < {"direct", "neuroagent"}:
                    only = mgrp["mode"].iloc[0]
                    print(f"  Filter: drop {model} on {task} - only has '{only}' mode")
                    continue
                                        
                d_preds = _load_per_subject(model, task, "direct")
                n_preds = _load_per_subject(model, task, "neuroagent")
                matched_n = len(set(d_preds) & set(n_preds))
                if matched_n < MIN_N_SUBJECTS:
                    print(f"  Filter: drop {model} on {task} - matched n={matched_n} < {MIN_N_SUBJECTS}")
                    continue
                keep_idx.extend(mgrp.index.tolist())
        work = work.loc[keep_idx].copy()
    return work

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_all_results()
    if df_raw.empty:
        print("No results found. Run the comprehensive study first.")
        return

    df = _apply_strict_filter(df_raw)
    if df.empty:
        print(f"After strict filter (N>={MIN_N_SUBJECTS}, paired modes only), nothing left.")
        return

    df.to_csv(STUDY_DIR / "summary.csv", index=False)
    print(f"Loaded {len(df_raw)} raw -> {len(df)} after strict filter (N>={MIN_N_SUBJECTS}, paired)")
    print(f"Models surviving: {sorted(df['model_name'].unique().tolist())}")
    print()
    print("Per-task summary:")
    cols = ["model_name", "balanced_accuracy", "f1_macro", "auc_roc",
            "sensitivity", "specificity", "n_samples"]
    cols = [c for c in cols if c in df.columns]
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        print(f"\n  {task}:")
        print(sub[cols].sort_values("balanced_accuracy", ascending=False).to_string(index=False))

    df_plot = df.copy()
    if "mode" in df_plot.columns:
        def _label(r):
            mode = r.get("mode")
            if pd.isna(mode):                                             
                return r["model_name"]
            if mode == "neuroagent":
                return f"{r['model_name']} +NA"
            return r["model_name"]
        df_plot["model_name"] = df_plot.apply(_label, axis=1)

    print(f"\nGenerating plots in {PLOTS_DIR}...")
    for task in df_plot["task"].unique():
        plot_metric_bars(df_plot, task, PLOTS_DIR)
        plot_confusion_grid(df_plot, task, PLOTS_DIR)
        plot_leaderboard(df_plot, task, PLOTS_DIR)
    plot_heatmap(df_plot, PLOTS_DIR)

    for task in df["task"].unique():
        plot_direct_vs_neuroagent(df, task, PLOTS_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()
