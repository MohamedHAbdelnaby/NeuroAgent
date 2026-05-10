
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.structural_mri_agent import StructuralMRIAgent
from agents.functional_fmri_agent import FunctionalFMRIAgent, _EMPIRICAL_FMRI_NORMS
from agents.orchestrator import NeuroAgentOrchestrator

LABELS_CSV = REPO_ROOT / "eval" / "ground_truth" / "adhd_labels.csv"
FASTSURFER_ROOT = REPO_ROOT / "preprocessing" / "fastsurfer_output" / "ADHD-200"
FMRIPREP_ROOT = REPO_ROOT / "preprocessing" / "fmriprep_output"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"fMRI empirical norms loaded: {bool(_EMPIRICAL_FMRI_NORMS)}", flush=True)
    if _EMPIRICAL_FMRI_NORMS:
        sites = sorted([s for s in _EMPIRICAL_FMRI_NORMS["norms"] if s != "GLOBAL"])
        print(f"  fMRI norms sites: {sites}", flush=True)

    controls, adhd = [], []
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            (controls if int(row["label"]) == 0 else adhd).append(row["subject_id"])

    random.seed(args.seed)
    sample = (
        [(s, 0) for s in random.sample(controls, min(args.n, len(controls)))] +
        [(s, 1) for s in random.sample(adhd, min(args.n, len(adhd)))]
    )
    random.shuffle(sample)
    n_total = len(sample)

    structural = StructuralMRIAgent(condition="adhd")
    functional = FunctionalFMRIAgent(condition="adhd")

    confusion = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    n_with_fmri = 0
    all_struct_z: list[float] = []
    all_fmri_z: list[float] = []
    ctrl_struct_z: list[float] = []
    adhd_struct_z: list[float] = []
    preds: list[int] = []

    for i, (sid, true_label) in enumerate(sample, 1):
        sd = FASTSURFER_ROOT / sid
        s_out = structural.run(sid, str(sd))
        sm = s_out.global_metrics or {}
        for k, v in sm.items():
            if k.startswith("z_") and isinstance(v, (int, float)):
                all_struct_z.append(v)
                (ctrl_struct_z if true_label == 0 else adhd_struct_z).append(v)

        site, sub = sid.split("_", 1)
        sub_dir = FMRIPREP_ROOT / site / sub
        fm: dict = {}
        if sub_dir.exists():
            bolds = sorted(sub_dir.rglob(
                "*MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))
            if bolds:
                confs = sorted(sub_dir.rglob("*confounds_timeseries.tsv"))
                try:
                    f_out = functional.run(
                        sid, bolds[0], confs[0] if confs else None
                    )
                    fm = f_out.global_metrics or {}
                    n_with_fmri += 1
                    for k, v in fm.items():
                        if k.startswith("z_") and isinstance(v, (int, float)):
                            all_fmri_z.append(v)
                except Exception:
                    pass

        rule_label, _ = NeuroAgentOrchestrator._compute_decision("adhd_binary", sm, fm)
        confusion[(true_label, rule_label)] += 1
        preds.append(rule_label)

        if i % 10 == 0 or i == n_total:
            print(f"  {i}/{n_total} processed", flush=True)

    tn, fp = confusion[(0, 0)], confusion[(0, 1)]
    fn, tp = confusion[(1, 0)], confusion[(1, 1)]
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    bal = (sens + spec) / 2
    acc = (tn + tp) / n_total

    print()
    print(f"Validation ({n_total} subjects, {n_with_fmri} with fMRI)")
    print(f"  predictions: 1={preds.count(1):3d}    0={preds.count(0):3d}    "
          f"(positive rate: {preds.count(1)/n_total:.1%})")
    print(f"  confusion:   TN={tn:3d}  FP={fp:3d}  FN={fn:3d}  TP={tp:3d}")
    print(f"  accuracy:    {acc:.3f}")
    print(f"  sensitivity: {sens:.3f}    specificity: {spec:.3f}    balanced acc: {bal:.3f}")
    print()
    print("Z-score sanity:")
    if all_struct_z:
        print(f"  ALL subjects structural z: mean={statistics.mean(all_struct_z):+.3f} "
              f"median={statistics.median(all_struct_z):+.3f} "
              f"frac|z|>=1.5={sum(1 for z in all_struct_z if abs(z)>=1.5)/len(all_struct_z):.2f}")
    if ctrl_struct_z:
        print(f"  controls   structural z: mean={statistics.mean(ctrl_struct_z):+.3f} "
              f"median={statistics.median(ctrl_struct_z):+.3f}")
    if adhd_struct_z:
        print(f"  ADHD       structural z: mean={statistics.mean(adhd_struct_z):+.3f} "
              f"median={statistics.median(adhd_struct_z):+.3f}")
    if all_fmri_z:
        print(f"  ALL subjects fMRI z:       mean={statistics.mean(all_fmri_z):+.3f} "
              f"median={statistics.median(all_fmri_z):+.3f} "
              f"frac|z|>=1.5={sum(1 for z in all_fmri_z if abs(z)>=1.5)/len(all_fmri_z):.2f}")
    print()
    print("Pass criteria:")
    print(f"  [{'PASS' if preds.count(1) < n_total else 'FAIL'}] not all-positive")
    print(f"  [{'PASS' if abs(statistics.mean(ctrl_struct_z)) < 0.3 else 'FAIL'}] control structural z mean within +/-0.3 of zero")
    print(f"  [{'PASS' if bal > 0.45 else 'FAIL'}] balanced accuracy at least chance (0.45)")

if __name__ == "__main__":
    main()
