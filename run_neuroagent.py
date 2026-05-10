from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FS_ROOT = PROJECT_ROOT / "preprocessing" / "fastsurfer_output"
FMRI_ROOT = PROJECT_ROOT / "preprocessing" / "fmriprep_output"

FASTSURFER_DIRS = {
    "tumor":  FS_ROOT / "BraTS-GLI",
    "stroke": FS_ROOT / "ATLAS-v2",
    "adhd":   FS_ROOT / "ADHD-200",
}

FMRI_SITES = [
    "Brown", "KKI", "NeuroIMAGE", "NYU", "OHSU",
    "Peking_1", "Peking_2", "Peking_3", "Pittsburgh", "WashU",
]

def _find_first_subject(condition: str) -> tuple[str, Path, Path | None, Path | None]:
    fs_dir = FASTSURFER_DIRS.get(condition)
    if fs_dir is None or not fs_dir.exists():
        raise FileNotFoundError(f"FastSurfer dir not found: {fs_dir}")

    subjects = sorted([d for d in fs_dir.iterdir() if d.is_dir()])
    if not subjects:
        raise FileNotFoundError(f"No subjects in {fs_dir}")

    subj_dir = subjects[0]
    subject_id = subj_dir.name
    bold_path, conf_path = None, None

    if condition == "adhd":
        for site in FMRI_SITES:
            site_dir = FMRI_ROOT / site
            if not site_dir.exists():
                continue
            for subj in sorted(site_dir.iterdir()):
                bolds = list(subj.rglob("*desc-preproc_bold.nii.gz"))
                if bolds:
                    bold_path = bolds[0]
                    confs = list(subj.rglob("*confounds_timeseries.tsv"))
                    conf_path = confs[0] if confs else None
                    adhd_subj_id = subj.name
                    adhd_stats = FASTSURFER_DIRS["adhd"] / adhd_subj_id
                    if adhd_stats.exists():
                        return adhd_subj_id, adhd_stats, bold_path, conf_path
                    break
            if bold_path:
                break

    return subject_id, subj_dir, bold_path, conf_path

def cmd_test(args):
    print("NeuroAgent Smoke Test")

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from agents.atlas_mapper_agent import AtlasMapper
    from agents.structural_mri_agent import StructuralMRIAgent
    from agents.clinical_knowledge_agent import ClinicalKnowledgeAgent

    print("\n[1/4] Atlas Mapper")
    mapper = AtlasMapper()
    for label in ["Left-Caudate", "ctx-rh-precentral", "17", "brain stem"]:
        r = mapper.map_freesurfer(label) if not label.startswith("brain") else mapper.map_text(label)
        status = "OK" if r.success else "FAIL"
        name = r.region.canonical_name if r.region else "---"
        print(f"  [{status}] '{label}' -> {name} (conf={r.confidence:.2f})")

    print("\n[2/4] Structural MRI Agent")
    for condition in ["tumor", "stroke", "adhd"]:
        try:
            subject_id, subj_dir, _, _ = _find_first_subject(condition)
            agent = StructuralMRIAgent(condition=condition)
            output = agent.run(subject_id=subject_id, stats_dir=subj_dir)
            print(f"  [{condition.upper()}] {subject_id}: status={output.status}, "
                  f"n_findings={len(output.findings)}, "
                  f"n_structures={output.metadata.get('n_structures_parsed', '?')}")
            if output.findings:
                f0 = output.findings[0]
                print(f"    Top finding: {f0.summary_line()[:100]}")
        except Exception as e:
            print(f"  [{condition.upper()}] ERROR: {e}")

    print("\n[3/4] Clinical Knowledge Agent")
    ck = ClinicalKnowledgeAgent()
    for condition, query in [
        ("tumor",  "glioblastoma ring enhancement necrosis grade"),
        ("stroke", "MCA territory aphasia hemiplegia"),
        ("adhd",   "caudate volume reduction frontostriatal"),
    ]:
        docs = ck.retrieve(condition, query, top_k=2)
        print(f"  [{condition.upper()}] '{query[:40]}...' -> "
              f"{len(docs)} docs (top: '{docs[0]['title'] if docs else 'none'}')")

    print("\n[4/4] Functional fMRI Agent")
    try:
        adhd_sid, adhd_stats, bold_path, conf_path = _find_first_subject("adhd")
        if bold_path and bold_path.exists():
            from agents.functional_fmri_agent import FunctionalFMRIAgent
            agent = FunctionalFMRIAgent(condition="adhd")
            output = agent.run(adhd_sid, bold_path, conf_path)
            print(f"  [ADHD] {adhd_sid}: status={output.status}, "
                  f"n_findings={len(output.findings)}")
            if output.status == "failed":
                err = (output.errors or output.warnings or ["Unknown error"])[0]
                print(f"    Error: {err}")
            if output.global_metrics:
                r_pcc = output.global_metrics.get("r_PCC_mPFC")
                if r_pcc is not None:
                    print(f"    PCC-mPFC r={r_pcc:.3f} "
                          f"(z={output.global_metrics.get('z_PCC_mPFC', 'N/A')})")
        else:
            print("  [ADHD] No BOLD file found, skipping fMRI test")
    except Exception as e:
        print(f"  [ADHD] fMRI test error: {e}")

    print("\n[Smoke Test Complete]")

def cmd_run(args):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from agents.orchestrator import NeuroAgentOrchestrator

    print(f"Running NeuroAgent: condition={args.condition}, subject={args.subject}")

    orch = NeuroAgentOrchestrator(
        condition=args.condition,
        model=args.model,
        mock=args.mock,
    )

    result = orch.run(
        subject_id=args.subject,
        stats_dir=args.stats_dir,
        bold_path=args.bold,
        confounds_path=args.confounds,
        output_dir=args.out_dir,
    )

    print("\nDiagnostic Summary")
    print(result.diagnostic_summary)
    print(f"\nConfidence: {result.confidence:.0%}")
    print(f"\nReport markdown preview (first 1000 chars):")
    print(result.report_markdown[:1000])

def cmd_batch(args):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from agents.structural_mri_agent import StructuralMRIAgent

    fs_root = Path(args.fastsurfer_root)
    if not fs_root.exists():
        print(f"Error: FastSurfer root not found: {fs_root}")
        sys.exit(1)

    subjects = sorted([d for d in fs_root.iterdir() if d.is_dir()])
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    print(f"Batch structural analysis: condition={args.condition}, "
          f"n_subjects={len(subjects)}")

    agent = StructuralMRIAgent(condition=args.condition)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, subj_dir in enumerate(subjects, start=1):
        subject_id = subj_dir.name
        print(f"  [{i}/{len(subjects)}] {subject_id}...", end=" ", flush=True)
        try:
            output = agent.run(subject_id=subject_id, stats_dir=subj_dir)
            out_path = out_dir / f"{subject_id}_{args.condition}.json"
            out_path.write_text(output.to_json())
            n_abn = sum(1 for f in output.findings if f.severity != "normal")
            print(f"OK ({n_abn} abnormal findings)")
            results.append({
                "subject_id": subject_id,
                "status": output.status,
                "n_findings": len(output.findings),
                "n_abnormal": n_abn,
                "global_metrics": output.global_metrics,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"subject_id": subject_id, "status": "error", "error": str(e)})

    manifest_path = out_dir / f"batch_{args.condition}_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nManifest saved: {manifest_path}")
    print(f"Done: {sum(1 for r in results if r['status']=='success')}/{len(results)} succeeded")

def main():
    parser = argparse.ArgumentParser(
        description="NeuroAgent Disease-Agnostic Multi-Agent Neuro-Imaging Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("test", help="Smoke-test all agents on available data")

    run_p = subparsers.add_parser("run", help="Run full pipeline for one subject")
    run_p.add_argument("--condition",  required=True, choices=["tumor", "stroke", "adhd"])
    run_p.add_argument("--subject",    required=True)
    run_p.add_argument("--stats-dir",  required=True)
    run_p.add_argument("--bold",       default=None)
    run_p.add_argument("--confounds",  default=None)
    run_p.add_argument("--out-dir",    default="./neuroagent_reports")
    run_p.add_argument("--model",      default="gpt-4o-mini")
    run_p.add_argument("--mock",       action="store_true")

    batch_p = subparsers.add_parser("batch", help="Batch structural analysis for multiple subjects")
    batch_p.add_argument("--condition",       required=True, choices=["tumor", "stroke", "adhd"])
    batch_p.add_argument("--fastsurfer-root", required=True)
    batch_p.add_argument("--out-dir",         default="./neuroagent_batch_results")
    batch_p.add_argument("--max-subjects",    type=int, default=None)

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "batch":
        cmd_batch(args)

if __name__ == "__main__":
    main()
