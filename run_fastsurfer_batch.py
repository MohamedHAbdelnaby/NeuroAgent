import argparse
from typing import Optional
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

DATASET_CONFIG = {
    "BraTS-GLI": {
        "description": "Glioma (skull-stripped), all segmentation modules",
    },
    "BraTS-MEN-RT": {
        "description": "Meningioma (skull-stripped), all segmentation modules",
    },
    "ATLAS-v2": {
        "description": "Stroke (MNI-space T1w), all segmentation modules",
    },
    "ADHD-200": {
        "description": "ADHD (native T1w), segmentation only (surfaces not used downstream)",
    },
}

ARC_BASE = os.environ.get("PROJECT") or str(Path(__file__).resolve().parent)

WIN_PREFIX = "D:\\NeuroAgent"

def win_to_arc(path: str) -> str:
    if not path:
        return path
    if path.startswith("/"):
        return path
    p = path.replace("\\", "/")
    win_prefix_normalised = WIN_PREFIX.replace("\\", "/")
    if p.startswith(win_prefix_normalised):
        relative = p[len(win_prefix_normalised):]
        relative = relative.lstrip("/")
        return f"{ARC_BASE}/{relative}"
    return p

def build_fastsurfer_cmd(
    fastsurfer_dir: str,
    t1_path: str,
    subject_id: str,
    subjects_dir: str,
    device: str,
    threads: int,
    fs_license: Optional[str],
    batch_size: int = 1,
) -> list[str]:

    script = os.path.join(fastsurfer_dir, "run_fastsurfer.sh")

    cmd = [
        "bash", script,
        "--t1", t1_path,
        "--sid", subject_id,
        "--sd", subjects_dir,
        "--device", device,
        "--threads", str(threads),
        "--3T",
        "--seg_only",
        "--vox_size", "1",
        "--viewagg_device", "cpu",
    ]

    if fs_license:
        cmd.extend(["--fs_license", fs_license])

    if device == "cpu":
        cmd.append("--no_cuda")

    return cmd

def check_existing(subjects_dir: str, subject_id: str) -> bool:
    seg_file = os.path.join(subjects_dir, subject_id, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    return os.path.isfile(seg_file)

def load_manifest(manifest_path: str, dataset_filter: str = None) -> list[dict]:
    with open(manifest_path, newline="") as fh:
        records = list(csv.DictReader(fh))
    if dataset_filter:
        records = [r for r in records if r["dataset"] == dataset_filter]
    return records

def write_log(results: list[dict], log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        json.dump(results, fh, indent=2)

def process_subject(
    record: dict,
    fastsurfer_dir: str,
    output_root: str,
    device: str,
    threads: int,
    fs_license: Optional[str],
    dry_run: bool,
    batch_size: int = 1,
) -> dict:
    sid = record["subject_id"]
    dataset = record["dataset"]

    t1_path = win_to_arc(record["t1_path"])

    subjects_dir = os.path.join(output_root, dataset)
    os.makedirs(subjects_dir, exist_ok=True)

    result = {
        "subject_id": sid,
        "dataset": dataset,
        "status": None,
        "t1_path": t1_path,
        "output_dir": os.path.join(subjects_dir, sid),
        "elapsed_seconds": 0,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    if not dry_run and not os.path.isfile(t1_path):
        result["status"] = "error"
        result["error"] = f"T1 file not found: {t1_path}"
        return result

    cmd = build_fastsurfer_cmd(
        fastsurfer_dir=fastsurfer_dir,
        t1_path=t1_path,
        subject_id=sid,
        subjects_dir=subjects_dir,
        device=device,
        threads=threads,
        fs_license=fs_license,
        batch_size=batch_size,
    )

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14400,
        )
        result["elapsed_seconds"] = round(time.time() - start, 1)

        err_log = os.path.join(subjects_dir, sid, "scripts", "fastsurfer_batch.log")
        os.makedirs(os.path.dirname(err_log), exist_ok=True)
        with open(err_log, "w") as fh:
            fh.write(f"STDOUT\n{proc.stdout}\n\nSTDERR\n{proc.stderr}")

        if proc.returncode != 0:
            result["status"] = "error"
            result["error"] = (proc.stderr or proc.stdout or "Unknown error")[-3000:]
        elif check_existing(subjects_dir, sid):
            result["status"] = "success"
        else:
            result["status"] = "no_output"
            result["error"] = (
                "run_fastsurfer.sh returned 0 but aparc.DKTatlas+aseg.deep.mgz "
                f"is missing. Check {err_log}"
            )

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Exceeded 4-hour timeout"
        result["elapsed_seconds"] = round(time.time() - start, 1)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["elapsed_seconds"] = round(time.time() - start, 1)

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Batch FastSurfer for all NeuroAgent datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fastsurfer-dir", type=str,
        default=f"{ARC_BASE}/FastSurfer",
    )
    parser.add_argument(
        "--manifest", type=str,
        default=f"{ARC_BASE}/preprocessing/manifests/all_subjects.csv",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=f"{ARC_BASE}/preprocessing/fastsurfer_output",
    )
    parser.add_argument(
        "--fs-license", type=str, default=None,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASET_CONFIG.keys()),
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--threads", type=int, default=4,
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
    )
    parser.add_argument(
        "--log-suffix", type=str, default="",
    )

    args = parser.parse_args()

    if not args.fs_license:
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home:
            candidate = os.path.join(fs_home, "license.txt")
            if os.path.isfile(candidate):
                args.fs_license = candidate

    script = os.path.join(args.fastsurfer_dir, "run_fastsurfer.sh")
    if not os.path.isfile(script):
        sys.exit(f"ERROR: run_fastsurfer.sh not found at {script}")

    records = load_manifest(args.manifest, args.dataset)

    if args.chunk_size is not None:
        records = records[args.start_idx : args.start_idx + args.chunk_size]

    if args.skip_existing:
        before = len(records)
        records = [
            r for r in records
            if not check_existing(
                os.path.join(args.output_dir, r["dataset"]),
                r["subject_id"],
            )
        ]
        skipped = before - len(records)
        if skipped:
            print(f"Skipping {skipped} already-processed subjects\n")

    if args.max_subjects:
        records = records[:args.max_subjects]

    if not records:
        print("No subjects to process.")
        return

    ds_counts = Counter(r["dataset"] for r in records)
    log_suffix = f"_{args.log_suffix}" if args.log_suffix else ""
    log_path = os.path.join(args.output_dir, f"run_log{log_suffix}.json")

    print("FastSurfer Batch Processing")
    print(f"  FastSurfer   : {args.fastsurfer_dir}")
    print(f"  Manifest     : {args.manifest}")
    print(f"  Output       : {args.output_dir}")
    print(f"  Device       : {args.device}")
    print(f"  Threads      : {args.threads}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  FS License   : {args.fs_license or 'not set'}")
    if args.chunk_size is not None:
        print(f"  Chunk        : start={args.start_idx} size={args.chunk_size}")
    print(f"  Total        : {len(records)} subjects")
    print()
    for ds, count in sorted(ds_counts.items()):
        print(f"    {ds:<15} {count:>5}  [seg_only]")
    print()

    results = []

    for i, record in enumerate(records, 1):
        sid = record["subject_id"]
        ds = record["dataset"]
        print(f"[{i:>4}/{len(records)}] {ds} / {sid} ...", end=" ", flush=True)

        result = process_subject(
            record=record,
            fastsurfer_dir=args.fastsurfer_dir,
            output_root=args.output_dir,
            device=args.device,
            threads=args.threads,
            fs_license=args.fs_license,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
        results.append(result)

        status = result["status"]
        elapsed = result["elapsed_seconds"]
        if status == "success":
            print(f"done ({elapsed}s)")
        elif status == "dry_run":
            print("(dry run)")
        elif status == "no_output":
            err_short = (result.get("error") or "")[:120]
            print(f"NO OUTPUT ({elapsed}s), check fastsurfer_batch.log")
            print(f"        {err_short}")
        else:
            err_short = (result.get("error") or "")[:100]
            print(f"FAILED: {status}")
            print(f"        {err_short}")

        if not args.dry_run:
            write_log(results, log_path)

    status_counts = Counter(r["status"] for r in results)
    print(f"\nDone. {len(results)} subjects processed.")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(results)
        print(f"    {status:<25} {count:>5}  ({pct:.1f}%)")
    if not args.dry_run:
        print(f"  Log: {log_path}")

if __name__ == "__main__":
    main()
