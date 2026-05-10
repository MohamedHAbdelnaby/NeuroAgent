from pathlib import Path

import csv
import json
import os
import sys

ARC_BASE = os.environ.get("PROJECT") or str(Path(__file__).resolve().parent)
WIN_PREFIX = "D:\\NeuroAgent"
WIN_PREFIX_SLASH = "D:/NeuroAgent"

MANIFESTS_DIR = f"{ARC_BASE}/preprocessing/manifests"
CSV_PATH = f"{MANIFESTS_DIR}/all_subjects.csv"
JSON_PATH = f"{MANIFESTS_DIR}/all_subjects.json"

def win_to_arc(path: str) -> str:
    if not path:
        return path
    if path.startswith("/"):
        return path
    p = path.replace("\\", "/")
    if p.startswith(WIN_PREFIX_SLASH):
        relative = p[len(WIN_PREFIX_SLASH):].lstrip("/")
        return f"{ARC_BASE}/{relative}"
    return p

PATH_FIELDS = [
    "t1_path", "t1c_path", "t2w_path", "t2f_path",
    "seg_path", "fmri_path", "lesion_mask_path",
]

def fix_csv(path: str):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        for field in PATH_FIELDS:
            if field in row:
                row[field] = win_to_arc(row[field])

    backup = path + ".bak"
    os.rename(path, backup)
    print(f"  Backed up original to {backup}")

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Fixed CSV written to {path}")

def fix_json(path: str):
    with open(path) as fh:
        records = json.load(fh)

    for record in records:
        for field in PATH_FIELDS:
            if field in record:
                record[field] = win_to_arc(record[field])

    backup = path + ".bak"
    os.rename(path, backup)
    print(f"  Backed up original to {backup}")

    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)

    print(f"  Fixed JSON written to {path}")

def main():
    print("Fixing manifest paths: Windows -> ARC\n")

    if os.path.isfile(CSV_PATH):
        print(f"Processing {CSV_PATH}")
        fix_csv(CSV_PATH)
    else:
        print(f"WARNING: CSV not found at {CSV_PATH}", file=sys.stderr)

    print()

    if os.path.isfile(JSON_PATH):
        print(f"Processing {JSON_PATH}")
        fix_json(JSON_PATH)
    else:
        print(f"WARNING: JSON not found at {JSON_PATH}", file=sys.stderr)

    print("\nDone. Verify a few entries with:")
    print(f"  head -3 {CSV_PATH}")

if __name__ == "__main__":
    main()
