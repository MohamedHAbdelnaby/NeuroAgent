from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST     = PROJECT_ROOT / "preprocessing" / "manifests" / "all_subjects.csv"
PHENO_ROOT   = PROJECT_ROOT / "Datasets" / "ADHD200"
DEFAULT_OUT  = PROJECT_ROOT / "eval" / "ground_truth"

SITE_PHENO: dict[str, list[str]] = {
    "KKI":       ["KKI_phenotypic.csv"],
    "NYU":       ["NYU_phenotypic.csv"],
    "NeuroIMAGE":["NeuroIMAGE_phenotypic.csv"],
    "OHSU":      ["OHSU_phenotypic.csv", "OHSU_TestRelease_phenotypic.csv"],
    "Peking_1":  ["Peking_1_phenotypic.csv", "Peking_1_TestRelease_phenotypic.csv"],
    "Pittsburgh":["Pittsburgh_phenotypic.csv"],
}

def _load_adhd_pheno() -> pd.DataFrame:
    records = []
    for site, csvs in SITE_PHENO.items():
        for fname in csvs:
            fpath = PHENO_ROOT / fname
            if not fpath.exists():
                print(f"  [ADHD] Warning: {fpath} not found, skipping.", file=sys.stderr)
                continue
            df = pd.read_csv(fpath)
            id_col = "ScanDir ID" if "ScanDir ID" in df.columns else "ID"
            if "DX" not in df.columns or id_col not in df.columns:
                continue
            df = df.rename(columns={id_col: "scan_id"})
            df["scan_id"] = pd.to_numeric(df["scan_id"], errors="coerce")
            df["DX"]      = pd.to_numeric(df["DX"],      errors="coerce")
            df = df.dropna(subset=["scan_id", "DX"])
            df = df[df["DX"].isin([0, 1])].copy()
            df["label"]  = df["DX"].astype(int)
            df["site"]   = site
            df["source"] = fname
            age_col    = "Age"    if "Age"    in df.columns else None
            gender_col = "Gender" if "Gender" in df.columns else None
            df["age"]    = df[age_col]    if age_col    else np.nan
            df["gender"] = df[gender_col] if gender_col else np.nan
            records.append(df[["scan_id", "site", "label", "age", "gender", "source"]])
    if not records:
        return pd.DataFrame(columns=["scan_id", "site", "label", "age", "gender", "source"])
    return pd.concat(records, ignore_index=True).drop_duplicates(subset=["scan_id", "site"])

def extract_adhd(manifest: pd.DataFrame, out_dir: Path) -> Path:
    print("[ADHD] Building ground-truth labels...")
    pheno = _load_adhd_pheno()
    print(f"  Phenotypic rows with binary DX: {len(pheno)} "
          f"(ADHD={( pheno['label']==1).sum()}, control={(pheno['label']==0).sum()})")

    adhd_mf = manifest[manifest["dataset"] == "ADHD-200"].copy()
                                                                            
    adhd_mf["site"]      = adhd_mf["subject_id"].str.split("_sub-").str[0]
    adhd_mf["numeric_id"] = (
        adhd_mf["subject_id"].str.split("_sub-").str[1]
        .str.lstrip("0").apply(lambda x: int(x) if x else 0)
    )
    pheno = pheno.rename(columns={"scan_id": "numeric_id"})
    merged = adhd_mf.merge(pheno, on=["site", "numeric_id"], how="inner")
    print(f"  Matched {len(merged)} subjects with labels "
          f"(of {len(adhd_mf)} in manifest)")

    out = merged[["subject_id", "label", "site", "age", "gender", "source"]].copy()
    out = out.rename(columns={"source": "dx_source"})
    out_path = out_dir / "adhd_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(out)} subjects)")
    return out_path

def _centroid_mni(nii: nib.Nifti1Image, mask: np.ndarray) -> np.ndarray:
    vox = np.argwhere(mask > 0).astype(float)
    if len(vox) == 0:
        return np.array([0.0, 0.0, 0.0])
    vox_mean = vox.mean(axis=0)
    return nib.affines.apply_affine(nii.affine, vox_mean)

def _lateralize(x_mni: float) -> str:
    if abs(x_mni) < 5:
        return "bilateral"
    return "left" if x_mni < 0 else "right"

def extract_tumor(manifest: pd.DataFrame, out_dir: Path) -> Path:
    print("[Tumor] Building ground-truth labels from seg masks...")
    tumor_mf = manifest[manifest["dataset"] == "BraTS-GLI"].copy()
    tumor_mf = tumor_mf[tumor_mf["seg_path"].notna() & (tumor_mf["seg_path"] != "")]

    records = []
    for _, row in tumor_mf.iterrows():
        seg_path = Path(row["seg_path"])
        if not seg_path.exists():
            records.append({
                "subject_id": row["subject_id"],
                "grade_label": np.nan, "lateralization": np.nan,
                "has_et": np.nan, "lesion_volume_mm3": np.nan,
                "seg_path": str(seg_path),
            })
            continue
        try:
            nii   = nib.load(str(seg_path))
            data  = np.asarray(nii.dataobj)
            vox_vol = float(np.prod(np.abs(np.diag(nii.affine)[:3])))

            has_et      = bool((data == 3).any())
            grade_label = 1 if has_et else 0
            tumor_mask  = (data > 0).astype(np.uint8)
            volume_mm3  = float(tumor_mask.sum()) * vox_vol
            centroid    = _centroid_mni(nii, tumor_mask)
            lat         = _lateralize(centroid[0])
            records.append({
                "subject_id":       row["subject_id"],
                "grade_label":      grade_label,
                "lateralization":   lat,
                "has_et":           int(has_et),
                "lesion_volume_mm3": round(volume_mm3, 1),
                "seg_path":         str(seg_path),
            })
        except Exception as e:
            print(f"  Warning: failed for {row['subject_id']}: {e}", file=sys.stderr)
            records.append({
                "subject_id": row["subject_id"],
                "grade_label": np.nan, "lateralization": np.nan,
                "has_et": np.nan, "lesion_volume_mm3": np.nan,
                "seg_path": str(seg_path),
            })

    out = pd.DataFrame(records)
    valid = out.dropna(subset=["grade_label"])
    print(f"  Processed {len(out)} subjects, {len(valid)} valid")
    print(f"  Grade distribution: {valid['grade_label'].value_counts().to_dict()}")
    print(f"  Lateralization:     {valid['lateralization'].value_counts().to_dict()}")
    out_path = out_dir / "tumor_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path

def extract_stroke(manifest: pd.DataFrame, out_dir: Path) -> Path:
    print("[Stroke] Building ground-truth labels from lesion masks...")
    stroke_mf = manifest[manifest["dataset"] == "ATLAS-v2"].copy()
    stroke_mf = stroke_mf[
        stroke_mf["lesion_mask_path"].notna() & (stroke_mf["lesion_mask_path"] != "")
    ]

    records = []
    for _, row in stroke_mf.iterrows():
        mask_path = Path(row["lesion_mask_path"])
        if not mask_path.exists():
            records.append({
                "subject_id": row["subject_id"],
                "lateralization": np.nan,
                "lesion_volume_mm3": np.nan,
                "lesion_mask_path": str(mask_path),
            })
            continue
        try:
            nii  = nib.load(str(mask_path))
            data = np.asarray(nii.dataobj)
            vox_vol = float(np.prod(np.abs(np.diag(nii.affine)[:3])))

            mask = (data > 0).astype(np.uint8)
            volume_mm3 = float(mask.sum()) * vox_vol
            centroid   = _centroid_mni(nii, mask)
            lat        = _lateralize(centroid[0])

            fname = mask_path.name
            if "label-L" in fname:
                lat = "left"
            elif "label-R" in fname:
                lat = "right"

            records.append({
                "subject_id":       row["subject_id"],
                "lateralization":   lat,
                "lesion_volume_mm3": round(volume_mm3, 1),
                "lesion_mask_path": str(mask_path),
            })
        except Exception as e:
            print(f"  Warning: failed for {row['subject_id']}: {e}", file=sys.stderr)
            records.append({
                "subject_id": row["subject_id"],
                "lateralization": np.nan,
                "lesion_volume_mm3": np.nan,
                "lesion_mask_path": str(mask_path),
            })

    out = pd.DataFrame(records)
    valid = out.dropna(subset=["lateralization"])
    print(f"  Processed {len(out)} subjects, {len(valid)} valid")
    print(f"  Lateralization: {valid['lateralization'].value_counts().to_dict()}")
    out_path = out_dir / "stroke_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Extract ground-truth labels for NeuroAgent eval")
    parser.add_argument("--condition", default="all",
                        choices=["adhd", "tumor", "stroke", "all"])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    print(f"Manifest loaded: {len(manifest)} subjects")

    cond = args.condition
    if cond in ("adhd", "all"):
        extract_adhd(manifest, out_dir)
    if cond in ("tumor", "all"):
        extract_tumor(manifest, out_dir)
    if cond in ("stroke", "all"):
        extract_stroke(manifest, out_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()
