from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from agents.schemas import (
    AgentOutput, AtlasRegion, Measurement, Finding,
    make_finding, Condition,
)
from agents.atlas_mapper_agent import AtlasMapper

_FALLBACK_ADHD_NORMS: dict[str, tuple[float, float]] = {
    "Left-Caudate":          (3500.0, 450.0),
    "Right-Caudate":         (3600.0, 460.0),
    "Left-Putamen":          (5200.0, 620.0),
    "Right-Putamen":         (5300.0, 630.0),
    "Left-Pallidum":         (1800.0, 250.0),
    "Right-Pallidum":        (1800.0, 255.0),
    "Left-Hippocampus":      (4100.0, 480.0),
    "Right-Hippocampus":     (4150.0, 490.0),
    "Left-Amygdala":         (1650.0, 220.0),
    "Right-Amygdala":        (1680.0, 225.0),
    "Left-Accumbens-area":   (580.0,  100.0),
    "Right-Accumbens-area":  (590.0,  105.0),
    "Left-Thalamus":         (8000.0, 780.0),
    "Right-Thalamus":        (7900.0, 770.0),
    "Left-Cerebellum-Cortex":  (55000.0, 5500.0),
    "Right-Cerebellum-Cortex": (55000.0, 5500.0),
}

_ICV_REF = 1_400_000.0

_NORMS_PATH = Path(__file__).resolve().parent.parent / "data" / "adhd200_structural_norms.json"
_KNOWN_SITES = ("Peking_1", "Peking_2", "Peking_3", "KKI", "NYU", "OHSU",
                "NeuroIMAGE", "Pittsburgh", "Brown", "WashU")

def _load_empirical_norms() -> dict | None:
    if not _NORMS_PATH.exists():
        return None
    try:
        return json.loads(_NORMS_PATH.read_text())
    except Exception:
        return None

_EMPIRICAL_NORMS = _load_empirical_norms()

def _site_from_subject_id(subject_id: str) -> str | None:
    for site in _KNOWN_SITES:
        if subject_id.startswith(site + "_") or subject_id == site:
            return site
    if "_sub-" in subject_id:
        return subject_id.split("_sub-")[0]
    return None

def _norms_for_subject(subject_id: str) -> dict[str, tuple[float, float]]:
    if _EMPIRICAL_NORMS:
        all_sites = _EMPIRICAL_NORMS.get("norms", {})
        site = _site_from_subject_id(subject_id)
        site_norms = all_sites.get(site) if site else None
        if site_norms is None:
            site_norms = all_sites.get("GLOBAL")
        if site_norms:
            out: dict[str, tuple[float, float]] = {}
            for s, d in site_norms.items():
                if d.get("sd", 0) > 0:
                    out[s] = (float(d["mean"]), float(d["sd"]))
            for s, mu_sd in _FALLBACK_ADHD_NORMS.items():
                out.setdefault(s, mu_sd)
            return out
    return dict(_FALLBACK_ADHD_NORMS)

_ADHD_NORMS = dict(_FALLBACK_ADHD_NORMS)
if _EMPIRICAL_NORMS and _EMPIRICAL_NORMS.get("norms", {}).get("GLOBAL"):
    _ADHD_NORMS = {
        s: (float(d["mean"]), float(d["sd"]))
        for s, d in _EMPIRICAL_NORMS["norms"]["GLOBAL"].items()
        if d.get("sd", 0) > 0
    }

def _z_to_severity(z: float) -> str:
    az = abs(z)
    if az < 1.5:
        return "normal"
    if az < 2.0:
        return "mild"
    if az < 2.5:
        return "moderate"
    return "severe"

def _parse_stats_file(stats_path: Path) -> tuple[dict[str, float], float]:
    volumes: dict[str, float] = {}
    icv = 0.0

    if not stats_path.exists():
        return volumes, icv

    with open(stats_path) as f:
        for line in f:
            line = line.strip()
                                                        
            if line.startswith("# Measure"):
                parts = line.split(",")
                if len(parts) >= 4:
                    measure_name = parts[0].split()[-1].strip()
                    try:
                        val = float(parts[3].strip())
                    except ValueError:
                        continue
                    if measure_name in ("Mask", "MaskVol"):
                        icv = val
                                                                             
                    if measure_name == "BrainSeg":
                        volumes["BrainSeg_vol"] = val
                    elif measure_name == "BrainSegNotVent":
                        volumes["BrainSegNotVent_vol"] = val
                continue

            if line.startswith("#") or not line:
                continue

            cols = line.split()
            if len(cols) < 5:
                continue
            try:
                vol = float(cols[3])
                name = cols[4]
                volumes[name] = vol
            except (ValueError, IndexError):
                continue

    return volumes, icv

def _find_stats_file(stats_dir: Path, preferred: str = "aseg+DKT.VINN.stats") -> Optional[Path]:
    for name in [preferred, "aseg.VINN.stats", "aseg.stats"]:
        p = stats_dir / name
        if p.exists():
            return p
    return None

_TUMOR_GRADE_FEATS = [
    "BrainSeg_vol", "BrainSegNotVent_vol",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
    "3rd-Ventricle", "4th-Ventricle",
    "Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Thalamus", "Right-Thalamus",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Brain-Stem",
    "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
]
_TUMOR_LI_PAIRS = [
    ("Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
    ("Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter"),
    ("Left-Caudate", "Right-Caudate"),
    ("Left-Putamen", "Right-Putamen"),
    ("Left-Thalamus", "Right-Thalamus"),
    ("Left-Hippocampus", "Right-Hippocampus"),
    ("Left-Amygdala", "Right-Amygdala"),
    ("Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"),
]

def _build_tumor_grade_features(volumes: dict, icv: float) -> dict[str, float]:
    if icv <= 0: return {}
    out: dict[str, float] = {}
    for f in _TUMOR_GRADE_FEATS:
        v = volumes.get(f, 0.0)
        if v > 0:
            out[f] = v / icv * 1e6
    for pair in _TUMOR_LI_PAIRS:
        l = volumes.get(pair[0], 0.0); r = volumes.get(pair[1], 0.0)
        if l > 0 and r > 0:
            out[f"LI_{pair[0].replace('Left-', '')}"] = (l - r) / (l + r)
            out[f"sum_{pair[0].replace('Left-', '')}_per_icv"] = (l + r) / icv * 1e6
    tot_l = sum(v for k, v in volumes.items() if k.startswith("Left-") and v > 0)
    tot_r = sum(v for k, v in volumes.items() if k.startswith("Right-") and v > 0)
    if tot_l + tot_r > 0:
        out["TOTAL_LI"] = (tot_l - tot_r) / (tot_l + tot_r)
        out["abs_TOTAL_LI"] = abs((tot_l - tot_r) / (tot_l + tot_r))
    vl = volumes.get("Left-Lateral-Ventricle", 0.0)
    vr = volumes.get("Right-Lateral-Ventricle", 0.0)
    if vl + vr > 0:
        out["abs_LI_ventricle"] = abs((vl - vr) / (vl + vr))
        out["total_ventricle_per_icv"] = (vl + vr) / icv * 1e6
    return out

def _analyze_lesion_mask(mask_path: str | Path, label_prefix: str = "lesion") -> dict:
    out: dict = {}
    if not mask_path:
        return out
    p = Path(mask_path)
    if not p.exists():
        return out
    try:
        import nibabel as nib
        import numpy as np
        seg = nib.load(str(p))
        data = seg.get_fdata()
        nz = np.argwhere(data > 0)
        if len(nz) == 0:
            return {f"{label_prefix}_total_voxels": 0}
        mni = nib.affines.apply_affine(seg.affine, nz)
        x_coords = mni[:, 0]
        n_left  = int(np.sum(x_coords < 0))
        n_right = int(np.sum(x_coords >= 0))
        out[f"{label_prefix}_total_voxels"]   = int(len(nz))
        out[f"{label_prefix}_left_voxels"]    = n_left
        out[f"{label_prefix}_right_voxels"]   = n_right
        out[f"{label_prefix}_centroid_x_mni"] = round(float(x_coords.mean()), 3)
        if n_left + n_right > 0:
            out[f"{label_prefix}_lat_mask"] = round((n_right - n_left) / (n_right + n_left), 4)
    except Exception as e:
        out[f"{label_prefix}_mask_error"] = f"{type(e).__name__}: {e}"
    return out

def _analyze_tumor_seg_mask(seg_path: str | Path) -> dict:
    out: dict = {}
    if not seg_path:
        return out
    p = Path(seg_path)
    if not p.exists():
        return out
    try:
        import nibabel as nib
        import numpy as np
        seg = nib.load(str(p))
        data = seg.get_fdata()
        nz = np.argwhere(data > 0)
        if len(nz) == 0:
            return {"tumor_total_voxels": 0}
                                                             
        mni = nib.affines.apply_affine(seg.affine, nz)
        x_coords = mni[:, 0]
        n_left  = int(np.sum(x_coords < 0))
        n_right = int(np.sum(x_coords >= 0))
        out["tumor_total_voxels"]   = int(len(nz))
        out["tumor_left_voxels"]    = n_left
        out["tumor_right_voxels"]   = n_right
        out["tumor_centroid_x_mni"] = round(float(x_coords.mean()), 3)
                                                                             
        if n_left + n_right > 0:
            out["tumor_lat_mask"] = round((n_right - n_left) / (n_right + n_left), 4)
                                                                                          
        unique, counts = np.unique(data[data > 0].astype(int), return_counts=True)
        out["tumor_classes"] = {int(k): int(v) for k, v in zip(unique, counts)}
    except Exception as e:
        out["tumor_seg_error"] = f"{type(e).__name__}: {e}"
    return out

def _analyze_tumor(
    volumes: dict[str, float],
    icv: float,
    mapper: AtlasMapper,
    seg_path: Optional[str] = None,
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    idx = 0
    global_metrics: dict = {}

    tumor_features = _build_tumor_grade_features(volumes, icv)
    global_metrics["tumor_features"] = tumor_features

    if seg_path:
        seg_metrics = _analyze_tumor_seg_mask(seg_path)
        global_metrics.update(seg_metrics)
        if seg_metrics.get("tumor_total_voxels", 0) > 0:
            cx = seg_metrics.get("tumor_centroid_x_mni")
            n_l = seg_metrics.get("tumor_left_voxels", 0)
            n_r = seg_metrics.get("tumor_right_voxels", 0)
            dom = "left" if (cx is not None and cx < 0) else "right"
            idx += 1
            r = mapper.map_text(f"{dom} hemisphere")
            region = r.region if r.success else AtlasRegion(
                canonical_name=f"{dom.capitalize()} Cerebral Hemisphere",
                hemisphere=dom, lobe="frontal",
                atlas_label=f"Hemisphere_{dom[0].upper()}",
                mni_centroid=(-30.0 if dom == "left" else 30.0, 0, 20),
            )
            findings.append(make_finding(
                idx=idx,
                finding_type="tumor_localization",
                severity="severe",
                region=region,
                measurement=Measurement(
                    metric="tumor_centroid_x_mni",
                    value=cx if cx is not None else 0.0,
                    units="mm",
                    reference_min=-1.0, reference_max=1.0,
                    normative_source="brats_segmentation_mask",
                ),
                clinical_significance=(
                    f"Tumor centroid at MNI x={cx:+.1f} mm with "
                    f"{n_l} voxels left of midline and {n_r} right -> "
                    f"{dom}-hemisphere tumor."
                ),
                evidence_chain=[
                    f"Segmentation mask: {seg_path}",
                    f"Total tumor voxels: {seg_metrics['tumor_total_voxels']}",
                    f"Left vs right voxels: {n_l} / {n_r}",
                    f"Mean MNI x: {cx:+.2f} mm (negative=left)",
                ],
            ))

    total_l = 0.0
    total_r = 0.0
    for k, v in volumes.items():
        if not isinstance(v, (int, float)) or v <= 0: continue
        kl = k.lower()
        if kl.startswith("left-"):     total_l += v
        elif kl.startswith("right-"):  total_r += v
        elif kl.startswith("ctx-lh-"): total_l += v
        elif kl.startswith("ctx-rh-"): total_r += v
    if total_l + total_r > 0:
        global_metrics["total_li"] = round((total_l - total_r) / (total_l + total_r), 4)
        global_metrics["total_left_mm3"]  = round(total_l)
        global_metrics["total_right_mm3"] = round(total_r)

    lh_wm = volumes.get("Left-Cerebral-White-Matter", 0)
    rh_wm = volumes.get("Right-Cerebral-White-Matter", 0)
    if lh_wm > 0 and rh_wm > 0:
        total_wm = lh_wm + rh_wm
        li = (lh_wm - rh_wm) / total_wm                                   
        global_metrics["wm_lateralisation_index"] = round(li, 4)
        global_metrics["lh_wm_mm3"] = lh_wm
        global_metrics["rh_wm_mm3"] = rh_wm

        if abs(li) > 0.05:
            dom = "left" if li > 0 else "right"
            deficit = "right" if li > 0 else "left"
            sev = "mild" if abs(li) < 0.10 else ("moderate" if abs(li) < 0.20 else "severe")
            idx += 1
            r = mapper.map_text(f"{deficit} hemisphere white matter")
            region = r.region if r.success else AtlasRegion(
                canonical_name=f"{deficit.capitalize()} Cerebral Hemisphere",
                hemisphere=deficit, lobe="frontal",
                atlas_label=f"Hemisphere_{deficit[0].upper()}",
                mni_centroid=(-30.0 if deficit == "left" else 30.0, 0, 20),
            )
            findings.append(make_finding(
                idx=idx,
                finding_type="volume_asymmetry",
                severity=sev,
                region=region,
                measurement=Measurement(
                    metric="wm_lateralisation_index",
                    value=abs(li),
                    units="dimensionless",
                    reference_min=0.0, reference_max=0.05,
                    normative_source="symmetric_brain",
                ),
                clinical_significance=(
                    f"White matter volume asymmetry (LI={li:+.3f}): {dom} hemisphere larger. "
                    f"May indicate mass effect, oedema, or prior surgery on {deficit} side."
                ),
                evidence_chain=[
                    f"Left WM volume: {lh_wm:.0f} mm³",
                    f"Right WM volume: {rh_wm:.0f} mm³",
                    f"Lateralisation index: {li:+.4f} (>±0.05 abnormal)",
                ],
            ))

    lv_l = volumes.get("Left-Lateral-Ventricle", 0)
    lv_r = volumes.get("Right-Lateral-Ventricle", 0)
    if lv_l > 0 and lv_r > 0:
        vent_total = lv_l + lv_r
        global_metrics["total_lateral_ventricle_mm3"] = vent_total
        vent_li = (lv_l - lv_r) / (lv_l + lv_r)
                                                                                            
        if vent_total > 50_000:
            idx += 1
            r = mapper.map_text("lateral ventricle")
            region = r.region if r.success else AtlasRegion(
                canonical_name="Lateral Ventricles",
                hemisphere="bilateral", lobe="subcortical",
                atlas_label="LateralVentricle",
                mni_centroid=(0, 0, 14),
            )
            findings.append(make_finding(
                idx=idx,
                finding_type="mass_effect",
                severity="moderate" if vent_total < 80_000 else "severe",
                region=region,
                measurement=Measurement(
                    metric="total_lateral_ventricle_volume_mm3",
                    value=vent_total,
                    units="mm³",
                    reference_min=15_000.0, reference_max=35_000.0,
                    normative_source="population_estimate",
                ),
                clinical_significance=(
                    f"Enlarged lateral ventricles ({vent_total:.0f} mm³ > 35,000 mm³ ref). "
                    "Suggests obstructive or communicating hydrocephalus; "
                    "evaluate for mass-effect CSF obstruction."
                ),
                evidence_chain=[
                    f"Left lateral ventricle: {lv_l:.0f} mm³",
                    f"Right lateral ventricle: {lv_r:.0f} mm³",
                    f"Total: {vent_total:.0f} mm³ (ref 15,000-35,000 mm³)",
                ],
            ))

    cc_parts = ["CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"]
    cc_vol = sum(volumes.get(p, 0) for p in cc_parts)
    if cc_vol > 0:
        global_metrics["corpus_callosum_total_mm3"] = cc_vol

    lobar_summary: dict[str, float] = {}
    for name, vol in volumes.items():
        r = mapper.map_freesurfer(name)
        if r.success:
            lobe = r.region.lobe
            lobar_summary[lobe] = lobar_summary.get(lobe, 0) + vol
    global_metrics["lobar_volume_mm3"] = {k: round(v, 0) for k, v in lobar_summary.items()}

    return findings, global_metrics

def _analyze_stroke(
    volumes: dict[str, float],
    icv: float,
    mapper: AtlasMapper,
    lesion_mask_path: Optional[str] = None,
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    idx = 0
    global_metrics: dict = {}

    if lesion_mask_path:
        mask_metrics = _analyze_lesion_mask(lesion_mask_path, "lesion")
        global_metrics.update(mask_metrics)
        if mask_metrics.get("lesion_total_voxels", 0) > 0:
            cx = mask_metrics.get("lesion_centroid_x_mni")
            n_l = mask_metrics.get("lesion_left_voxels", 0)
            n_r = mask_metrics.get("lesion_right_voxels", 0)
            dom = "left" if (cx is not None and cx < 0) else "right"
            idx += 1
            r = mapper.map_text(f"{dom} hemisphere")
            region = r.region if r.success else AtlasRegion(
                canonical_name=f"{dom.capitalize()} Cerebral Hemisphere",
                hemisphere=dom, lobe="frontal",
                atlas_label=f"Hemisphere_{dom[0].upper()}",
                mni_centroid=(-30.0 if dom == "left" else 30.0, 0, 20),
            )
            findings.append(make_finding(
                idx=idx,
                finding_type="lesion_localization",
                severity="severe",
                region=region,
                measurement=Measurement(
                    metric="lesion_centroid_x_mni",
                    value=cx if cx is not None else 0.0,
                    units="mm",
                    reference_min=-1.0, reference_max=1.0,
                    normative_source="atlas_v2_lesion_mask",
                ),
                clinical_significance=(
                    f"Stroke lesion centroid at MNI x={cx:+.1f} mm "
                    f"({n_l} voxels left of midline, {n_r} right) -> "
                    f"{dom}-hemisphere stroke."
                ),
                evidence_chain=[
                    f"Lesion mask: {lesion_mask_path}",
                    f"Total lesion voxels: {mask_metrics['lesion_total_voxels']}",
                    f"L vs R voxels: {n_l} / {n_r}",
                    f"Mean MNI x: {cx:+.2f} mm (negative = left-hemisphere stroke)",
                ],
            ))

    total_l = 0.0
    total_r = 0.0
    for k, v in volumes.items():
        if not isinstance(v, (int, float)) or v <= 0: continue
        kl = k.lower()
        if kl.startswith("left-"):     total_l += v
        elif kl.startswith("right-"):  total_r += v
        elif kl.startswith("ctx-lh-"): total_l += v
        elif kl.startswith("ctx-rh-"): total_r += v
    if total_l + total_r > 0:
        global_metrics["total_li"] = round((total_l - total_r) / (total_l + total_r), 4)
        global_metrics["total_left_mm3"]  = round(total_l)
        global_metrics["total_right_mm3"] = round(total_r)

    paired = [
        ("Left-Cerebral-White-Matter",  "Right-Cerebral-White-Matter"),
        ("Left-Hippocampus",            "Right-Hippocampus"),
        ("Left-Caudate",                "Right-Caudate"),
        ("Left-Putamen",                "Right-Putamen"),
        ("Left-Thalamus",               "Right-Thalamus"),
    ]

    for left_name, right_name in paired:
        lv = volumes.get(left_name, 0)
        rv = volumes.get(right_name, 0)
        if lv <= 0 or rv <= 0:
            continue

        li = (lv - rv) / (lv + rv)                                                    
        global_metrics[f"LI_{left_name.replace('Left-','')}"] = round(li, 4)

        if abs(li) > 0.08:
            deficit_side = "right" if li > 0 else "left"
            sev = "mild" if abs(li) < 0.15 else ("moderate" if abs(li) < 0.30 else "severe")
            struct_name = left_name.replace("Left-", "")

            r = mapper.map_freesurfer(left_name if deficit_side == "left" else right_name)
            region = r.region if r.success else AtlasRegion(
                canonical_name=f"{deficit_side.capitalize()} {struct_name}",
                hemisphere=deficit_side, lobe="subcortical",
                atlas_label=struct_name,
                mni_centroid=(0, 0, 0),
            )

            territory = region.stroke_territory if r.success else "Unknown vascular territory"
            idx += 1
            findings.append(make_finding(
                idx=idx,
                finding_type="lesion",
                severity=sev,
                region=region,
                measurement=Measurement(
                    metric="lateralisation_index",
                    value=abs(li),
                    units="dimensionless",
                    reference_min=0.0, reference_max=0.08,
                    normative_source="bilateral_symmetry",
                ),
                clinical_significance=(
                    f"{deficit_side.capitalize()} {struct_name} volume reduced "
                    f"(LI={li:+.3f}). Consistent with ischaemic lesion. "
                    f"Vascular territory: {territory[:120]}..."
                    if len(territory) > 120 else
                    f"{deficit_side.capitalize()} {struct_name} volume reduced "
                    f"(LI={li:+.3f}). {territory}"
                ),
                evidence_chain=[
                    f"Left {struct_name}: {lv:.0f} mm³",
                    f"Right {struct_name}: {rv:.0f} mm³",
                    f"Lateralisation index: {li:+.4f} (>±0.08 flagged)",
                    f"Territory: {territory[:200]}",
                ],
            ))

    bs_vol = volumes.get("Brain-Stem", 0)
    if bs_vol > 0:
        global_metrics["brainstem_vol_mm3"] = bs_vol
                                                 
        if bs_vol < 18_000:
            idx += 1
            r = mapper.map_freesurfer("Brain-Stem")
            findings.append(make_finding(
                idx=idx,
                finding_type="lesion",
                severity="moderate",
                region=r.region,
                measurement=Measurement(
                    metric="volume_mm3",
                    value=bs_vol,
                    units="mm³",
                    reference_min=18_000.0, reference_max=28_000.0,
                    normative_source="population_estimate",
                ),
                clinical_significance=(
                    f"Brainstem volume reduced ({bs_vol:.0f} mm³ < 18,000 mm³). "
                    "Consistent with basilar territory infarct or brainstem atrophy."
                ),
                evidence_chain=[
                    f"Brainstem volume: {bs_vol:.0f} mm³",
                    "Reference range: 18,000-28,000 mm³",
                    "Basilar artery territory; consider Wallenberg syndrome if PICA involved.",
                ],
            ))

    return findings, global_metrics

def _analyze_adhd(
    volumes: dict[str, float],
    icv: float,
    mapper: AtlasMapper,
    subject_id: str = "",
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    idx = 0
    global_metrics: dict = {}

    if icv <= 0:
        icv = _ICV_REF
    icv_ratio = _ICV_REF / icv                     

    icv_corrected: dict[str, float] = {}
    norms_for_subj = _norms_for_subject(subject_id)
    site = _site_from_subject_id(subject_id)
    norms_source = (
        f"adhd200_controls/{site}" if site and _EMPIRICAL_NORMS
        and _EMPIRICAL_NORMS.get("norms", {}).get(site)
        else "adhd200_controls/GLOBAL" if _EMPIRICAL_NORMS
        else "Hoogman_2017_estimate"
    )
    global_metrics["norms_source"] = norms_source

    for struct, (norm_mean, norm_sd) in norms_for_subj.items():
        raw_vol = volumes.get(struct, 0)
        if raw_vol <= 0:
            continue

        corrected = raw_vol * icv_ratio
        icv_corrected[struct] = corrected
        z = (corrected - norm_mean) / norm_sd
        global_metrics[f"z_{struct}"] = round(z, 3)

        sev = _z_to_severity(z)
        if sev == "normal":
            continue

        r = mapper.map_freesurfer(struct)
        region = r.region if r.success else AtlasRegion(
            canonical_name=struct.replace("-", " "),
            hemisphere="left" if struct.startswith("Left") else "right",
            lobe="subcortical",
            atlas_label=struct,
            mni_centroid=(0, 0, 0),
        )

        direction = "reduced" if z < 0 else "enlarged"
        adhd_note = region.adhd_relevance if r.success else ""

        idx += 1
        findings.append(make_finding(
            idx=idx,
            finding_type="atrophy" if z < 0 else "structural_anomaly",
            severity=sev,
            region=region,
            measurement=Measurement(
                metric="icv_corrected_volume_z",
                value=round(z, 3),
                units="z-score",
                reference_min=-1.0, reference_max=1.0,
                normative_source=norms_source,
            ),
            clinical_significance=(
                f"{region.canonical_name} volume {direction} (z={z:+.2f} vs same-site "
                f"ADHD-200 controls). Note: ADHD-related subcortical effect sizes are "
                f"small (Hoogman 2017 d=-0.11 to -0.19); a single |z|>=1.5 is within "
                f"expected control variation."
            ),
            evidence_chain=[
                f"Raw volume: {raw_vol:.0f} mm³",
                f"ICV: {icv:.0f} mm³ (correction factor: {icv_ratio:.3f})",
                f"ICV-corrected: {corrected:.0f} mm³",
                f"Normative mean ± SD: {norm_mean:.0f} ± {norm_sd:.0f} mm³",
                f"z-score: {z:+.3f}",
            ],
        ))

    if "Left-Caudate" in icv_corrected and "Right-Caudate" in icv_corrected:
        lc = icv_corrected["Left-Caudate"]
        rc = icv_corrected["Right-Caudate"]
        li = (lc - rc) / (lc + rc)
        global_metrics["caudate_lateralisation_index"] = round(li, 4)

    if "Left-Putamen" in icv_corrected and "Right-Putamen" in icv_corrected:
        lp = icv_corrected["Left-Putamen"]
        rp = icv_corrected["Right-Putamen"]
        li_p = (lp - rp) / (lp + rp)
        global_metrics["putamen_lateralisation_index"] = round(li_p, 4)

    global_metrics["icv_corrected_volumes"] = {
        k: round(v, 0) for k, v in icv_corrected.items()
    }

    return findings, global_metrics

class StructuralMRIAgent:

    CONDITION_DISPATCH = {
        "tumor":  _analyze_tumor,
        "stroke": _analyze_stroke,
        "adhd":   _analyze_adhd,
    }

    def __init__(self, condition: Condition):
        if condition not in self.CONDITION_DISPATCH:
            raise ValueError(f"Unknown condition: {condition}. "
                             "Must be 'tumor', 'stroke', or 'adhd'.")
        self.condition = condition
        self.mapper = AtlasMapper()

    def run(
        self,
        subject_id: str,
        stats_dir: str | Path,
        seg_path: Optional[str | Path] = None,
        lesion_mask_path: Optional[str | Path] = None,
    ) -> AgentOutput:
        t0 = time.time()
        stats_dir = Path(stats_dir)

        stats_subdir = stats_dir / "stats" if (stats_dir / "stats").exists() else stats_dir
        stats_file = _find_stats_file(stats_subdir)

        if stats_file is None:
            return AgentOutput.failed(
                "structural_mri", self.condition, subject_id,
                f"No stats file found in {stats_subdir}",
            )

        volumes, icv = _parse_stats_file(stats_file)
        if not volumes:
            return AgentOutput.failed(
                "structural_mri", self.condition, subject_id,
                f"Empty or unparseable stats file: {stats_file}",
            )

        analyze_fn = self.CONDITION_DISPATCH[self.condition]
        if self.condition == "adhd":
            findings, global_metrics = analyze_fn(volumes, icv, self.mapper, subject_id)
        elif self.condition == "tumor":
            findings, global_metrics = analyze_fn(volumes, icv, self.mapper,
                                                   str(seg_path) if seg_path else None)
        elif self.condition == "stroke":
            findings, global_metrics = analyze_fn(volumes, icv, self.mapper,
                                                   str(lesion_mask_path) if lesion_mask_path else None)
        else:
            findings, global_metrics = analyze_fn(volumes, icv, self.mapper)

        elapsed = time.time() - t0

        warnings = []
        if icv <= 0:
            warnings.append("ICV not found in stats file; used population reference for z-scores.")

        return AgentOutput(
            agent_name="structural_mri",
            condition=self.condition,
            subject_id=subject_id,
            status="success" if findings is not None else "partial",
            findings=findings,
            global_metrics=global_metrics,
            warnings=warnings,
            metadata={
                "stats_file": str(stats_file),
                "n_structures_parsed": len(volumes),
                "icv_mm3": icv,
                "processing_time_s": round(elapsed, 2),
            },
        )

    def run_batch(
        self,
        subjects: list[tuple[str, str | Path]],
    ) -> list[AgentOutput]:
        return [self.run(sid, sdir) for sid, sdir in subjects]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Structural MRI Agent")
    parser.add_argument("--condition",  required=True, choices=["tumor", "stroke", "adhd"])
    parser.add_argument("--subject",    required=True, help="Subject ID")
    parser.add_argument("--stats-dir",  required=True, help="Path to FastSurfer subject output dir")
    parser.add_argument("--out",        default=None,  help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    agent = StructuralMRIAgent(condition=args.condition)
    output = agent.run(subject_id=args.subject, stats_dir=args.stats_dir)

    out_json = output.to_json()
    if args.out:
        Path(args.out).write_text(out_json)
        print(f"Written to {args.out}")
    else:
        print(out_json)

    if output.findings:
        print("\n--- Finding Summary ---")
        print(output.finding_summary())
