from __future__ import annotations

import argparse
import json
import time
import warnings as _warnings
from pathlib import Path
from typing import Optional

import numpy as np

from agents.schemas import (
    AgentOutput, AtlasRegion, Measurement, Finding,
    make_finding, Condition,
)
from agents.atlas_mapper_agent import AtlasMapper

_SEEDS = {
    "PCC":       (0.0,  -52.0,  26.0),                               
    "mPFC":      (0.0,   54.0,  -6.0),                             
    "Precuneus": (0.0,  -56.0,  38.0),              
    "L.ANG":     (-46.0, -66.0, 32.0),                       
    "R.ANG":     (46.0,  -66.0, 32.0),                        
    "ACC":       (0.0,   28.0,  20.0),                              
    "R.INS":     (36.0,   0.0,   4.0),                          
    "L.INS":     (-36.0,  0.0,   4.0),                         
    "L.DLPFC":   (-44.0, 36.0,  28.0),               
    "R.DLPFC":   (44.0,  36.0,  28.0),                
    "L.CAU":     (-14.0,  8.0,  14.0),                 
    "R.CAU":     (14.0,   8.0,  14.0),                  
    "R.IFG":     (52.0,  18.0,  10.0),                                 
}

_FALLBACK_NORMS: dict[str, tuple[float, float]] = {
    "PCC_mPFC":         (0.62, 0.14),
    "PCC_Precuneus":    (0.55, 0.16),
    "ACC_R.INS":        (0.42, 0.15),
    "L.DLPFC_R.DLPFC":  (0.38, 0.14),
    "L.DLPFC_L.CAU":    (0.25, 0.12),
    "R.DLPFC_R.CAU":    (0.25, 0.12),
    "R.IFG_R.CAU":      (0.20, 0.10),
}

_FMRI_NORMS_PATH = Path(__file__).resolve().parent.parent / "data" / "adhd200_fmri_norms.json"
_KNOWN_SITES = ("Peking_1", "Peking_2", "Peking_3", "KKI", "NYU", "OHSU",
                "NeuroIMAGE", "Pittsburgh", "Brown", "WashU")

def _load_empirical_fmri_norms() -> dict | None:
    if not _FMRI_NORMS_PATH.exists():
        return None
    try:
        d = json.loads(_FMRI_NORMS_PATH.read_text())
        if d.get("norms") and any(d["norms"].get("GLOBAL", {}).values()):
            return d
        return None
    except Exception:
        return None

_EMPIRICAL_FMRI_NORMS = _load_empirical_fmri_norms()

def _fmri_site_from_subject_id(subject_id: str) -> str | None:
    for site in _KNOWN_SITES:
        if subject_id.startswith(site + "_") or subject_id == site:
            return site
    if "_sub-" in subject_id:
        return subject_id.split("_sub-")[0]
    return None

def _norms_for_subject(subject_id: str) -> dict[str, tuple[float, float]]:
    if _EMPIRICAL_FMRI_NORMS:
        all_sites = _EMPIRICAL_FMRI_NORMS.get("norms", {})
        site = _fmri_site_from_subject_id(subject_id)
        site_norms = all_sites.get(site) if site else None
        if not site_norms:
            site_norms = all_sites.get("GLOBAL")
        if site_norms:
            out: dict[str, tuple[float, float]] = {}
            for k, d in site_norms.items():
                if d.get("sd", 0) > 0:
                    out[k] = (float(d["mean"]), float(d["sd"]))
            for k, mu_sd in _FALLBACK_NORMS.items():
                out.setdefault(k, mu_sd)
            return out
    return dict(_FALLBACK_NORMS)

_NORMS = dict(_FALLBACK_NORMS)
if _EMPIRICAL_FMRI_NORMS and _EMPIRICAL_FMRI_NORMS.get("norms", {}).get("GLOBAL"):
    _NORMS = {
        k: (float(d["mean"]), float(d["sd"]))
        for k, d in _EMPIRICAL_FMRI_NORMS["norms"]["GLOBAL"].items()
        if d.get("sd", 0) > 0
    }

def _z_score(r: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return 0.0
    return (r - mean) / sd

def _z_to_severity(z: float) -> str:
    az = abs(z)
    if az < 1.5:
        return "normal"
    if az < 2.0:
        return "mild"
    if az < 2.5:
        return "moderate"
    return "severe"

_CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    "white_matter", "csf",
]

def _load_confounds(tsv_path: Path) -> Optional[np.ndarray]:
    try:
        import pandas as pd
        df = pd.read_csv(tsv_path, sep="\t")
        cols = [c for c in _CONFOUND_COLS if c in df.columns]
        if not cols:
            return None
        conf = df[cols].fillna(0).values
                                                                 
        return conf.astype(float)
    except Exception:
        return None

def _extract_seed_timeseries(
    bold_path: Path,
    seed_mni: dict[str, tuple],
    confounds: Optional[np.ndarray],
    smoothing_fwhm: float = 6.0,
    t_r: float = 2.0,
    low_pass: float = 0.1,
    high_pass: float = 0.01,
) -> tuple[Optional[dict[str, np.ndarray]], Optional[str]]:
    try:
        from nilearn.maskers import NiftiSpheresMasker

        seeds = list(seed_mni.values())
        seed_names = list(seed_mni.keys())

        masker = NiftiSpheresMasker(
            seeds=seeds,
            radius=8,
            allow_overlap=True,
            detrend=True,
            standardize=True,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            smoothing_fwhm=smoothing_fwhm,
            verbose=0,
        )

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            if confounds is not None:
                ts = masker.fit_transform(str(bold_path), confounds=confounds)
            else:
                ts = masker.fit_transform(str(bold_path))

        return {name: ts[:, i] for i, name in enumerate(seed_names)}, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def _not_applicable(condition: str) -> tuple[list, dict]:
    return [], {"note": f"Functional fMRI analysis not applicable for condition '{condition}'."}

def _analyze_adhd_fmri(
    seed_ts: dict[str, np.ndarray],
    mapper: AtlasMapper,
    subject_id: str = "",
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    idx = 0
    global_metrics: dict = {}
    norms_for_subj = _norms_for_subject(subject_id)
    site = _fmri_site_from_subject_id(subject_id)
    norms_source = (
        f"adhd200_controls/{site}" if (site and _EMPIRICAL_FMRI_NORMS
        and _EMPIRICAL_FMRI_NORMS.get("norms", {}).get(site))
        else "adhd200_controls/GLOBAL" if _EMPIRICAL_FMRI_NORMS
        else "HCP_ADHD200_estimate"
    )
    global_metrics["norms_source"] = norms_source

    pairs = [
        ("PCC",    "mPFC",    "PCC_mPFC",        "DMN core coupling"),
        ("PCC",    "Precuneus","PCC_Precuneus",   "DMN posterior hub"),
        ("ACC",    "R.INS",   "ACC_R.INS",        "Salience network"),
        ("L.DLPFC","R.DLPFC", "L.DLPFC_R.DLPFC", "Bilateral executive network"),
        ("L.DLPFC","L.CAU",   "L.DLPFC_L.CAU",   "Left frontostriatal"),
        ("R.DLPFC","R.CAU",   "R.DLPFC_R.CAU",   "Right frontostriatal"),
        ("R.IFG",  "R.CAU",   "R.IFG_R.CAU",     "Right IFG-caudate inhibition circuit"),
    ]

    region_cache: dict[str, AtlasRegion] = {}

    def get_region(seed_name: str) -> AtlasRegion:
        if seed_name not in region_cache:
            coord = _SEEDS.get(seed_name, (0, 0, 0))
            r = mapper.map_mni(*coord)
            region_cache[seed_name] = r.region if r.success else AtlasRegion(
                canonical_name=seed_name,
                hemisphere="bilateral", lobe="frontal",
                atlas_label=seed_name,
                mni_centroid=coord,
            )
        return region_cache[seed_name]

    for seed_a, seed_b, norm_key, circuit_name in pairs:
        if seed_a not in seed_ts or seed_b not in seed_ts:
            global_metrics[f"r_{norm_key}"] = None
            continue

        r_val = _pearson_r(seed_ts[seed_a], seed_ts[seed_b])
        global_metrics[f"r_{norm_key}"] = round(r_val, 4)

        if norm_key not in norms_for_subj:
            continue

        norm_mean, norm_sd = norms_for_subj[norm_key]
        z = _z_score(r_val, norm_mean, norm_sd)
        global_metrics[f"z_{norm_key}"] = round(z, 3)
        sev = _z_to_severity(z)

        if sev == "normal":
            continue

        region_a = get_region(seed_a)
        region_b = get_region(seed_b)

        direction = "reduced" if z < 0 else "elevated"

        idx += 1
        findings.append(make_finding(
            idx=idx,
            finding_type="connectivity_deficit" if z < 0 else "connectivity_excess",
            severity=sev,
            region=region_a,
            measurement=Measurement(
                metric="pearson_r",
                value=round(r_val, 4),
                units="r",
                reference_min=norm_mean - norm_sd,
                reference_max=norm_mean + norm_sd,
                normative_source=norms_source,
            ),
            clinical_significance=(
                f"{circuit_name} ({seed_a}-{seed_b}) connectivity {direction} "
                f"vs same-site ADHD-200 controls (r={r_val:.3f}, z={z:+.2f}). "
                f"A single |z|>=1.5 deviation is within control-group variation; "
                f"ADHD interpretation requires convergent evidence across circuits."
            ),
            evidence_chain=[
                f"Seed A: {seed_a} at MNI {_SEEDS.get(seed_a)}",
                f"Seed B: {seed_b} at MNI {_SEEDS.get(seed_b)}",
                f"Pearson r: {r_val:.4f}",
                f"Normative: mean={norm_mean:.3f}, SD={norm_sd:.3f}",
                f"z-score: {z:+.3f}",
                f"Atlas grounding: {region_a.canonical_name}",
            ],
        ))

    pcc_mpfc_r = global_metrics.get("r_PCC_mPFC")
    if pcc_mpfc_r is not None and "PCC_mPFC" in norms_for_subj:
        norm_mean, norm_sd = norms_for_subj["PCC_mPFC"]
        if norm_sd > 0:
            global_metrics["dmn_deactivation_failure_index"] = round(
                (pcc_mpfc_r - norm_mean) / norm_sd, 3
            )

    return findings, global_metrics

def _adhd_interpretation(norm_key: str, z: float) -> str:
    interpretations = {
        "PCC_mPFC": (
            "Reduced PCC-mPFC coupling is the canonical fMRI signature of ADHD "
            "(Castellanos et al. 2008). Indicates Default Mode Network fragmentation, "
            "correlating with task-induced mind-wandering and inattentive symptoms."
            if z < 0 else
            "Elevated PCC-mPFC coupling suggests Default Mode Network hypercoherence, "
            "consistent with failure to suppress DMN during cognitively demanding tasks."
        ),
        "PCC_Precuneus": (
            "Reduced PCC-Precuneus connectivity reflects disrupted DMN posterior hub. "
            "Correlates with episodic memory and self-referential processing deficits in ADHD."
            if z < 0 else
            "Elevated PCC-Precuneus connectivity may indicate hyperactive default mode state."
        ),
        "ACC_R.INS": (
            "Reduced salience network (ACC-right insula) connectivity is linked to "
            "impaired error monitoring (reduced ERN), reduced conflict detection, "
            "and emotional dysregulation in ADHD."
            if z < 0 else
            "Elevated salience network connectivity; atypical salience processing."
        ),
        "L.DLPFC_R.DLPFC": (
            "Reduced bilateral DLPFC coupling reflects disrupted executive network "
            "interhemispheric connectivity. Linked to working memory and planning deficits."
            if z < 0 else "Elevated bilateral DLPFC coupling; possible compensatory response."
        ),
        "L.DLPFC_L.CAU": (
            "Reduced left frontostriatal coupling. Frontostriatal circuit (DLPFC-caudate) "
            "dysfunction is the core neurobiological model of ADHD. "
            "Correlates with inattention severity (Castellanos & Tannock 2002)."
            if z < 0 else "Elevated left frontostriatal coupling."
        ),
        "R.DLPFC_R.CAU": (
            "Reduced right frontostriatal coupling. Right DLPFC-caudate circuit critical "
            "for response inhibition. Reduced coupling correlates with stop-signal deficits."
            if z < 0 else "Elevated right frontostriatal coupling."
        ),
        "R.IFG_R.CAU": (
            "Reduced right IFG-caudate connectivity. This circuit is the primary substrate "
            "of response inhibition (stop-signal task). Hypoconnectivity here is the "
            "strongest neuroimaging predictor of impulsivity in ADHD "
            "(Aron & Poldrack 2006; Rubia et al. 2010)."
            if z < 0 else "Elevated right IFG-caudate connectivity."
        ),
    }
    return interpretations.get(norm_key, "Abnormal functional connectivity.")

class FunctionalFMRIAgent:

    def __init__(self, condition: Condition):
        self.condition = condition
        self.mapper = AtlasMapper()

    def run(
        self,
        subject_id: str,
        bold_path: str | Path,
        confounds_path: Optional[str | Path] = None,
        t_r: float = 2.0,
    ) -> AgentOutput:
        t0 = time.time()
        bold_path = Path(bold_path)

        if self.condition in ("tumor", "stroke"):
            return AgentOutput(
                agent_name="functional_fmri",
                condition=self.condition,
                subject_id=subject_id,
                status="success",
                findings=[],
                global_metrics={"note": f"Functional fMRI not available for '{self.condition}' condition."},
                warnings=["No resting-state fMRI in BraTS-GLI or ATLAS-v2 datasets."],
            )

        if not bold_path.exists():
            return AgentOutput.failed(
                "functional_fmri", self.condition, subject_id,
                f"BOLD file not found: {bold_path}",
            )

        confounds = None
        if confounds_path and Path(confounds_path).exists():
            confounds = _load_confounds(Path(confounds_path))

        warnings_list = []
        if confounds is None:
            warnings_list.append("Confounds file not found or empty; running without denoising.")

        json_sidecar = bold_path.with_suffix("").with_suffix(".json")
        if json_sidecar.exists():
            try:
                meta = json.loads(json_sidecar.read_text())
                t_r = float(meta.get("RepetitionTime", t_r))
            except Exception:
                pass

        seed_ts, extract_err = _extract_seed_timeseries(
            bold_path=bold_path,
            seed_mni=_SEEDS,
            confounds=confounds,
            t_r=t_r,
        )

        if seed_ts is None:
            return AgentOutput.failed(
                "functional_fmri", self.condition, subject_id,
                f"Failed to extract seed time-series: {extract_err}",
            )

        findings, global_metrics = _analyze_adhd_fmri(seed_ts, self.mapper, subject_id)
        elapsed = time.time() - t0

        n_trs = len(next(iter(seed_ts.values()))) if seed_ts else 0
        global_metrics["n_timepoints"] = n_trs
        global_metrics["t_r_s"] = t_r

        return AgentOutput(
            agent_name="functional_fmri",
            condition=self.condition,
            subject_id=subject_id,
            status="success",
            findings=findings,
            global_metrics=global_metrics,
            warnings=warnings_list,
            metadata={
                "bold_path": str(bold_path),
                "confounds_used": confounds is not None,
                "n_seeds": len(seed_ts),
                "processing_time_s": round(elapsed, 2),
            },
        )

    def run_from_fmriprep_dir(
        self,
        subject_id: str,
        fmriprep_subj_dir: str | Path,
        site: str = "",
    ) -> AgentOutput:
        subj_dir = Path(fmriprep_subj_dir)

        bold_files = sorted(subj_dir.rglob("*desc-preproc_bold.nii.gz"))
        if not bold_files:
            return AgentOutput.failed(
                "functional_fmri", self.condition, subject_id,
                f"No preproc_bold.nii.gz found under {subj_dir}",
            )
        bold_path = bold_files[0]

        conf_files = sorted(subj_dir.rglob("*confounds_timeseries.tsv"))
        confounds_path = conf_files[0] if conf_files else None

        return self.run(
            subject_id=subject_id,
            bold_path=bold_path,
            confounds_path=confounds_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Functional fMRI Agent")
    parser.add_argument("--condition",  required=True, choices=["tumor", "stroke", "adhd"])
    parser.add_argument("--subject",    required=True, help="Subject ID")
    parser.add_argument("--bold",       required=True, help="Path to preproc_bold.nii.gz")
    parser.add_argument("--confounds",  default=None,  help="Path to confounds_timeseries.tsv")
    parser.add_argument("--tr",         type=float, default=2.0, help="TR in seconds")
    parser.add_argument("--out",        default=None,  help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    agent = FunctionalFMRIAgent(condition=args.condition)
    output = agent.run(
        subject_id=args.subject,
        bold_path=args.bold,
        confounds_path=args.confounds,
        t_r=args.tr,
    )

    out_json = output.to_json()
    if args.out:
        Path(args.out).write_text(out_json)
        print(f"Written to {args.out}")
    else:
        print(out_json)

    if output.findings:
        print("\n--- Finding Summary ---")
        print(output.finding_summary())
