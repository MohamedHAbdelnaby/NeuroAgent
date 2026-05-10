
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

HERE = Path(__file__).parent
SLICES_DIR = HERE / "data" / "slices"

SAMPLES = {
    "adhd": {
        "subject_id":      "demo_ADHD_001",
        "condition":       "ADHD",
        "age":             11,
        "sex":             "M",
        "site":            "ADHD-200 / KKI",
        "true_label_text": "ADHD (combined type)",
        "modalities":      ["T1w", "fMRI"],
    },
    "tumor": {
        "subject_id":      "demo_TUMOR_001",
        "condition":       "Brain Tumor (glioma)",
        "age":             56,
        "sex":             "F",
        "site":            "BraTS-GLI",
        "true_label_text": "Right hemisphere high-grade glioma (GBM-like)",
        "modalities":      ["T1w", "T1c", "Segmentation"],
    },
    "stroke": {
        "subject_id":      "demo_STROKE_001",
        "condition":       "Chronic Stroke",
        "age":             62,
        "sex":             "M",
        "site":            "ATLAS-v2",
        "true_label_text": "Left MCA territory chronic infarct",
        "modalities":      ["T1w"],
    },
}

def load_slice(subject: str, modality: str, axis: str) -> Image.Image | None:
    fp = SLICES_DIR / f"{subject}_{modality}_{axis}.png"
    if fp.exists():
        return Image.open(fp)
    return None

def synthetic_slice() -> Image.Image:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[-128:128, -128:128]
    r2 = xx**2 + yy**2
    skull  = (xx**2 / 100**2 + yy**2 / 110**2) < 1.0
    cortex = ((r2 > 7000) & (r2 < 11500)) & skull
    inner  = (r2 < 5500) & skull
    img = np.zeros_like(skull, dtype=np.float32)
    img[skull]  = 0.35 + rng.normal(0, 0.05, size=skull.sum())
    img[cortex] = 0.85 + rng.normal(0, 0.05, size=cortex.sum())
    img[inner]  = 0.55 + rng.normal(0, 0.05, size=inner.sum())
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8), mode="L").convert("RGB")

SAMPLE_FASTSURFER = {
    "adhd": """# ColHeaders  Index SegId NVoxels Volume_mm3 StructName
  1   2     156421  156421.0  Left-Cerebral-White-Matter
  2   4       7820    7820.0  Left-Lateral-Ventricle
  3  10       7642    7642.0  Left-Thalamus
  4  11       3204    3204.0  Left-Caudate           # below normal range
  5  12       4123    4123.0  Left-Putamen           # below normal range
  6  13       1622    1622.0  Left-Pallidum
  7  17       3812    3812.0  Left-Hippocampus
  8  18       1430    1430.0  Left-Amygdala
  9  41     157983  157983.0  Right-Cerebral-White-Matter
 10  43       7560    7560.0  Right-Lateral-Ventricle
 11  49       7711    7711.0  Right-Thalamus
 12  50       3570    3570.0  Right-Caudate          # below normal range
 13  51       4280    4280.0  Right-Putamen          # mildly reduced
 14  52       1574    1574.0  Right-Pallidum
 15  53       3905    3905.0  Right-Hippocampus
 16  54       1572    1572.0  Right-Amygdala
# Measure ICV, Intracranial volume, 1452103.0, mm^3
# Measure BrainSeg, Brain Segmentation Volume, 1126540.0, mm^3
# Measure SubCortGray, Subcortical gray matter volume, 52997.0, mm^3""",
    "tumor": """# ColHeaders  Index SegId NVoxels Volume_mm3 StructName
  1   2     142100  142100.0  Left-Cerebral-White-Matter   # contralateral, preserved
  2   4       9120    9120.0  Left-Lateral-Ventricle
  3  10       8211    8211.0  Left-Thalamus
  4  11       4015    4015.0  Left-Caudate
  5  17       4090    4090.0  Left-Hippocampus
  6  41      71540   71540.0  Right-Cerebral-White-Matter   # *** SEVERE deficit ***
  7  43      24800   24800.0  Right-Lateral-Ventricle       # *** dilated (mass effect) ***
  8  49       5102    5102.0  Right-Thalamus                # compressed
  9  50       2110    2110.0  Right-Caudate                 # compressed
 10  53       2105    2105.0  Right-Hippocampus             # severely reduced
 # Right hemisphere shows extensive WM loss + ventricle dilation
 # consistent with a right temporo-parietal mass + edema + midline shift.
# Measure ICV,        1390000.0, mm^3
# Note: Cortical reconstruction PARTIAL on right side due to mass effect.""",
    "stroke": """# ColHeaders  Index SegId NVoxels Volume_mm3 StructName
  1   2      85320   85320.0  Left-Cerebral-White-Matter   # *** REDUCED (left MCA) ***
  2   4      12450   12450.0  Left-Lateral-Ventricle       # ex-vacuo enlargement
  3  10       6120    6120.0  Left-Thalamus                # mildly reduced
  4  11       2810    2810.0  Left-Caudate                 # reduced
  5  17       3020    3020.0  Left-Hippocampus
  6  41     154820  154820.0  Right-Cerebral-White-Matter  # preserved
  7  43       7820    7820.0  Right-Lateral-Ventricle
  8  49       7902    7902.0  Right-Thalamus
  9  50       4012    4012.0  Right-Caudate
 10  53       3990    3990.0  Right-Hippocampus
 # Lateralization Index (L-R)/(L+R) = -0.29  (significant left atrophy)
 # Pattern consistent with chronic left MCA infarct.
# Measure ICV,        1410000.0, mm^3""",
}

SAMPLE_TOOLS_OUTPUT = {
    "adhd": [
        ("run_structural_mri",
         {"stats_dir": "/preprocessing/fastsurfer_output/ADHD-200/demo_ADHD_001"},
         {
             "status": "success",
             "n_findings": 4,
             "findings": [
                 "[MODERATE] Left Caudate: volume 3204 mm^3 (z=-1.46 vs ICV-corrected normative), frontostriatal circuit",
                 "[MODERATE] Right Caudate: volume 3570 mm^3 (z=-1.31)",
                 "[MILD]     Left Putamen: volume 4123 mm^3 (z=-1.12)",
                 "[MILD]     Right Putamen: volume 4280 mm^3 (z=-0.98)",
             ],
             "global_metrics": {"icv_mm3": 1452103, "n_structures_parsed": 100,
                                "z_caudate_left": -1.46, "z_caudate_right": -1.31}
         }),
        ("run_functional_fmri",
         {"bold_path": "/preprocessing/fmriprep/.../bold.nii.gz"},
         {
             "status": "success",
             "n_findings": 2,
             "global_metrics": {
                 "r_PCC_mPFC":  0.62,  "z_PCC_mPFC": +1.84,
                 "r_DLPFC_caudate": 0.18, "z_DLPFC_caudate": -1.42,
                 "dmn_deactivation_failure_index": +0.72,
             },
             "findings": [
                 "[MODERATE] DMN hyperconnectivity (PCC<->mPFC z=+1.84), task-suppression failure marker",
                 "[MODERATE] Frontostriatal hypoconnectivity (DLPFC<->caudate z=-1.42)",
             ]
         }),
        ("map_atlas_region", {"label": "Left-Caudate"},
         {"canonical_name": "Caudate (Left)", "hemisphere": "left", "lobe": "subcortical",
          "atlas_label": "AAL: Caudate_L", "mni_centroid": [-12, 9, 8],
          "networks": ["frontostriatal", "salience"], "adhd_relevance":
          "Core ADHD biomarker (Hoogman 2017). 5-8% volume reduction in childhood ADHD."}),
        ("retrieve_clinical_knowledge",
         {"query": "ADHD caudate volume reduction frontostriatal circuit", "top_k": 3},
         {"docs": [
             {"title": "Subcortical brain volume differences in ADHD: a mega-analysis",
              "authors": "Hoogman et al.", "year": 2017, "journal": "Lancet Psychiatry", "score": 0.91},
             {"title": "Default mode network and frontostriatal connectivity in ADHD",
              "authors": "Castellanos & Aoki", "year": 2016, "journal": "Biol Psychiatry", "score": 0.84},
             {"title": "Resting-state fMRI biomarkers in pediatric ADHD",
              "authors": "Posner et al.", "year": 2014, "journal": "Brain Imaging Behav", "score": 0.79},
         ]}),
        ("generate_report",
         {"reasoning_summary":
            "Subcortical findings (bilateral caudate z<-1.3, putamen mild reduction) align with "
            "Hoogman 2017 ADHD pattern. Functional fMRI shows BOTH DMN over-engagement (failure "
            "to deactivate) and frontostriatal underconnectivity, convergent multi-modal evidence "
            "for ADHD phenotype."},
         {"confidence": 0.78, "report_pages": 3}),
    ],
    "tumor": [
        ("run_structural_mri", {"stats_dir": "/.../BraTS-GLI-00005-100"},
         {"status": "partial", "n_findings": 5, "global_metrics":
          {"hemispheric_LI": +0.31, "right_white_matter_loss_pct": 55,
           "right_ventricle_dilation_x": 3.1, "midline_shift_mm": 8.2,
           "icv_mm3": 1390000, "cortical_reconstruction": "failed-on-right"},
          "findings": [
              "[SEVERE]   Right WM volume 71,540 mm^3, 55% reduction vs left (mass + edema)",
              "[SEVERE]   Right lateral ventricle DILATED 3.1x (mass effect -> CSF redistribution)",
              "[MODERATE] Midline shift ~8 mm (right->left), high-grade lesion likely",
              "[SEVERE]   Right hippocampus 2,105 mm^3, severe medial-temporal compression",
              "[WARNING]  FastSurfer reconstruction PARTIAL on right hemisphere",
          ]}),
        ("map_atlas_region", {"label": "Right-Temporal-Lobe"},
         {"canonical_name": "Right Temporal Lobe", "hemisphere": "right", "lobe": "temporal",
          "atlas_label": "AAL: Temporal_Sup_R, Temporal_Mid_R", "mni_centroid": [55, -10, -10],
          "tumor_relevance": "Common GBM location. Eloquent: Wernicke region (left) NOT involved. "
                             "Surgical considerations: optic radiations, hippocampus."}),
        ("retrieve_clinical_knowledge", {"query": "right temporal high-grade glioma mass effect", "top_k": 3},
         {"docs": [
             {"title": "Imaging features of glioblastoma: contrast enhancement and necrosis",
              "authors": "Pope et al.", "year": 2018, "journal": "Neuro-Oncology", "score": 0.93},
             {"title": "Mass effect and midline shift in supratentorial tumors",
              "authors": "Steiger & Pohlhammer", "year": 2015, "journal": "J Neurosurg", "score": 0.86},
             {"title": "BraTS challenge: tumor sub-region segmentation benchmarks",
              "authors": "Bakas et al.", "year": 2019, "journal": "ArXiv", "score": 0.81},
         ]}),
        ("generate_report",
         {"reasoning_summary":
          "Severe right-hemisphere mass effect (+0.31 LI), 55% WM loss, 8mm midline shift, "
          "ventricular dilation. Pattern consistent with high-grade right temporo-parietal glioma. "
          "Surgical urgency category. Suggest contrast MRI follow-up, neurosurgery consult."},
         {"confidence": 0.91, "report_pages": 4}),
    ],
    "stroke": [
        ("run_structural_mri", {"stats_dir": "/.../R019_sub-r019s003"},
         {"status": "success", "n_findings": 4, "global_metrics":
          {"hemispheric_LI": -0.29, "left_white_matter_loss_pct": 45,
           "icv_mm3": 1410000, "left_ventricle_enlarged": True},
          "findings": [
              "[SEVERE]   Left cerebral WM 85,320 mm^3, 45% reduction (chronic infarct atrophy)",
              "[MODERATE] Left lateral ventricle ENLARGED ex-vacuo (compensates atrophy)",
              "[MILD]     Left thalamus mildly reduced (z=-0.92), distal Wallerian degeneration",
              "[MILD]     Left caudate reduced (z=-1.04), subcortical involvement",
          ]}),
        ("map_atlas_region", {"label": "Left-MCA-territory"},
         {"canonical_name": "Left Middle Cerebral Artery Territory",
          "hemisphere": "left", "lobe": "frontotemporoparietal",
          "atlas_label": "Vascular: Left MCA",
          "mni_centroid": [-40, -10, 20],
          "stroke_territory": "Left MCA, supplies lateral frontal/temporal/parietal cortex, "
                              "putamen, internal capsule. Common stroke location. "
                              "Clinical: contralateral hemiparesis, possible aphasia."}),
        ("retrieve_clinical_knowledge", {"query": "chronic left MCA stroke atrophy ex-vacuo", "top_k": 3},
         {"docs": [
             {"title": "Chronic stroke imaging: ex-vacuo dilation and Wallerian degeneration",
              "authors": "Saver", "year": 2017, "journal": "NEJM", "score": 0.89},
             {"title": "ATLAS dataset: large-scale stroke lesion segmentation",
              "authors": "Liew et al.", "year": 2018, "journal": "Sci Data", "score": 0.85},
             {"title": "MCA territory anatomy and stroke syndromes",
              "authors": "Caplan", "year": 2009, "journal": "Caplan's Stroke", "score": 0.78},
         ]}),
        ("generate_report",
         {"reasoning_summary":
          "Lateralization Index -0.29 (significant left atrophy), 45% L-WM loss, ex-vacuo "
          "ventricular dilation, distal subcortical involvement. Classic chronic left MCA "
          "territory infarct pattern. Recommend MR-angiography, secondary prevention review."},
         {"confidence": 0.88, "report_pages": 3}),
    ],
}

SAMPLE_REPORTS = {
    "adhd": """# Diagnostic Imaging Report: demo_ADHD_001

**Condition of interest:** ADHD (combined type)
**Subject:** 11 y/o male, KKI site
**Confidence:** 78%

---

## Key Findings
- Bilateral **caudate volume reduction** (z = -1.46 left, -1.31 right)
- Mild bilateral **putamen reduction**
- **DMN hyperconnectivity** (PCC<->mPFC z = +1.84), task-suppression failure
- **Frontostriatal hypoconnectivity** (DLPFC<->caudate z = -1.42)

## Assessment
Multi-modal convergent evidence consistent with ADHD phenotype.
Subcortical pattern aligns with Hoogman et al. (2017) meta-analysis;
functional pattern aligns with Castellanos & Aoki (2016) DMN model.

## Limitations
- Single-session resting-state, replication recommended
- Structural effect sizes are small population-level; not diagnostic at the individual level
- Clinical history, behavioral rating scales (Conners), and parent/teacher reports required

## References
1. Hoogman et al. (2017) *Lancet Psychiatry*
2. Castellanos & Aoki (2016) *Biol Psychiatry*
3. Posner et al. (2014) *Brain Imaging Behav*""",
    "tumor": """# Diagnostic Imaging Report: demo_TUMOR_001

**Condition of interest:** Brain tumor, right hemisphere
**Subject:** 56 y/o female
**Confidence:** 91%
**SURGICAL URGENCY**

---

## Key Findings
- **55% right white matter loss** vs left (mass + peri-tumoral edema)
- **Right lateral ventricle dilated 3.1x** (mass effect -> CSF redistribution)
- **Midline shift ~8 mm** (right -> left)
- **Right hippocampus** severely compressed (2,105 mm^3)
- FastSurfer reconstruction failed on right side, typical of large lesions

## Assessment
Findings consistent with **high-grade right temporo-parietal glioma** (GBM-like).
Mass effect and midline shift indicate surgical consideration.

## Recommendations
- Contrast MRI (T1c) to characterize enhancement pattern
- Neurosurgery consult, biopsy / resection planning
- DTI for tractography (corticospinal, optic radiations)
- Eloquent cortex mapping if peri-Wernicke involvement suspected (left side appears spared)

## References
1. Pope et al. (2018) *Neuro-Oncology*
2. Steiger & Pohlhammer (2015) *J Neurosurg*
3. Bakas et al. (2019) BraTS challenge""",
    "stroke": """# Diagnostic Imaging Report: demo_STROKE_001

**Condition of interest:** Chronic stroke, left MCA territory
**Subject:** 62 y/o male
**Confidence:** 88%

---

## Key Findings
- **Lateralization Index -0.29** (left atrophy)
- **45% left white matter reduction** (chronic infarct)
- **Left lateral ventricle ex-vacuo dilation** (compensates parenchymal loss)
- **Left thalamus / caudate** mildly reduced, distal Wallerian degeneration

## Assessment
Pattern is classic for chronic left MCA territory infarct.
Atrophy and ex-vacuo dilation consistent with established (>6 months) lesion.

## Recommendations
- MR angiography, assess residual stenosis
- Secondary stroke prevention review (BP, antiplatelet, lipids)
- Speech-language assessment if not already on file (left MCA -> aphasia risk)
- Functional outcome assessment (NIHSS, modified Rankin)

## References
1. Saver (2017) *NEJM*
2. Liew et al. (2018) *Sci Data*
3. Caplan (2009) *Caplan's Stroke*""",
}

st.set_page_config(page_title="NeuroAgent Demo", layout="wide")

st.sidebar.title("NeuroAgent Demo")
st.sidebar.markdown(
    "End-to-end walkthrough of the NeuroAgent multi-agent neuroimaging "
    "framework. **All data is sample / controlled** for demonstration."
)
st.sidebar.markdown("---")
choice = st.sidebar.radio("Demo subject:",
                          options=["adhd", "tumor", "stroke"],
                          format_func=lambda k: f"{SAMPLES[k]['condition']} - {SAMPLES[k]['subject_id']}")
st.sidebar.markdown("---")
st.sidebar.caption("Sample 3D MRI, FastSurfer outputs, agent traces, and the final "
                   "diagnostic report are all hardcoded for this walkthrough.")

s = SAMPLES[choice]

st.title(f"NeuroAgent: {s['condition']} demo")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Subject ID", s["subject_id"])
col_b.metric("Age / Sex", f"{s['age']} / {s['sex']}")
col_c.metric("Site", s["site"])
col_d.metric("Ground truth", s["true_label_text"], delta="positive label")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Imaging viewer",
    "FastSurfer output",
    "Agent pipeline",
    "Final report",
    "About"
])

with tab1:
    st.subheader("Raw imaging, orthogonal views")
    st.caption("Pre-rendered axial / coronal / sagittal slices of real MNI-space data, "
               "bundled in `data/slices/` for portable demo.")

    cols = st.columns(len(s["modalities"]))
    for col, modality in zip(cols, s["modalities"]):
        col.markdown(f"**{modality}**")
        for axis in ("axial", "coronal", "sagittal"):
            img = load_slice(choice, modality, axis) or synthetic_slice()
            col.image(img, caption=f"{axis.title()}", use_container_width=True)

with tab2:
    st.subheader("FastSurfer output (parsed)")
    st.caption("Truncated `aseg+DKT.VINN.stats` excerpt, the raw input to the structural-MRI agent.")
    st.code(SAMPLE_FASTSURFER[choice], language="python")
    st.info(
        "The structural-MRI agent reads this file, extracts per-region volumes, "
        "computes ICV-corrected z-scores against normative population data, and "
        "produces a list of findings (severity-tagged observations)."
    )

with tab3:
    st.subheader("Agentic tool-call trace")
    st.caption("Each step shows the tool call (with arguments) and its output JSON. "
               "In production, the orchestrator's LLM decides when to invoke each tool.")

    for i, (tool_name, args, result) in enumerate(SAMPLE_TOOLS_OUTPUT[choice], 1):
        with st.expander(f"**Step {i}: {tool_name}**", expanded=(i == 1)):
            st.markdown("**Input arguments:**")
            st.json(args)
            st.markdown("**Output:**")
            st.json(result)

    st.markdown("---")
    st.info(
        "The orchestrator now has all agent outputs to synthesize. The next step is "
        "the final structured-output call that classifies the subject and produces "
        "the diagnostic report."
    )

with tab4:
    st.subheader("Final diagnostic report")
    st.caption("Generated by the report-generator agent from the accumulated evidence chain.")

    md = SAMPLE_REPORTS[choice]
    st.markdown(md)
    st.markdown("---")
    cols = st.columns(3)
    cols[0].download_button("Download Markdown", data=md,
                            file_name=f"{s['subject_id']}_report.md")
    cols[1].download_button("Download JSON",
                            data=json.dumps({"subject_id": s["subject_id"],
                                             "condition":  s["condition"],
                                             "report_md":  md,
                                             "tools":      [t for t,_,_ in SAMPLE_TOOLS_OUTPUT[choice]]},
                                            indent=2),
                            file_name=f"{s['subject_id']}_report.json")
    cols[2].metric("Confidence",
                   f"{SAMPLE_TOOLS_OUTPUT[choice][-1][2]['confidence']*100:.0f}%")

with tab5:
    st.subheader("About this demo")
    st.markdown("""
**NeuroAgent** is a disease-agnostic multi-agent framework for neuroimaging analysis.

#### Architecture
1. **Structural MRI agent**: parses FastSurfer volumetrics, computes z-scores, flags abnormalities
2. **Functional fMRI agent** *(ADHD only)*: extracts resting-state connectivity
   (DMN, salience, frontostriatal, response-inhibition)
3. **Atlas mapper**: grounds anatomical labels to MNI / AAL space, attaches network membership
4. **Clinical knowledge agent**: RAG over PubMed/curated corpora
   (ADHD biomarkers, stroke syndromes, glioma imaging)
5. **Report generator**: synthesizes findings into a structured diagnostic report
6. **LLM orchestrator**: multi-turn function calling decides which tools to invoke
   and integrates evidence

#### Why a multi-agent design?
- **Disease-agnostic:** same orchestrator handles ADHD, stroke, and tumors
- **Auditable:** every claim is grounded in a tool output (the *evidence chain*)
- **Zero training data:** no labeled neuroimaging required for inference
- **Explainable:** the diagnostic report is a chain of *findings -> atlas -> literature -> assessment*

#### What this demo shows
Pre-baked (controlled) outputs at each pipeline stage so you can see exactly *what* an
end-to-end NeuroAgent run produces, without needing GPUs, LLM API keys, or the full data.
""")
    st.markdown("---")
    st.caption("Built with Streamlit. NeuroAgent project")
