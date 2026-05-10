from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from agents.schemas import AgentOutput, AtlasRegion, Measurement, Finding, make_finding, Condition

_DATA_DIR = Path(__file__).parent.parent / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

_TUMOR_SEED: list[dict] = [
    {
        "id": "T001",
        "title": "WHO CNS Tumor Classification 2021 - Overview",
        "tags": ["grading", "classification", "WHO", "IDH", "GBM"],
        "text": (
            "The 2021 WHO Classification of Tumors of the Central Nervous System (CNS5) "
            "integrates molecular parameters. Adult-type diffuse gliomas are classified as: "
            "(1) Glioblastoma IDH-wildtype (Grade 4, EGFR amplification, TERT promoter mutation, +7/-10); "
            "(2) Astrocytoma IDH-mutant (Grades 2-4, no 1p/19q codeletion); "
            "(3) Oligodendroglioma IDH-mutant and 1p/19q codeleted (Grades 2-3). "
            "IDH-wildtype GBM: median OS 14-16 months. IDH-mutant astrocytomas: better prognosis. "
            "BraTS-GLI provides T1, T1ce, T2, FLAIR for ET, TC, and WT subregion segmentation."
        ),
    },
    {
        "id": "T002",
        "title": "GBM Imaging Features - Ring Enhancement and Mass Effect",
        "tags": ["GBM", "ring enhancement", "necrosis", "mass effect", "edema"],
        "text": (
            "GBM classic MRI features: ring-enhancing lesion on T1ce (gadolinium-enhancing rim "
            "around central necrosis); heterogeneous T2/FLAIR; perilesional vasogenic edema; "
            "mass effect: midline shift, sulcal effacement, transtentorial herniation risk; "
            "corpus callosum crossing → butterfly glioma. ET volume correlates with OS. "
            "Large ET (>20 cm³) and midline shift (>10 mm) are poor prognostic markers."
        ),
    },
    {
        "id": "T003",
        "title": "Eloquent Cortex and Surgical Risk",
        "tags": ["eloquent", "motor cortex", "Broca", "Wernicke", "surgical risk", "awake craniotomy"],
        "text": (
            "Eloquent cortex requires preservation during tumor surgery. High-risk regions: "
            "(1) Primary motor cortex (precentral gyrus): resection → contralateral hemiplegia. "
            "(2) Broca's area (left IFG, BA44/45): expressive aphasia. "
            "(3) Wernicke's area (left posterior STG, BA22): receptive aphasia. "
            "(4) Primary visual cortex (calcarine): homonymous hemianopia. "
            "(5) SMA (medial superior frontal): SMA syndrome (transient). "
            "Awake craniotomy with cortical mapping is standard. DTI tractography is critical "
            "for deep tumors near the corticospinal tract."
        ),
    },
    {
        "id": "T004",
        "title": "Butterfly Glioma - Corpus Callosum Involvement",
        "tags": ["butterfly glioma", "corpus callosum", "GBM", "bilateral"],
        "text": (
            "Butterfly glioma: bilateral hemispheric tumor connected through the corpus callosum. "
            "Most commonly GBM, crossing via genu or body of CC. Generally inoperable; biopsy "
            "for molecular profiling, then chemoradiation. Callosal invasion detected as absent "
            "or distorted CC on T1; FLAIR signal extending across CC; enhancing lesion crossing midline."
        ),
    },
    {
        "id": "T005",
        "title": "Mass Effect, Herniation, and Hydrocephalus",
        "tags": ["mass effect", "herniation", "midline shift", "hydrocephalus", "ICP"],
        "text": (
            "Midline shift >5 mm is clinically significant; >10 mm → increased ICP, herniation risk. "
            "Subfalcine herniation: cingulate gyrus shifts under falx. "
            "Transtentorial herniation: temporal lobe through tentorial notch → CN III palsy, hemiplegia. "
            "Tonsillar herniation: cerebellar tonsils through foramen magnum → respiratory arrest. "
            "Obstructive hydrocephalus: tumor blocks CSF pathways → enlarged ventricles. "
            "Dexamethasone reduces vasogenic edema (standard acute management)."
        ),
    },
]

_STROKE_SEED: list[dict] = [
    {
        "id": "S001",
        "title": "Stroke Classification - TOAST Criteria",
        "tags": ["TOAST", "ischemic", "embolic", "lacunar", "cryptogenic"],
        "text": (
            "TOAST classification: (1) Large artery atherosclerosis (≥50% stenosis); "
            "(2) Cardioembolism (AF, mechanical valve, LV thrombus, PFO); "
            "(3) Small vessel occlusion (lacunar infarct <15 mm in deep perforator territory); "
            "(4) Other determined etiology; (5) Cryptogenic. "
            "ATLAS-v2 provides T1 MRI with manual lesion masks for chronic stroke survivors."
        ),
    },
    {
        "id": "S002",
        "title": "MCA Territory - Middle Cerebral Artery Stroke",
        "tags": ["MCA", "aphasia", "hemiplegia", "neglect"],
        "text": (
            "MCA supplies: frontal (lateral), parietal, temporal, basal ganglia, internal capsule. "
            "Left MCA superior: right hemiplegia + Broca's aphasia. "
            "Left MCA inferior: Wernicke's aphasia + right superior visual field defect. "
            "Right MCA: left hemiplegia + hemispatial neglect + emotional prosody loss. "
            "Lacunar (lenticulostriate): pure motor stroke (IC), pure sensory stroke (thalamus)."
        ),
    },
    {
        "id": "S003",
        "title": "PCA / Posterior Circulation Stroke",
        "tags": ["PCA", "basilar", "PICA", "Wallenberg", "hemianopia", "locked-in"],
        "text": (
            "PCA stroke: homonymous hemianopia ± macular sparing; alexia without agraphia (left PCA); "
            "thalamic syndrome (Dejerine-Roussy): hemisensory loss, dysesthesia; amnesia (hippocampal). "
            "Basilar artery: locked-in syndrome (bilateral pontine infarct - preserved vertical gaze). "
            "PICA: Wallenberg syndrome - ipsilateral facial numbness + contralateral body numbness "
            "+ ipsilateral Horner + dysphagia + vertigo + ipsilateral ataxia."
        ),
    },
    {
        "id": "S004",
        "title": "Lacunar Infarct Syndromes and Internal Capsule",
        "tags": ["lacunar", "pure motor", "pure sensory", "internal capsule", "lenticulostriate"],
        "text": (
            "Lacunar syndromes: Pure Motor Stroke (posterior IC - lenticulostriate): contralateral "
            "hemiplegia WITHOUT cortical signs (no aphasia, no neglect). "
            "Pure Sensory Stroke (VPL thalamus): hemisensory loss only. "
            "Ataxic Hemiparesis (pons or posterior IC). "
            "Internal capsule posterior limb: corticospinal tract; anterior limb: thalamocortical fibers. "
            "Lenticulostriate arteries are end-arteries → watershed vulnerability."
        ),
    },
    {
        "id": "S005",
        "title": "Lesion-Symptom Mapping and Stroke Outcomes",
        "tags": ["VLSM", "NIHSS", "mRS", "lesion volume", "aphasia", "neglect"],
        "text": (
            "VLSM: voxel-based lesion-symptom mapping correlates lesion location with deficits. "
            "Aphasia: left perisylvian (Broca: left IFG BA44; Wernicke: left pSTG BA22). "
            "Neglect: right inferior parietal, right superior temporal. "
            "NIHSS: 0-42 (minor <5; moderate 5-15; severe >20). "
            "mRS ≤2 = functional independence. Lesion volume > 70 cm³: malignant MCA infarction risk. "
            "Chronic stroke T1: hypointense core (encephalomalacia), Wallerian degeneration in CST."
        ),
    },
]

_ADHD_SEED: list[dict] = [
    {
        "id": "A001",
        "title": "Frontostriatal Model of ADHD - Neurobiological Basis",
        "tags": ["frontostriatal", "dopamine", "DLPFC", "caudate", "model"],
        "text": (
            "The frontostriatal model (Castellanos & Tannock 2002) is the dominant ADHD model: "
            "dopaminergic and noradrenergic dysregulation in prefrontal-striatal circuits. "
            "Circuit: PFC (DLPFC, ACC) → striatum (caudate, putamen) → GPi → thalamus → PFC. "
            "Stimulant medications: increase synaptic DA/NE → normalise prefrontal function. "
            "Structural: caudate (strongest), putamen, accumbens volume reductions. "
            "Functional: hypoactivation of right IFG + caudate during response inhibition."
        ),
    },
    {
        "id": "A002",
        "title": "ADHD Structural MRI Biomarkers - ENIGMA Meta-Analysis",
        "tags": ["structural", "caudate", "putamen", "cerebellum", "cortical thickness", "volume"],
        "text": (
            "Hoogman et al. 2017 (Lancet Psychiatry) ENIGMA mega-analysis (n=1,713 ADHD, "
            "n=1,529 controls): subcortical volume reductions in ADHD: "
            "caudate (d=-0.18), putamen (d=-0.11), amygdala (d=-0.19), hippocampus (d=-0.11), "
            "accumbens (d=-0.14), total cerebral volume (d=-0.10). "
            "Effects largest in children; attenuate in adults. "
            "Corpus callosum: significant reduction in all 5 sub-regions. "
            "Most robust finding: reduced right caudate volume in children."
        ),
    },
    {
        "id": "A003",
        "title": "ADHD Resting-State fMRI - DMN and Connectivity",
        "tags": ["resting state", "fMRI", "DMN", "salience", "frontostriatal", "connectivity"],
        "text": (
            "Default Mode Network (DMN) in ADHD: elevated DMN activity during tasks "
            "(failure to suppress PCC, mPFC, precuneus) - Castellanos et al. 2008. "
            "Reduced DMN-task-positive network anticorrelation. "
            "Frontostriatal hypoconnectivity: reduced DLPFC-caudate FC. "
            "Salience network disruption: reduced ACC-insula connectivity. "
            "Right IFG-caudate hypoconnectivity: most specific biomarker for impulsivity. "
            "ADHD-200 connectivity biomarkers validated across 8 sites."
        ),
    },
    {
        "id": "A004",
        "title": "Response Inhibition and Right IFG in ADHD",
        "tags": ["response inhibition", "stop signal", "right IFG", "impulsivity", "go/no-go"],
        "text": (
            "Response inhibition is the cardinal cognitive deficit in ADHD (Barkley 1997). "
            "Neural substrate: right IFG (rIFG, BA44/45) + subthalamic nucleus + right caudate. "
            "Stop-signal reaction time (SSRT) is the gold-standard measure. "
            "ADHD vs controls: longer SSRT; rIFG hypoactivation during stop trials. "
            "Methylphenidate normalises rIFG activation. "
            "rIFG cortical thinning and volume reduction in pars triangularis. "
            "rIFG-caudate connectivity predicts SSRT across individuals."
        ),
    },
    {
        "id": "A005",
        "title": "ADHD-200 Dataset and Site Effects",
        "tags": ["ADHD-200", "dataset", "sites", "phenotype", "fMRI", "structural"],
        "text": (
            "ADHD-200 Consortium: 973 subjects (362 ADHD, 585 controls) across 8 sites: "
            "Brown, KKI, NeuroIMAGE, NYU, OHSU, Peking (1,2,3), Pittsburgh, WashU. "
            "Age 7-21 years; 2:1 male:female ratio. T1-weighted MRI + resting-state fMRI. "
            "Site effects ≈ 50% of total variance → requires site correction (ComBat). "
            "Phenotypic data: ADHD diagnosis, subtype, IQ, medication status, Conners/ADHD-RS. "
            "Medication status at scan is a critical covariate - stimulants normalise caudate activity."
        ),
    },
]

_SEED_CORPORA: dict[str, list[dict]] = {
    "tumor":  _TUMOR_SEED,
    "stroke": _STROKE_SEED,
    "adhd":   _ADHD_SEED,
}

class _TFIDFRetriever:

    def __init__(self, corpus: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs

        self._corpus = corpus
        self._cs = cs
        texts = [f"{d['title']} {d['text']}" for d in corpus]
        self._vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1,
                                    stop_words="english")
        self._mat = self._vec.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity as cs
                                                                      
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        q_vec  = self._vec.transform([query])
        scores = cs(q_vec, self._mat)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._corpus[i],
             "score": round(float(scores[i]), 4),
             "pmid": "",
             "year": "",
             "journal": "seed corpus",
             "authors": "",
             "doi": "",
             "mesh": ""}
            for i in ranked
            if scores[i] >= 0.01
        ]

class ClinicalKnowledgeAgent:

    def __init__(self):
        self._tfidf_retrievers: dict[str, _TFIDFRetriever] = {}
        self._chroma_available: Optional[bool] = None                          

    def _check_chroma(self, condition: str) -> bool:
        try:
            from build_knowledge_base import collection_exists
            return collection_exists(condition)
        except Exception:
            return False

    def _chroma_retrieve(self, condition: str, query: str, top_k: int) -> list[dict]:
        from build_knowledge_base import retrieve as chroma_retrieve
        raw = chroma_retrieve(condition, query, top_k=top_k)
        results = []
        for r in raw:
            results.append({
                "id":      r.get("pmid", ""),
                "title":   r.get("title", ""),
                "text":    r.get("text", ""),
                "tags":    r.get("mesh", "").split(" | ") if r.get("mesh") else [],
                "score":   r.get("score", 0.0),
                "pmid":    r.get("pmid", ""),
                "year":    r.get("year", ""),
                "journal": r.get("journal", ""),
                "authors": r.get("authors", ""),
                "doi":     r.get("doi", ""),
                "mesh":    r.get("mesh", ""),
            })
        return results

    def _tfidf_retrieve(self, condition: str, query: str, top_k: int) -> list[dict]:
        if condition not in self._tfidf_retrievers:
            corpus = _SEED_CORPORA.get(condition)
            if corpus is None:
                raise ValueError(f"Unknown condition: {condition}")
            self._tfidf_retrievers[condition] = _TFIDFRetriever(corpus)
        return self._tfidf_retrievers[condition].retrieve(query, top_k=top_k)

    def retrieve(
        self,
        condition: Condition,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        if self._check_chroma(condition):
            try:
                return self._chroma_retrieve(condition, query, top_k)
            except Exception:
                pass                          
        return self._tfidf_retrieve(condition, query, top_k)

    def get_clinical_context(
        self,
        condition: Condition,
        region_name: str,
        finding_summary: str,
        top_k: int = 3,
    ) -> AgentOutput:
        from agents.schemas import AgentOutput, AtlasRegion, Measurement, Finding, make_finding

        t0 = time.time()
        query = f"{region_name} {finding_summary}"
        docs  = self.retrieve(condition, query, top_k=top_k)

        findings: list[Finding] = []
        if docs:
            region = AtlasRegion(
                canonical_name=region_name,
                hemisphere="bilateral",
                lobe="unknown",
                atlas_label="RAG",
                mni_centroid=(0, 0, 0),
            )
            findings.append(make_finding(
                idx=1,
                finding_type="clinical_context",
                severity="unknown",
                region=region,
                measurement=Measurement(
                    metric="n_retrieved",
                    value=len(docs),
                    units="documents",
                ),
                clinical_significance=f"Retrieved {len(docs)} relevant passages for: {query}",
                evidence_chain=[
                    f"[{d['id']}] {d['title']} (score={d['score']:.3f}): {d['text'][:400]}"
                    for d in docs
                ],
            ))

        elapsed = time.time() - t0
        return AgentOutput(
            agent_name="clinical_knowledge",
            condition=condition,
            subject_id="corpus_query",
            status="success",
            findings=findings,
            global_metrics={
                "query":        query,
                "n_retrieved":  len(docs),
            },
            metadata={"processing_time_s": round(elapsed, 3)},
        )

    def format_context_for_llm(
        self,
        condition: Condition,
        query: str,
        top_k: int = 4,
    ) -> str:
        docs = self.retrieve(condition, query, top_k=top_k)
        if not docs:
            return "No relevant clinical knowledge retrieved."

        lines = ["RETRIEVED CLINICAL KNOWLEDGE:\n"]
        for i, doc in enumerate(docs, start=1):
            ref = f"PMID:{doc['pmid']} {doc['year']}" if doc.get("pmid") else doc.get("journal", "")
            lines.append(f"[{i}] {doc['title']} - {ref} (score={doc['score']:.3f})")
            lines.append(doc["text"])
            lines.append("")
        return "\n".join(lines)

    def retrieve_for_finding(
        self,
        condition: Condition,
        finding: "Finding",
        top_k: int = 4,
    ) -> list[dict]:
        parts = [finding.region.canonical_name]

        if condition == "adhd" and finding.region.adhd_relevance:
            parts.append(finding.region.adhd_relevance)
        elif condition == "stroke" and finding.region.stroke_territory:
            parts.append(finding.region.stroke_territory)
        elif condition == "tumor" and finding.region.tumor_relevance:
            parts.append(finding.region.tumor_relevance)

        parts.append(
            f"{finding.measurement.metric} {finding.measurement.value:.2f} {finding.measurement.units}"
        )

        parts.append(finding.clinical_significance[:120])

        query = " ".join(parts)
        return self.retrieve(condition, query, top_k=top_k)

    def get_clinical_context_for_finding(
        self,
        condition: Condition,
        finding: "Finding",
        top_k: int = 3,
    ) -> AgentOutput:
        t0 = time.time()
        docs = self.retrieve_for_finding(condition, finding, top_k=top_k)

        findings_out: list[Finding] = []
        if docs:
            region = AtlasRegion(
                canonical_name=finding.region.canonical_name,
                hemisphere="bilateral",
                lobe="unknown",
                atlas_label="RAG",
                mni_centroid=(0, 0, 0),
            )
            findings_out.append(make_finding(
                idx=1,
                finding_type="clinical_context",
                severity="unknown",
                region=region,
                measurement=Measurement(
                    metric="n_retrieved",
                    value=len(docs),
                    units="documents",
                ),
                clinical_significance=(
                    f"Retrieved {len(docs)} passages for: {finding.region.canonical_name}"
                ),
                evidence_chain=[
                    f"[{d['id']}] {d['title']} (score={d['score']:.3f}): {d['text'][:400]}"
                    for d in docs
                ],
            ))

        elapsed = time.time() - t0
        return AgentOutput(
            agent_name="clinical_knowledge",
            condition=condition,
            subject_id="corpus_query",
            status="success",
            findings=findings_out,
            global_metrics={"n_retrieved": len(docs)},
            metadata={"processing_time_s": round(elapsed, 3)},
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Clinical Knowledge Agent")
    parser.add_argument("--condition", required=True, choices=["tumor", "stroke", "adhd"])
    parser.add_argument("--query",     required=True, help="Retrieval query string")
    parser.add_argument("--top-k",     type=int, default=5)
    args = parser.parse_args()

    agent = ClinicalKnowledgeAgent()
    docs  = agent.retrieve(args.condition, args.query, top_k=args.top_k)

    print(f"\nTop {len(docs)} passages for condition='{args.condition}', "
          f"query='{args.query}':\n")
    for doc in docs:
        print(f"  [{doc['id']}] {doc['title']} (score={doc['score']:.4f})")
        print(f"  {doc['text'][:300]}...")
        print()

    print("\nFormatted for LLM prompt:\n")
    print(agent.format_context_for_llm(args.condition, args.query, top_k=3))
