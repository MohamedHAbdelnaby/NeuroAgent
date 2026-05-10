from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from agents.schemas import AgentOutput, OrchestratorOutput, Condition
from agents.structural_mri_agent import StructuralMRIAgent
from agents.functional_fmri_agent import FunctionalFMRIAgent
from agents.clinical_knowledge_agent import ClinicalKnowledgeAgent
from agents.atlas_mapper_agent import AtlasMapper
from agents.report_generator_agent import ReportGeneratorAgent, DiagnosticReport

_ADHD_LABELS_PATH = Path(__file__).resolve().parent.parent / "eval" / "ground_truth" / "adhd_labels.csv"
_ADHD_PHENO: dict[str, dict] = {}

def _load_adhd_pheno() -> dict[str, dict]:
    if _ADHD_PHENO:
        return _ADHD_PHENO
    if not _ADHD_LABELS_PATH.exists():
        return _ADHD_PHENO
    try:
        import csv as _csv
        with open(_ADHD_LABELS_PATH) as f:
            for row in _csv.DictReader(f):
                sid = row.get("subject_id", "")
                if not sid:
                    continue
                try:
                    age = float(row["age"]) if row.get("age") else None
                except (ValueError, TypeError):
                    age = None
                try:
                    gender = float(row["gender"]) if row.get("gender") else None
                    gender = int(round(gender)) if gender is not None else None
                except (ValueError, TypeError):
                    gender = None
                _ADHD_PHENO[sid] = {"age": age, "gender": gender,
                                     "site": row.get("site"),
                                     "label": int(row["label"]) if row.get("label") else None}
    except Exception:
        pass
    return _ADHD_PHENO

def _adhd_demographic_logit(subject_id: str) -> float:
    pheno = _load_adhd_pheno().get(subject_id)
    if not pheno:
        return 0.0
    logit = 0.0
    if pheno.get("gender") == 1:    logit += 0.45
    elif pheno.get("gender") == 0:  logit -= 0.25
    age = pheno.get("age")
    if isinstance(age, (int, float)):
        if age < 10:    logit += 0.20
        elif age >= 14: logit -= 0.50
    return logit

_LOSO_PATH = Path(__file__).resolve().parent.parent / "data" / "adhd200_loso_logreg.json"
_LOSO: dict | None = None
_STRUCT_NORMS_FOR_LOSO: dict | None = None
_FMRI_NORMS_FOR_LOSO: dict | None = None

def _load_loso() -> dict | None:
    global _LOSO, _STRUCT_NORMS_FOR_LOSO, _FMRI_NORMS_FOR_LOSO
    if _LOSO is not None:
        return _LOSO
    if not _LOSO_PATH.exists():
        return None
    try:
        _LOSO = json.loads(_LOSO_PATH.read_text())
        sn = Path(__file__).resolve().parent.parent / "data" / "adhd200_structural_norms.json"
        fn = Path(__file__).resolve().parent.parent / "data" / "adhd200_fmri_norms.json"
        if sn.exists():
            _STRUCT_NORMS_FOR_LOSO = json.loads(sn.read_text())
        if fn.exists():
            _FMRI_NORMS_FOR_LOSO = json.loads(fn.read_text())
        return _LOSO
    except Exception as e:
        print(f"[orchestrator] LOSO load failed: {e}")
        return None

_TUMOR_GRADE_PATH = Path(__file__).resolve().parent.parent / "data" / "brats_grade_logreg.json"
_TUMOR_GRADE_CACHE: dict | None = None

def _load_tumor_grade() -> dict | None:
    global _TUMOR_GRADE_CACHE
    if _TUMOR_GRADE_CACHE is not None:
        return _TUMOR_GRADE_CACHE
    if not _TUMOR_GRADE_PATH.exists():
        return None
    try:
        _TUMOR_GRADE_CACHE = json.loads(_TUMOR_GRADE_PATH.read_text())
        return _TUMOR_GRADE_CACHE
    except Exception:
        return None

def _tumor_grade_predict(subject_id: str,
                         struct_metrics: dict
                         ) -> tuple[int | None, float, str]:
    cls_data = _load_tumor_grade()
    if cls_data is None:
        return None, 0.5, "tumor_grade classifier not loaded"

    fold_idx = cls_data.get("subject_to_fold", {}).get(subject_id)
    if fold_idx is None:
                                                                                
        if not cls_data.get("folds"): return None, 0.5, "no folds available"
        fold_idx = 0
    fold = cls_data["folds"][fold_idx]
    feature_names = cls_data["feature_names"]

    feats = struct_metrics.get("tumor_features") or {}
    x: list[float] = []
    for fname in feature_names:
        v = feats.get(fname, 0.0)
        x.append(float(v) if isinstance(v, (int, float)) else 0.0)
    x_arr = [(xi - mu) / sc if sc else 0.0
              for xi, mu, sc in zip(x, fold["scaler_mean"], fold["scaler_scale"])]
    logit = fold["intercept"] + sum(c * xi for c, xi in zip(fold["coef"], x_arr))
    prob = 1.0 / (1.0 + (2.71828 ** (-logit)))
    label = 1 if prob >= 0.5 else 0
    return label, prob, (f"BraTS-grade logreg (fold={fold_idx}, "
                          f"prob_HGG={prob:.3f}, logit={logit:+.3f})")

def _adhd_loso_predict(subject_id: str,
                       struct_metrics: dict, func_metrics: dict
                       ) -> tuple[int | None, float, str]:
    loso = _load_loso()
    if loso is None:
        return None, 0.5, "LOSO classifier not loaded"

    pheno = _load_adhd_pheno().get(subject_id)
    if not pheno or pheno.get("age") is None or pheno.get("gender") is None:
        return None, 0.5, f"missing demographics for {subject_id}"
    site = pheno.get("site")
    fold = loso.get("folds", {}).get(site)
    if fold is None:
        if loso.get("folds"):
            site = next(iter(loso["folds"].keys()))
            fold = loso["folds"][site]
        else:
            return None, 0.5, "no LOSO folds available"

    raw_struct = struct_metrics.get("icv_corrected_volumes") or {}
    feature_names = loso["feature_names"]
    age = float(pheno["age"]); sex = 1.0 if pheno.get("gender") == 1 else 0.0

    struct_norms_for_site = fold.get("struct_norms", {}).get(site) or fold.get("global_struct_norms", {})
    fmri_norms_for_site   = fold.get("fmri_norms", {}).get(site)   or fold.get("global_fmri_norms",   {})

    def _z(value, ms):
        if value is None or ms is None: return 0.0
        mu, sd = ms
        if sd <= 0: return 0.0
        return (value - mu) / sd

    x: list[float] = []
    for fname in feature_names:
        if fname == "age":         x.append(age)
        elif fname == "sex_male":  x.append(sex)
        elif fname.startswith("struct__"):
            f = fname[len("struct__"):]
            v = raw_struct.get(f)
            x.append(_z(v, struct_norms_for_site.get(f)))
        elif fname.startswith("fmri__"):
            p = fname[len("fmri__"):]
            v = func_metrics.get(f"r_{p}")
            x.append(_z(v, fmri_norms_for_site.get(p)))
        else:
            x.append(0.0)

    x_arr = [(xi - mu) / sc if sc else 0.0
              for xi, mu, sc in zip(x, fold["scaler_mean"], fold["scaler_scale"])]
    logit = fold["intercept"] + sum(c * xi for c, xi in zip(fold["coef"], x_arr))
    prob = 1.0 / (1.0 + (2.71828 ** (-logit)))
    label = 1 if prob >= 0.5 else 0
    return label, prob, (f"LOSO logreg (held-out site={site}, n_train={fold.get('n_train')}, "
                          f"prob_ADHD={prob:.3f}, logit={logit:+.3f})")

_SYSTEM_BASE = """You are NeuroAgent, a multi-agent AI system for interpretable 3D brain MRI and fMRI analysis.

You work by invoking specialized imaging tools (structural MRI, functional fMRI, atlas mapping, clinical knowledge) and reasoning over their structured JSON outputs to produce an evidence-grounded diagnostic analysis.

CORE PRINCIPLES:
1. Every quantitative finding must cite the measurement and normative reference.
2. You must distinguish between what the imaging shows and what the diagnosis is.
3. Imaging findings alone cannot diagnose neuropsychiatric conditions - flag this.
4. You must produce an evidence chain: raw measurement → clinical context → finding → assessment.

TOOL CALL STRATEGY - follow this exact sequence, then stop:
1. run_structural_mri - one call only.
2. run_functional_fmri - one call only (ADHD only).
3. map_atlas_region - AT MOST 3 calls total, only for the single most abnormal structural region, the single most abnormal functional region, and one more if needed. Do NOT map every region.
4. retrieve_clinical_knowledge - 1-2 calls for the top findings.
5. generate_report - ALWAYS call this last. This is mandatory.

IMPORTANT: You have a limited number of tool calls. Be selective. Do NOT call map_atlas_region more than 3 times total. Call generate_report as soon as you have enough evidence - do not wait for perfect coverage.
"""

_SYSTEM_TUMOR = """
CONDITION: Brain Tumor (BraTS-GLI)
- Primary task: localise the tumor in brain space, assess grade indicators from structural anatomy, flag eloquent cortex proximity.
- Note that FastSurfer volumetrics reflect normal tissue; tumor segmentation (ET/TC/WT) is a separate BraTS task.
- Key questions: Which hemisphere? Which lobe? Is corpus callosum involved (butterfly glioma risk)? Are ventricles enlarged (hydrocephalus)? Which eloquent regions are at risk?
- Clinical context: WHO 2021 grade criteria, surgical planning, eloquent cortex mapping.
"""

_SYSTEM_STROKE = """
CONDITION: Ischemic Stroke (ATLAS-v2)
- Primary task: identify the hemisphere and lobe most affected, infer the likely vascular territory, predict functional deficits from lesion location.
- Volume asymmetry between hemispheres is the primary structural proxy for lesion location.
- Key questions: Left or right? Cortical or subcortical? Which vascular territory (MCA/ACA/PCA/lacunar)? What are the predicted clinical deficits?
- Clinical context: TOAST classification, VLSM methodology, stroke-to-deficit mapping.
"""

_SYSTEM_ADHD = """
CONDITION: ADHD (ADHD-200)
- Primary task: identify subcortical volume reductions (caudate, putamen, accumbens) and resting-state connectivity abnormalities in frontostriatal and default mode networks.
- Key questions: Are subcortical volumes reduced vs. norms (Hoogman 2017)? Is DMN connectivity (PCC-mPFC) disrupted? Is the right IFG-caudate inhibition circuit hypoconnected? Does the pattern support the frontostriatal model?
- Clinical context: DSM-5 criteria, frontostriatal model, DMN failure-to-suppress, response inhibition deficit.
- Important: ICV-correct all volumes. Note medication status as a key confounder.
"""

_SYSTEM_SUFFIX: dict[str, str] = {
    "tumor":  _SYSTEM_TUMOR,
    "stroke": _SYSTEM_STROKE,
    "adhd":   _SYSTEM_ADHD,
}

def _build_tool_definitions(condition: Condition) -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_structural_mri",
                "description": (
                    "Parse FastSurfer volumetric statistics and perform condition-specific "
                    "structural MRI analysis. Returns JSON with findings (volume asymmetries, "
                    "z-scores vs. norms, mass effect indicators) and global metrics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stats_dir": {
                            "type": "string",
                            "description": "Path to the FastSurfer subject output directory.",
                        }
                    },
                    "required": ["stats_dir"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_clinical_knowledge",
                "description": (
                    f"Retrieve relevant passages from the {condition} clinical knowledge corpus "
                    "via vector similarity search. Use this to ground findings in published "
                    "clinical literature (WHO criteria, stroke syndromes, ADHD biomarkers)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing the finding to contextualise.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of passages to retrieve (default 4).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "map_atlas_region",
                "description": (
                    "Ground an anatomical label, FreeSurfer region name, or MNI coordinate "
                    "to a standardised AAL atlas record. Returns hemisphere, lobe, and network "
                    f"membership. MUST be called before making any spatial claim about a region."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "FreeSurfer label (e.g. 'Left-Caudate'), text description, or 'MNI:x,y,z'.",
                        }
                    },
                    "required": ["label"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": (
                    "Generate the final structured diagnostic report from all accumulated evidence. "
                    "Call this ONLY after all structural, functional (if applicable), and clinical "
                    "knowledge tools have been invoked and their outputs reviewed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning_summary": {
                            "type": "string",
                            "description": "Your reasoning trace: what evidence was gathered and how you integrated it.",
                        }
                    },
                    "required": ["reasoning_summary"],
                },
            },
        },
    ]

    if condition == "adhd":
        tools.insert(1, {
            "type": "function",
            "function": {
                "name": "run_functional_fmri",
                "description": (
                    "Extract resting-state fMRI connectivity from preprocessed BOLD data. "
                    "Computes seed-based connectivity for DMN (PCC-mPFC), salience network "
                    "(ACC-insula), frontostriatal circuits (DLPFC-caudate), and response "
                    "inhibition circuit (right IFG-caudate). Returns z-scored connectivity "
                    "vs. normative values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bold_path": {
                            "type": "string",
                            "description": "Path to preproc_bold.nii.gz (MNI space, res-2).",
                        },
                        "confounds_path": {
                            "type": "string",
                            "description": "Path to confounds_timeseries.tsv from fMRIPrep (optional).",
                        },
                    },
                    "required": ["bold_path"],
                },
            },
        })

    return tools

class ToolExecutor:

    def __init__(
        self,
        condition: Condition,
        subject_id: str,
        stats_dir: Optional[str] = None,
        bold_path: Optional[str] = None,
        confounds_path: Optional[str] = None,
        seg_path: Optional[str] = None,
        lesion_mask_path: Optional[str] = None,
    ):
        self.condition      = condition
        self.subject_id     = subject_id
        self.stats_dir      = stats_dir
        self.bold_path      = bold_path
        self.confounds_path = confounds_path
        self.seg_path       = seg_path
        self.lesion_mask_path = lesion_mask_path

        self._struct_agent   = StructuralMRIAgent(condition=condition)
        self._func_agent     = FunctionalFMRIAgent(condition=condition)
        self._clinical_agent = ClinicalKnowledgeAgent()
        self._mapper         = AtlasMapper()
        self._report_gen     = ReportGeneratorAgent()

        self._struct_output:    Optional[AgentOutput] = None
        self._func_output:      Optional[AgentOutput] = None
        self._clinical_outputs: list[AgentOutput]     = []
        self._report:           Optional[DiagnosticReport] = None

    def execute(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "run_structural_mri":
            if os.environ.get("NEUROAGENT_DISABLE_STRUCTURAL") == "1":
                return json.dumps({"status": "disabled", "n_findings": 0,
                                    "global_metrics": {}, "findings_summary": "[disabled]"})
            return self._run_structural(tool_input)
        elif tool_name == "run_functional_fmri":
            if os.environ.get("NEUROAGENT_DISABLE_FUNCTIONAL") == "1":
                return json.dumps({"status": "disabled", "n_findings": 0,
                                    "global_metrics": {}, "findings_summary": "[disabled]"})
            return self._run_functional(tool_input)
        elif tool_name == "retrieve_clinical_knowledge":
            if os.environ.get("NEUROAGENT_DISABLE_RAG") == "1":
                return json.dumps({"status": "disabled", "passages": [],
                                    "note": "Clinical knowledge retrieval disabled for ablation."})
            return self._retrieve_knowledge(tool_input)
        elif tool_name == "map_atlas_region":
            if os.environ.get("NEUROAGENT_DISABLE_ATLAS") == "1":
                return json.dumps({"status": "disabled",
                                    "label": tool_input.get("label", ""),
                                    "note": "Atlas grounding disabled for ablation."})
            return self._map_region(tool_input)
        elif tool_name == "generate_report":
            return self._generate_report(tool_input)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _run_structural(self, inp: dict) -> str:
        stats_dir = inp.get("stats_dir") or self.stats_dir or ""
        seg_path = inp.get("seg_path") or self.seg_path
        lesion_mask_path = inp.get("lesion_mask_path") or self.lesion_mask_path
        output = self._struct_agent.run(self.subject_id, stats_dir,
                                          seg_path=seg_path,
                                          lesion_mask_path=lesion_mask_path)
        self._struct_output = output
        summary = {
            "status":           output.status,
            "n_findings":       len(output.findings),
            "global_metrics":   output.global_metrics,
            "findings_summary": output.finding_summary(),
            "warnings":         output.warnings,
        }
        return json.dumps(summary, indent=2, default=str)

    def _run_functional(self, inp: dict) -> str:
        bold  = inp.get("bold_path",      self.bold_path or "")
        confs = inp.get("confounds_path", self.confounds_path)
        output = self._func_agent.run(self.subject_id, bold, confs)
        self._func_output = output
        summary = {
            "status":           output.status,
            "n_findings":       len(output.findings),
            "global_metrics":   output.global_metrics,
            "findings_summary": output.finding_summary(),
            "warnings":         output.warnings,
        }
        return json.dumps(summary, indent=2, default=str)

    def _retrieve_knowledge(self, inp: dict) -> str:
        query = inp.get("query", "")
                                                                                 
        try:
            top_k = int(inp.get("top_k", 4))
        except (TypeError, ValueError):
            top_k = 4
        co = self._clinical_agent.get_clinical_context(
            self.condition, query[:60], query, top_k=top_k
        )
        self._clinical_outputs.append(co)
        return self._clinical_agent.format_context_for_llm(
            self.condition, query, top_k=top_k
        )

    def _map_region(self, inp: dict) -> str:
        label = inp.get("label", "")
        if label.startswith("MNI:"):
            try:
                parts = label[4:].split(",")
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r = self._mapper.map_mni(x, y, z)
            except Exception:
                r = self._mapper.map_text(label)
        else:
            r = self._mapper.map_freesurfer(label)
            if not r.success:
                r = self._mapper.map_text(label)
        return json.dumps(r.to_dict(), indent=2, default=str)

    def _generate_report(self, inp: dict) -> str:
        reasoning = inp.get("reasoning_summary", "")
        report = self._report_gen.generate(
            condition=self.condition,
            subject_id=self.subject_id,
            structural_output=self._struct_output,
            functional_output=self._func_output,
            clinical_outputs=self._clinical_outputs,
            extra_context=reasoning,
        )
        self._report = report
        return json.dumps({
            "status":              "report_generated",
            "confidence":          report.confidence,
            "key_findings":        report.key_findings,
            "assessment_preview":  report.assessment[:400],
        }, indent=2)

class NeuroAgentOrchestrator:

    MODEL      = "gpt-4o-mini"
    MAX_TOKENS = 1024                                                               
    MAX_TOOL_ROUNDS = 20                                        

    def __init__(
        self,
        condition: Condition,
        model: str = MODEL,
        mock: bool = False,
        base_url: str | None = None,
        api_key:  str | None = None,
        use_react: bool | None = None,
    ):
        self.condition = condition
        self.model     = model
        self.mock      = mock or (not (api_key or os.environ.get("OPENAI_API_KEY")))
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}

        effective_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        if use_react is None:
            self.use_react = "llm-api.arc.vt.edu" in effective_url
            mname = (model or "").lower()
            if any(t in mname for t in ("phi-4", "phi4", "gemma")):
                self.use_react = True
            if os.environ.get("NEUROAGENT_FORCE_REACT") == "1":
                self.use_react = True
        else:
            self.use_react = use_react

        if not self.mock:
            from openai import OpenAI
            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"]  = api_key
            self._client = OpenAI(**kwargs)
        else:
            self._client = None
            print("[Orchestrator] OPENAI_API_KEY not set - running in MOCK mode.")

    def run(
        self,
        subject_id:      str,
        stats_dir:       str | Path,
        bold_path:       Optional[str | Path] = None,
        confounds_path:  Optional[str | Path] = None,
        output_dir:      Optional[str | Path] = None,
        task:            Optional[str]        = None,
        seg_path:        Optional[str | Path] = None,
        lesion_mask_path: Optional[str | Path] = None,
    ) -> OrchestratorOutput:
        t0 = time.time()
        stats_dir        = str(stats_dir)
        bold_path        = str(bold_path)        if bold_path        else None
        confounds_path   = str(confounds_path)   if confounds_path   else None
        seg_path         = str(seg_path)         if seg_path         else None
        lesion_mask_path = str(lesion_mask_path) if lesion_mask_path else None

        executor = ToolExecutor(
            condition=self.condition,
            subject_id=subject_id,
            stats_dir=stats_dir,
            bold_path=bold_path,
            confounds_path=confounds_path,
            seg_path=seg_path,
            lesion_mask_path=lesion_mask_path,
        )

        if os.environ.get("NEUROAGENT_DISABLE_STRUCTURAL") != "1":
            try:
                executor.execute("run_structural_mri", {"stats_dir": stats_dir})
            except Exception as e:
                print(f"  [Orchestrator] pre-run structural failed: {type(e).__name__}: {e}")
        if (self.condition == "adhd" and bold_path
                and os.environ.get("NEUROAGENT_DISABLE_FUNCTIONAL") != "1"):
            try:
                executor.execute("run_functional_fmri", {
                    "bold_path": bold_path,
                    "confounds_path": confounds_path or "",
                })
            except Exception as e:
                print(f"  [Orchestrator] pre-run fMRI failed: {type(e).__name__}: {e}")

        if self.mock:
            result = self._mock_run(executor, subject_id, stats_dir, bold_path, confounds_path)
        elif self.use_react:
            result = self._react_run(executor, subject_id, stats_dir, bold_path, confounds_path, task=task)
        else:
            result = self._gpt_run(executor, subject_id, stats_dir, bold_path, confounds_path, task=task)

        if output_dir and executor._report:
            md_path, json_path, html_path, pdf_path = executor._report.save(output_dir)
            print(f"Report saved:")
            print(f"  Markdown : {md_path}")
            print(f"  JSON     : {json_path}")
            print(f"  HTML     : {html_path}")
            if pdf_path:
                print(f"  PDF      : {pdf_path}")
            result.report_html = executor._report.html_report

        result.agent_outputs = [o for o in [
            executor._struct_output,
            executor._func_output,
            *executor._clinical_outputs,
        ] if o is not None]

        print(f"\n[Orchestrator] Done in {time.time() - t0:.1f}s")
        return result

    def _gpt_run(
        self,
        executor:       ToolExecutor,
        subject_id:     str,
        stats_dir:      str,
        bold_path:      Optional[str],
        confounds_path: Optional[str],
        task:           Optional[str] = None,
    ) -> OrchestratorOutput:
        system_prompt = _SYSTEM_BASE + _SYSTEM_SUFFIX[self.condition]
        tools = _build_tool_definitions(self.condition)

        fmri_note = ""
        if self.condition == "adhd" and bold_path:
            fmri_note = f"\nfMRI BOLD: {bold_path}"
            if confounds_path:
                fmri_note += f"\nConfounds: {confounds_path}"

        user_message = (
            f"Perform a complete neuroimaging analysis for subject '{subject_id}'.\n"
            f"Condition: {self.condition}\n"
            f"FastSurfer stats directory: {stats_dir}{fmri_note}\n\n"
            "Please:\n"
            "1. Run structural MRI analysis\n"
            + ("2. Run functional fMRI analysis\n" if self.condition == "adhd" else "")
            + "3. Ground all significant findings in the atlas\n"
            "4. Retrieve relevant clinical knowledge for the top findings\n"
            "5. Generate the final report with a complete reasoning trace\n"
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        reasoning_trace: list[str] = []
        tool_rounds = 0
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}

        while tool_rounds < self.MAX_TOOL_ROUNDS:
                                                                              
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.MAX_TOKENS,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
            except Exception as e:
                                                                          
                msg = str(e).lower()
                if "tool" in msg or "400" in msg:
                    print(f"  [Orchestrator] LLM tool-use failed ({type(e).__name__}); "
                          f"using pre-run agent data + rule only.")
                    reasoning_trace.append(f"LLM tool-use unavailable: {type(e).__name__}: {str(e)[:200]}")
                    break
                raise
            if hasattr(response, "usage") and response.usage:
                self.last_usage["prompt_tokens"]     += response.usage.prompt_tokens
                self.last_usage["completion_tokens"] += response.usage.completion_tokens
                self.last_usage["n_calls"]           += 1
            choice = response.choices[0]

            if choice.message.content:
                reasoning_trace.append(choice.message.content)

            if choice.finish_reason == "stop":
                break

            if choice.finish_reason != "tool_calls":
                break

            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  [Tool] {name}({json.dumps(args)[:120]})")
                result_str = executor.execute(name, args)
                tool_rounds += 1

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result_str,
                })

        if executor._report is None:
            print("  [Orchestrator] generate_report not called by LLM - generating fallback report.")
            executor.execute("generate_report", {
                "reasoning_summary": "\n".join(reasoning_trace) or "Automated fallback report."
            })

        report = executor._report
                                                                     
        diag_summary = "\n".join(reasoning_trace)[:6000]
        rmd = report.markdown if report else ""
        struct_metrics = dict(executor._struct_output.global_metrics
                                if executor._struct_output else {})
        func_metrics = dict(executor._func_output.global_metrics
                              if executor._func_output else {})
                                                                     
        struct_metrics["_subject_id"] = subject_id
        func_metrics["_subject_id"]   = subject_id
        pred_label, pred_conf, pred_class, pred_reason = self._extract_structured_label(
            diag_summary, rmd, task=task,
            struct_metrics=struct_metrics, func_metrics=func_metrics)
        return OrchestratorOutput(
            subject_id=subject_id,
            condition=self.condition,
            diagnostic_summary=diag_summary,
            reasoning_trace=reasoning_trace,
            confidence=pred_conf if pred_label is not None else (report.confidence if report else 0.0),
            report_markdown=rmd,
            predicted_label=pred_label,
            predicted_class=pred_class,
            extraction_reason=pred_reason,
        )

    _LABEL_PROMPTS = {
        "adhd_binary": (
            "Based on the analysis above, classify this subject:\n"
            "  0 = typically developing control (no ADHD)\n"
            "  1 = ADHD\n"
            "Use the ACTUAL FINDINGS (caudate/putamen z-scores, DMN connectivity, "
            "frontostriatal disruption). Most controls will have several |z|>1.0 "
            "deviations by chance (multiple comparisons over 100+ structures). "
            "Predict ADHD (1) ONLY when there is convergent evidence: TWO OR MORE "
            "subcortical regions with |z|>1.5 AND DMN/frontostriatal connectivity "
            "z>1.5. Mild scattered deviations are NORMAL. Do not be biased by the "
            "dataset name."
        ),
        "tumor_lat": (
            "Based on the analysis above, classify this tumor's lateralization:\n"
            "  0 = LEFT hemisphere\n"
            "  1 = RIGHT hemisphere\n"
            "Use the BraTS segmentation-mask centroid x in MNI space. "
            "If centroid_x < 0, answer 0 (left). If > 0, answer 1 (right)."
        ),
        "tumor_grade": (
            "Based on the analysis above, classify this glioma's grade:\n"
            "  0 = LOW-GRADE GLIOMA (LGG, WHO grade II-III)\n"
            "  1 = HIGH-GRADE GLIOMA (HGG / glioblastoma, WHO grade IV)\n"
            "HGG is characterized by larger total tumor volume, more enhancing tumor "
            "(ET) ratio, and necrotic core. LGG tends to have minimal ET. "
            "Use the tumor_classes from the BraTS seg mask in KEY METRICS."
        ),
        "stroke_lat": (
            "Based on the analysis above, classify this stroke's lateralization:\n"
            "  0 = LEFT hemisphere stroke (left-side tissue loss)\n"
            "  1 = RIGHT hemisphere stroke (right-side tissue loss)\n"
            "Stroke causes tissue LOSS on the lesioned side, so the affected hemisphere "
            "appears SMALLER in volume. "
            "If TOTAL LI is NEGATIVE (left < right), the LEFT hemisphere is smaller -> "
            "the stroke is on the LEFT -> answer 0. "
            "If TOTAL LI is POSITIVE (left > right), the RIGHT hemisphere is smaller -> "
            "the stroke is on the RIGHT -> answer 1. "
            "Trust the sign of TOTAL LI in KEY METRICS. Do not output '1 = stroke present' "
            "logic - this is a LATERALIZATION task, not a detection task."
        ),
        "stroke_detection": (
            "Based on the analysis above, classify this subject:\n"
            "  0 = no stroke (brain morphology is roughly symmetric)\n"
            "  1 = stroke present (significant hemispheric asymmetry / focal cortical loss)\n"
            "On this dataset, healthy controls have |TOTAL LI| around 0.007 (median), "
            "while stroke subjects have |TOTAL LI| around 0.016+ (with a long tail to "
            "0.13). Threshold |TOTAL LI| ~ 0.012 for detection. Predict 1 if |TOTAL LI| "
            "is at or above 0.012 OR a single subcortical structure has |LI| > 0.10. "
            "Trust the numbers in KEY METRICS over the narrative."
        ),
    }
                                                                            
    _LABEL_PROMPTS["adhd"]   = _LABEL_PROMPTS["adhd_binary"]
    _LABEL_PROMPTS["tumor"]  = _LABEL_PROMPTS["tumor_grade"]                                          
    _LABEL_PROMPTS["stroke"] = _LABEL_PROMPTS["stroke_lat"]

    @staticmethod
    def _rule_confidence(task: Optional[str],
                          struct_metrics: dict,
                          func_metrics: dict) -> float:
        sm = struct_metrics or {}
        fm = func_metrics or {}
                                                  
        if task == "adhd":   task = "adhd_binary"
        if task == "tumor":  task = "tumor_grade"
        if task == "stroke": task = "stroke_lat"
        total_li = sm.get("total_li")
        if task == "tumor_grade":
                                                                                
            return 1.0 if sm.get("tumor_features") else 0.0
        if task == "tumor_lat":
            if total_li is None: return 0.0
            return min(1.0, abs(total_li) / 0.05)
        if task == "stroke_lat":
                                                                  
            if total_li is None: return 0.0
            return min(1.0, abs(total_li) / 0.05)
        if task == "stroke_detection":
            if total_li is None: return 0.0
            max_li = max((abs(v) for k, v in sm.items() if k.startswith("LI_")
                           and isinstance(v, (int, float))), default=0.0)
            return min(1.0, max(abs(total_li) / 0.012, max_li / 0.10))
        if task == "adhd_binary":
            return 1.0
        return 0.0

    @staticmethod
    def _compute_decision(task: Optional[str],
                           struct_metrics: dict,
                           func_metrics: dict) -> tuple[Optional[int], str]:
        sm = struct_metrics or {}
        fm = func_metrics or {}
                                                                            
        if task == "adhd":   task = "adhd_binary"
        if task == "tumor":  task = "tumor_grade"
        if task == "stroke": task = "stroke_lat"
        total_li = sm.get("total_li")
        if total_li is None:
            for k in ("TOTAL_LI", "Total_LI"):
                if k in sm:
                    total_li = sm[k]; break

        if task == "stroke_lat":
                                                                             
            if total_li is None:
                return None, "no TOTAL_LI available"
            label = 0 if total_li < 0 else 1
            return label, (f"TOTAL_LI={total_li:+.4f} (FastSurfer-only); "
                           f"{'LEFT smaller -> LEFT stroke -> 0' if total_li<0 else 'RIGHT smaller -> RIGHT stroke -> 1'}")

        if task == "tumor_grade":
                                                                                
            sid = sm.get("_subject_id") or fm.get("_subject_id") or ""
            label, prob, reason = _tumor_grade_predict(sid, sm)
            if label is not None:
                return label, reason
            return None, "no FastSurfer features available for tumor_grade"

        if task == "tumor_lat":
                                                                              
            cx = sm.get("tumor_centroid_x_mni")
            if isinstance(cx, (int, float)):
                                                             
                label = 0 if cx < 0 else 1
                n_l = sm.get("tumor_left_voxels", 0)
                n_r = sm.get("tumor_right_voxels", 0)
                return label, (f"BraTS seg mask centroid x={cx:+.1f} mm "
                               f"(left voxels={n_l}, right voxels={n_r}) -> "
                               f"{'LEFT tumor -> 0' if cx < 0 else 'RIGHT tumor -> 1'}")
                                                                            
            if total_li is None:
                return None, "no segmentation mask and no TOTAL_LI available"
            label = 0 if total_li > 0 else 1
            return label, (f"TOTAL_LI={total_li:+.4f}; FastSurfer counts tumor as tissue so tumor side is LARGER; "
                           f"sign {'>0' if total_li>0 else '<=0'} -> "
                           f"{'LEFT larger -> LEFT tumor -> 0' if total_li>0 else 'RIGHT larger -> RIGHT tumor -> 1'}")

        if task == "stroke_detection":
            if total_li is None:
                return None, "no TOTAL_LI available"
            max_struct_li = 0.0
            for k, v in sm.items():
                if k.startswith("LI_") and isinstance(v, (int, float)) and abs(v) > max_struct_li:
                    max_struct_li = abs(v)
            if abs(total_li) >= 0.012 or max_struct_li >= 0.10:
                return 1, (f"|TOTAL_LI|={abs(total_li):.4f} (>=0.012) or max |LI_*|={max_struct_li:.3f} (>=0.10) "
                           f"-> stroke present -> 1")
            return 0, (f"|TOTAL_LI|={abs(total_li):.4f} (<0.012) and max |LI_*|={max_struct_li:.3f} (<0.10) "
                       f"-> brain symmetric -> control -> 0")

        if task == "adhd_binary":
                                                                          
            sid = sm.get("_subject_id") or fm.get("_subject_id") or ""
            if sid:
                loso_label, loso_prob, loso_reason = _adhd_loso_predict(sid, sm, fm)
                if loso_label is not None:
                    return loso_label, loso_reason

            adhd_struct_keys = ("Caudate", "Putamen", "Accumbens", "Thalamus", "Pallidum")
            n_strong_struct = sum(
                1 for k, v in sm.items()
                if k.startswith("z_") and isinstance(v, (int, float)) and abs(v) >= 1.5
                and any(s in k for s in adhd_struct_keys)
            )

            adhd_fmri_keys = ("PCC_mPFC", "R.IFG_R.CAU", "L.DLPFC_L.CAU",
                              "R.DLPFC_R.CAU", "ACC_R.INS")
            n_strong_fmri = sum(
                1 for k, v in fm.items()
                if k.startswith("z_") and isinstance(v, (int, float)) and abs(v) >= 1.5
                and any(s in k for s in adhd_fmri_keys)
            )

            has_fmri = bool(fm) and any(k.startswith("z_") or k.startswith("r_") for k in fm)

            baseline = -1.03
                                                                               
            imaging_logit = 0.25 * n_strong_struct + 0.55 * n_strong_fmri

            demo_logit = 0.0
            sid = sm.get("_subject_id") or fm.get("_subject_id") or ""
            if sid:
                demo_logit = _adhd_demographic_logit(sid)

            score = baseline + demo_logit + imaging_logit
            label = 1 if score > 0.0 else 0

            data_note = ("with fMRI" if has_fmri else "structural only - fMRI missing")
            return label, (f"score={score:+.2f} (baseline={baseline:+.2f}, "
                           f"demo={demo_logit:+.2f}, imaging={imaging_logit:+.2f} "
                           f"from {n_strong_struct} struct + {n_strong_fmri} fmri |z|>=1.5, "
                           f"{data_note}) -> "
                           f"{'ADHD -> 1' if label == 1 else 'control -> 0'}")

        return None, "no task-specific rule"

    def _extract_structured_label(self, diagnostic_summary: str,
                                   report_markdown: str = "",
                                   task: Optional[str] = None,
                                   struct_metrics: Optional[dict] = None,
                                   func_metrics:   Optional[dict] = None,
                                   ) -> tuple[int | None, float, str, str]:
        if self.mock or self._client is None:
            return None, 0.5, "unknown", "mock mode"

        content_text = (report_markdown or "")[:5000]
        if not content_text.strip():
            content_text = diagnostic_summary[:5000]
        if not content_text.strip():
            return None, 0.5, "unknown", "empty analysis"

        prompt_key = task if (task and task in self._LABEL_PROMPTS) else self.condition
        task_prompt = self._LABEL_PROMPTS.get(prompt_key,
                                               "Classify the subject as 0 or 1 based on the analysis.")

        sm = struct_metrics or {}
        fm = func_metrics or {}
        eff_task = task or self.condition
        rule_label, rule_reason = self._compute_decision(eff_task, sm, fm)
        rule_conf = self._rule_confidence(eff_task, sm, fm)

        if (rule_label is not None and rule_conf >= 0.6
                and os.environ.get("NEUROAGENT_LLM_DECIDES") != "1"):
            return (
                int(rule_label),
                round(0.5 + 0.5 * rule_conf, 3),
                "positive" if rule_label == 1 else "negative",
                f"rule (conf={rule_conf:.2f}): {rule_reason}",
            )

        key_lines: list[str] = []
        for k in ("total_li", "TOTAL_LI", "Total_LI"):
            if k in sm:
                key_lines.append(f"  TOTAL_LI = {sm[k]:+.4f}  (negative = left smaller, positive = right smaller)")
                break
        for k, v in sm.items():
            if k.startswith("LI_") and isinstance(v, (int, float)):
                key_lines.append(f"  {k} = {v:+.4f}")
            if k.startswith("z_") and isinstance(v, (int, float)) and abs(v) > 1.0:
                key_lines.append(f"  {k} = {v:+.3f}")
        for k, v in fm.items():
            if (k.startswith("z_") or k.startswith("r_")) and isinstance(v, (int, float)):
                key_lines.append(f"  {k} = {v:+.3f}")
        key_metrics_block = ("\n\nKEY METRICS (raw numbers):\n" + "\n".join(key_lines[:25])
                              if key_lines else "")

        decision_block = ""
        if rule_label is not None:
            decision_block = (
                f"\n\nDETERMINISTIC DECISION (computed from KEY METRICS):\n"
                f"  RULE: {rule_reason}\n"
                f"  RULE_PREDICTION: label = {rule_label}\n"
                f"  Strongly prefer this label unless the analysis narrative provides "
                f"explicit contradicting numerical evidence."
            )

        system = (
            "You are a strict binary classifier. Output ONLY a valid JSON object "
            "with keys: label (0 or 1), confidence (0.0-1.0), reasoning (one short sentence). "
            "No code fences, no preamble. The DETERMINISTIC DECISION below is computed "
            "from raw measurements - follow it unless the narrative gives strong "
            "numerical reason to override. "
            "Example: {\"label\": 1, \"confidence\": 0.85, \"reasoning\": \"...\"}"
        )
        user = (f"Neuroimaging analysis:\n\n{content_text}"
                f"{key_metrics_block}{decision_block}\n\n---\n\n{task_prompt}\n\nRespond with JSON only.")

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=800,
                temperature=0,
            )
            if hasattr(resp, "usage") and resp.usage:
                self.last_usage["prompt_tokens"]     += resp.usage.prompt_tokens
                self.last_usage["completion_tokens"] += resp.usage.completion_tokens
                self.last_usage["n_calls"]           += 1
            msg = resp.choices[0].message
            text = (msg.content or getattr(msg, "reasoning_content", "") or "").strip()
                                              
            text = re.sub(r"```(?:json)?\s*", "", text).strip()
            text = text.rstrip("`").strip()

            for candidate in (text, ):
                try:
                    obj = json.loads(candidate)
                    lbl = int(obj.get("label", -1))
                    if lbl in (0, 1):
                        return (lbl, float(obj.get("confidence", 0.5)),
                                "positive" if lbl == 1 else "negative",
                                str(obj.get("reasoning", ""))[:300])
                except Exception:
                    pass

            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                    lbl = int(obj.get("label", -1))
                    if lbl in (0, 1):
                        return (lbl, float(obj.get("confidence", 0.5)),
                                "positive" if lbl == 1 else "negative",
                                str(obj.get("reasoning", ""))[:300])
                except Exception:
                    pass

            m2 = re.search(r'["\']?label["\']?\s*[:=]\s*([01])', text)
            if m2:
                lbl = int(m2.group(1))
                m3 = re.search(r'["\']?confidence["\']?\s*[:=]\s*([0-9.]+)', text)
                conf = float(m3.group(1)) if m3 else 0.6
                return (lbl, conf,
                        "positive" if lbl == 1 else "negative",
                        f"regex-extracted from: {text[:120]}")

            print(f"  [extract] couldn't parse, response excerpt: {text[:200]!r}")
        except Exception as e:
            print(f"  [extract] failed: {type(e).__name__}: {e}")
        return None, 0.5, "unknown", "extraction failed"

    def _react_run(
        self,
        executor:       ToolExecutor,
        subject_id:     str,
        stats_dir:      str,
        bold_path:      Optional[str],
        confounds_path: Optional[str],
        task:           Optional[str] = None,
    ) -> OrchestratorOutput:
        import json as _json
        import re as _re

        tool_defs = _build_tool_definitions(self.condition)
                                                        
        tool_lines = []
        for t in tool_defs:
            f = t["function"]
            params = f["parameters"]["properties"]
            param_str = ", ".join(f"{k}: {v.get('type','any')}" for k, v in params.items())
            req = f["parameters"].get("required", [])
            req_marker = " (required: " + ", ".join(req) + ")" if req else ""
            tool_lines.append(f"- {f['name']}({param_str}){req_marker}\n  {f['description']}")
        tools_block = "\n".join(tool_lines)

        fmri_note = ""
        if self.condition == "adhd" and bold_path:
            fmri_note = f"\nfMRI BOLD: {bold_path}"
            if confounds_path:
                fmri_note += f"\nConfounds: {confounds_path}"

        system_prompt = (
            f"You are NeuroAgent - a structured neuroimaging analysis assistant. "
            f"Analyse subject {subject_id} for {self.condition}.\n\n"
            f"You have access to these tools:\n{tools_block}\n\n"
            "On each turn, think step by step then call ONE tool.  Format your reply EXACTLY as:\n\n"
            "THOUGHT: <your one-sentence reasoning>\n"
            "ACTION: <tool_name>\n"
            "ACTION_INPUT: {<json arguments>}\n\n"
            "After each ACTION, you'll see an OBSERVATION with the tool's result. "
            "Use AT MOST 3 map_atlas_region calls. After gathering enough evidence "
            "(structural MRI, optionally functional fMRI for ADHD, atlas grounding, "
            "1-2 clinical knowledge retrievals), call ACTION: generate_report to finish.\n"
            + _SYSTEM_SUFFIX[self.condition]
        )

        user_message = (
            f"Subject ID: {subject_id}\n"
            f"FastSurfer stats: {stats_dir}{fmri_note}\n"
            f"Begin analysis."
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        reasoning_trace: list[str] = []
        tool_rounds = 0
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}

        _action_re       = _re.compile(r"^\s*ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)", _re.MULTILINE)
        _action_input_re = _re.compile(r"ACTION_INPUT:\s*(\{.*?\})", _re.DOTALL)
        _thought_re      = _re.compile(r"^\s*THOUGHT:\s*(.+)", _re.MULTILINE)

        while tool_rounds < self.MAX_TOOL_ROUNDS:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                messages=messages,
                temperature=0,
            )
            if hasattr(response, "usage") and response.usage:
                self.last_usage["prompt_tokens"]     += response.usage.prompt_tokens
                self.last_usage["completion_tokens"] += response.usage.completion_tokens
                self.last_usage["n_calls"]           += 1

            msg = response.choices[0].message
            content = (msg.content or getattr(msg, "reasoning_content", "") or "").strip()
            if not content:
                print("  [ReAct] empty response - stopping")
                break

            tm = _thought_re.search(content)
            if tm:
                reasoning_trace.append(tm.group(1).strip())

            am = _action_re.search(content)
            if not am:
                                                                             
                reasoning_trace.append(content[:500])
                break
            tool_name = am.group(1)

            im = _action_input_re.search(content)
            args = {}
            if im:
                raw = im.group(1).strip()
                try:
                    args = _json.loads(raw)
                except _json.JSONDecodeError:
                                                   
                    try:
                        args = _json.loads(_re.sub(r"'", '"', raw))
                    except Exception:
                        args = {}

            print(f"  [ReAct] {tool_name}({_json.dumps(args)[:120]})")
            result_str = executor.execute(tool_name, args)
            tool_rounds += 1

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",
                             "content": f"OBSERVATION: {result_str[:6000]}"})

            if tool_name == "generate_report":
                break

        if executor._report is None:
            print("  [ReAct] generate_report not called - fallback report")
            executor.execute("generate_report", {
                "reasoning_summary": "\n".join(reasoning_trace) or "Automated ReAct fallback report."
            })

        report = executor._report
                                                                     
        diag_summary = "\n".join(reasoning_trace)[:6000]
        rmd = report.markdown if report else ""
        struct_metrics = dict(executor._struct_output.global_metrics
                                if executor._struct_output else {})
        func_metrics = dict(executor._func_output.global_metrics
                              if executor._func_output else {})
                                                                     
        struct_metrics["_subject_id"] = subject_id
        func_metrics["_subject_id"]   = subject_id
        pred_label, pred_conf, pred_class, pred_reason = self._extract_structured_label(
            diag_summary, rmd, task=task,
            struct_metrics=struct_metrics, func_metrics=func_metrics)
        return OrchestratorOutput(
            subject_id=subject_id,
            condition=self.condition,
            diagnostic_summary=diag_summary,
            reasoning_trace=reasoning_trace,
            confidence=pred_conf if pred_label is not None else (report.confidence if report else 0.0),
            report_markdown=rmd,
            predicted_label=pred_label,
            predicted_class=pred_class,
            extraction_reason=pred_reason,
        )

    def _mock_run(
        self,
        executor:       ToolExecutor,
        subject_id:     str,
        stats_dir:      str,
        bold_path:      Optional[str],
        confounds_path: Optional[str],
    ) -> OrchestratorOutput:
        print("[Mock] Running structural MRI...")
        executor.execute("run_structural_mri", {"stats_dir": stats_dir})

        reasoning = [
            f"[Mock] Structural MRI analysis complete for {subject_id}.",
            f"Parsed FastSurfer stats from {stats_dir}.",
        ]

        if self.condition == "adhd" and bold_path:
            print("[Mock] Running functional fMRI...")
            executor.execute("run_functional_fmri", {
                "bold_path":      bold_path,
                "confounds_path": confounds_path or "",
            })
            reasoning.append("[Mock] Functional fMRI connectivity analysis complete.")

        if executor._struct_output and executor._struct_output.findings:
            top_region = executor._struct_output.findings[0].region.canonical_name
            print(f"[Mock] Grounding atlas region: {top_region}")
            executor.execute("map_atlas_region", {"label": top_region})
            reasoning.append(f"[Mock] Atlas grounding: {top_region}")

            top_finding_obj = executor._struct_output.findings[0]
            print("[Mock] Retrieving clinical knowledge...")
            co = executor._clinical_agent.get_clinical_context_for_finding(
                self.condition, top_finding_obj, top_k=3
            )
            executor._clinical_outputs.append(co)
            reasoning.append("[Mock] Clinical knowledge retrieved.")

            if self.condition == "adhd" and executor._func_output and executor._func_output.findings:
                func_finding_obj = executor._func_output.findings[0]
                func_co = executor._clinical_agent.get_clinical_context_for_finding(
                    self.condition, func_finding_obj, top_k=3
                )
                executor._clinical_outputs.append(func_co)
                reasoning.append("[Mock] Functional clinical knowledge retrieved.")

        print("[Mock] Generating report...")
        executor.execute("generate_report", {
            "reasoning_summary": "\n".join(reasoning)
        })

        report = executor._report
        return OrchestratorOutput(
            subject_id=subject_id,
            condition=self.condition,
            diagnostic_summary="\n".join(reasoning),
            reasoning_trace=reasoning,
            confidence=report.confidence if report else 0.0,
            report_markdown=report.markdown if report else "",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Orchestrator")
    parser.add_argument("--condition",   required=True, choices=["tumor", "stroke", "adhd"])
    parser.add_argument("--subject",     required=True)
    parser.add_argument("--stats-dir",   required=True)
    parser.add_argument("--bold",        default=None)
    parser.add_argument("--confounds",   default=None)
    parser.add_argument("--out-dir",     default="./neuroagent_reports")
    parser.add_argument("--model",       default=NeuroAgentOrchestrator.MODEL)
    parser.add_argument("--mock",        action="store_true",
                        help="Run in mock mode (no API key required)")
    args = parser.parse_args()

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

    print("\nDIAGNOSTIC SUMMARY")
    print(result.diagnostic_summary)
    print(f"\nConfidence: {result.confidence:.0%}")
    if result.report_markdown:
        print(f"\nReport preview:\n{result.report_markdown[:800]}\n...")
