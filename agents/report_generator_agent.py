from __future__ import annotations

import html as _html_mod
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.schemas import AgentOutput, Finding, Condition

@dataclass
class DiagnosticReport:
    subject_id:   str
    condition:    Condition
    timestamp:    str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    markdown:     str   = ""
    html_report:  str   = ""
    json_report:  dict  = field(default_factory=dict)
    confidence:   float = 0.0
    key_findings: list[str] = field(default_factory=list)
    assessment:   str   = ""
    limitations:  list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "subject_id":   self.subject_id,
            "condition":    self.condition,
            "timestamp":    self.timestamp,
            "confidence":   self.confidence,
            "key_findings": self.key_findings,
            "assessment":   self.assessment,
            "limitations":  self.limitations,
            "report":       self.json_report,
        }, indent=indent, default=str)

    def save(self, output_dir: str | Path) -> tuple[Path, Path, Path, Optional[Path]]:
        patient_dir = Path(output_dir) / self.subject_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        md_path   = patient_dir / f"{self.condition}.md"
        json_path = patient_dir / f"{self.condition}.json"
        html_path = patient_dir / f"{self.condition}.html"
        pdf_path  = patient_dir / f"{self.condition}.pdf"
        md_path.write_text(self.markdown,      encoding="utf-8")
        json_path.write_text(self.to_json(),   encoding="utf-8")
        html_path.write_text(self.html_report, encoding="utf-8")
        try:
            from weasyprint import HTML
            HTML(string=self.html_report, base_url=str(patient_dir)).write_pdf(str(pdf_path))
        except Exception as e:
            print(f"  [Report] PDF export failed: {e}")
            pdf_path = None
        return md_path, json_path, html_path, pdf_path

_SEV_SYMBOLS = {"normal": "✓", "mild": "⚠", "moderate": "⚠⚠", "severe": "⛔", "unknown": "?"}

_SEV_HTML = {
    "normal":   {"bg": "#dcfce7", "text": "#166534", "border": "#22c55e", "label": "Normal"},
    "mild":     {"bg": "#fef9c3", "text": "#854d0e", "border": "#eab308", "label": "Mild"},
    "moderate": {"bg": "#ffedd5", "text": "#9a3412", "border": "#f97316", "label": "Moderate"},
    "severe":   {"bg": "#fee2e2", "text": "#991b1b", "border": "#ef4444", "label": "Severe"},
    "unknown":  {"bg": "#f1f5f9", "text": "#475569", "border": "#94a3b8", "label": "Unknown"},
}

_CONDITION_META = {
    "tumor":  {"label": "Brain Tumor - BraTS-GLI",    "color": "#7c3aed", "light": "#ede9fe", "icon": "🧬"},
    "stroke": {"label": "Ischemic Stroke - ATLAS-v2", "color": "#dc2626", "light": "#fee2e2", "icon": "⚡"},
    "adhd":   {"label": "ADHD - ADHD-200",            "color": "#1d4ed8", "light": "#dbeafe", "icon": "🧠"},
}

def _esc(s: str) -> str:
    return _html_mod.escape(str(s))

def _sev_badge_html(sev: str) -> str:
    c = _SEV_HTML.get(sev, _SEV_HTML["unknown"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'font-size:0.78em;font-weight:700;letter-spacing:0.04em;'
        f'background:{c["bg"]};color:{c["text"]};border:1px solid {c["border"]}">'
        f'{c["label"].upper()}</span>'
    )

def _findings_table_md(findings: list[Finding]) -> str:
    if not findings:
        return "_No significant findings._\n"
    lines = [
        "| Status | Region | Metric | Value | Clinical Significance |",
        "|--------|--------|--------|-------|-----------------------|",
    ]
    for f in findings:
        sym = _SEV_SYMBOLS.get(f.severity, "?")
        sig = f.clinical_significance[:100] + ("..." if len(f.clinical_significance) > 100 else "")
        lines.append(
            f"| {sym} {f.severity.upper()} "
            f"| {f.region.canonical_name} ({f.region.hemisphere}) "
            f"| {f.measurement.metric} "
            f"| {f.measurement.value:.3g} {f.measurement.units} "
            f"| {sig} |"
        )
    return "\n".join(lines) + "\n"

def _evidence_chain_md(findings: list[Finding]) -> str:
    if not findings:
        return ""
    lines = []
    for f in findings:
        lines += [f"#### {f.finding_id}", f"**{f.region.canonical_name}** - {f.severity.upper()}", ""]
        for ev in f.evidence_chain:
            lines.append(f"- {ev}")
        lines.append("")
    return "\n".join(lines)

def _tumor_assessment(findings, global_metrics):
    key, conf = [], 0.5
    li = global_metrics.get("wm_lateralisation_index", 0)
    if abs(li) > 0.05:
        dom = "left" if li > 0 else "right"
        key.append(f"White matter volume asymmetry (LI={li:+.3f}): {dom} hemisphere larger - possible mass effect or oedema on contralateral side.")
        conf = 0.6
    vent = global_metrics.get("total_lateral_ventricle_mm3", 0)
    if vent > 50_000:
        key.append(f"Enlarged lateral ventricles ({vent:.0f} mm³): possible obstructive hydrocephalus.")
        conf = 0.7
    lobar = global_metrics.get("lobar_volume_mm3", {})
    if lobar:
        dom_lobe = max(lobar, key=lobar.get)
        key.append(f"Largest lobar volume: {dom_lobe} ({lobar[dom_lobe]:.0f} mm³).")
    cc = global_metrics.get("corpus_callosum_total_mm3")
    if cc is not None:
        key.append(f"Corpus callosum total volume: {cc:.0f} mm³ ({'possible infiltration flag' if cc < 3000 else 'within range'}).")
    assessment = (
        "Structural MRI analysis reveals "
        + (", ".join(key[:2]) if key else "no significant structural abnormalities.")
        + " BraTS-GLI ground truth segmentation (ET, TC, WT subregions) is required for "
        "definitive tumor characterization and grading. Atlas-grounded lobar localisation "
        "provides surgical planning context (eloquent cortex proximity, vascular territory). "
        "Molecular markers (IDH, MGMT, EGFR) from histopathology are needed for WHO 2021 "
        "classification and treatment planning."
    )
    limitations = [
        "FastSurfer volumes reflect gross anatomy, not tumor-specific segmentation.",
        "Gadolinium enhancement (T1ce) and FLAIR patterns require separate segmentation model.",
        "Peritumoral oedema and mass effect require BraTS-format segmentation labels for quantification.",
        "Surgical risk assessment requires DTI tractography (not available in this pipeline).",
    ]
    return assessment, key, conf, limitations

def _stroke_assessment(findings, global_metrics):
    key, conf = [], 0.5
    li_keys = {k: v for k, v in global_metrics.items() if k.startswith("LI_") and isinstance(v, float)}
    has_significant_asymmetry = False
    if li_keys:
        worst_key = max(li_keys, key=lambda k: abs(li_keys[k]))
        worst_li  = li_keys[worst_key]
        if abs(worst_li) > 0.08:
            has_significant_asymmetry = True
            struct    = worst_key.replace("LI_", "")
            side      = "right" if worst_li > 0 else "left"
            key.append(f"Greatest volume asymmetry in {struct} (LI={worst_li:+.3f}, {side} side larger) - suggests lesion on {'left' if worst_li > 0 else 'right'} side.")
            conf = 0.65
    if findings:
        territories = set()
        for f in findings:
            if f.region.stroke_territory:
                territories.add(f.region.stroke_territory.split(";")[0].split(".")[0][:80])
        if territories:
            key.append("Implicated vascular territories: " + "; ".join(list(territories)[:3]))
            conf = 0.7
    bs = global_metrics.get("brainstem_vol_mm3")
    if bs and bs < 18_000:
        key.append(f"Brainstem volume reduced ({bs:.0f} mm3) - consistent with posterior circulation involvement.")
    if has_significant_asymmetry or findings:
        opener = (
            "Structural MRI analysis identifies hemispheric volume asymmetries "
            "consistent with focal ischaemic lesion(s). "
        )
    else:
        opener = (
            "Structural MRI analysis shows hemispheric volumes within the symmetric range "
            "(|TOTAL LI| below 0.08 threshold); no morphometric evidence of focal stroke "
            "from this scan. "
        )
    assessment = (
        opener
        + (", ".join(key[:2]) if key else "")
        + " Vascular territory inference is atlas-based and should be confirmed with lesion mask "
        "segmentation. ATLAS-v2 manual lesion masks provide ground truth for VLSM analysis. "
        "Predicted functional deficits (motor, language, spatial cognition) are based on "
        "lesion location in the atlas-to-territory mapping. "
        "NIHSS score, clinical examination, and longitudinal imaging are required for "
        "complete outcome prediction."
    )
    limitations = [
        "T1 volume asymmetry is an indirect proxy for lesion location - direct lesion mask segmentation is more accurate.",
        "Chronic vs acute stroke cannot be distinguished from T1 alone without DWI.",
        "Atrophy from Wallerian degeneration may confound volume measurements in ipsilateral pathways.",
        "Multi-lesion stroke (lacunar burden) requires lesion count, not just asymmetry.",
    ]
    return assessment, key, conf, limitations

def _adhd_assessment(struct_findings, func_findings, struct_metrics, func_metrics):
    key, conf = [], 0.5

    z_keys = {k: v for k, v in struct_metrics.items()
              if k.startswith("z_") and isinstance(v, (int, float))}
    strong_struct = {k: v for k, v in z_keys.items() if abs(v) >= 1.5}

    if strong_struct:
        worst = max(strong_struct, key=lambda k: abs(strong_struct[k]))
        z_val = strong_struct[worst]
        name = worst.replace("z_", "").replace("-", " ")
        direction = "below" if z_val < 0 else "above"
        key.append(
            f"{name} ICV-corrected volume {direction} same-site control range "
            f"(z={z_val:+.2f})."
        )
        conf = 0.55

    n_strong_func = 0
    if func_metrics:
        for k, v in func_metrics.items():
            if k.startswith("z_") and isinstance(v, (int, float)) and abs(v) >= 1.5:
                n_strong_func += 1
                pair = k.replace("z_", "")
                key.append(
                    f"{pair} resting-state connectivity {('reduced' if v < 0 else 'elevated')} "
                    f"vs control norm (z={v:+.2f})."
                )

    if len(strong_struct) >= 2 and n_strong_func >= 2:
        conf = min(0.75, conf + 0.2)
        key.append(
            f"Convergent structural ({len(strong_struct)} regions |z|>=1.5) and "
            f"functional ({n_strong_func} circuits |z|>=1.5) deviations - pattern "
            f"consistent with the frontostriatal/DMN ADHD model."
        )
    elif len(strong_struct) + n_strong_func >= 1:
                                                                                 
        key.append(
            "Number of |z|>=1.5 deviations is within the range expected for "
            "control subjects under multiple-comparisons correction."
        )

    has_real_fmri = bool(func_findings) or any(
        k.startswith("r_") for k in (func_metrics or {})
    )
    fmri_clause = (
        " Resting-state connectivity is compared to same-site ADHD-200 control norms."
        if has_real_fmri else
        " No resting-state fMRI for this subject; assessment rests on structural morphometry alone."
    )

    if len(strong_struct) >= 2 and n_strong_func >= 2:
        opener = (
            "Imaging shows convergent structural and functional deviations vs "
            "same-site ADHD-200 controls; pattern is compatible with - but not "
            "diagnostic of - ADHD."
        )
    elif key:
        opener = (
            "Imaging shows isolated deviations vs same-site ADHD-200 controls. "
            "Such isolated deviations are common in healthy controls and do not "
            "support an ADHD interpretation on their own."
        )
    else:
        opener = (
            "Imaging shows no subcortical volume or connectivity deviations "
            "vs same-site ADHD-200 controls; findings do not support ADHD."
        )

    assessment = (
        opener
        + " Volumes are ICV-corrected and z-scored against per-site ADHD-200 control distributions."
        + fmri_clause
        + " ADHD effect sizes from morphometry are small (Hoogman 2017 d=-0.11 to -0.19); "
        "imaging alone cannot diagnose ADHD - DSM-5 criteria (clinical interview, "
        "symptom rating scales, multi-informant assessment) are required."
    )

    limitations = [
        "Norms are computed from ADHD-200 controls, per site; cross-site generalization requires ComBat or equivalent.",
        "Individual-level z-scores have wide confidence intervals; group-level ADHD effects are d=-0.1 to -0.2.",
        "Medication status at scan time is a critical unmeasured confounder.",
        "fMRI connectivity uses estimated TR - verify from BIDS metadata.",
    ]
    return assessment, key, conf, limitations

class ReportGeneratorAgent:

    def generate(
        self,
        condition:         Condition,
        subject_id:        str,
        structural_output: Optional[AgentOutput] = None,
        functional_output: Optional[AgentOutput] = None,
        clinical_outputs:  Optional[list[AgentOutput]] = None,
        extra_context:     str = "",
    ) -> DiagnosticReport:
        t0 = time.time()

        struct_findings = structural_output.findings if structural_output else []
                                                                                
        struct_metrics = {}
        if structural_output:
            struct_metrics.update(structural_output.metadata or {})
            struct_metrics.update(structural_output.global_metrics or {})
        func_findings = functional_output.findings if functional_output else []
        func_metrics  = {}
        if functional_output:
            func_metrics.update(functional_output.metadata or {})
            func_metrics.update(functional_output.global_metrics or {})

        if condition == "tumor":
            assessment, key_findings, confidence, limitations = _tumor_assessment(struct_findings, struct_metrics)
        elif condition == "stroke":
            assessment, key_findings, confidence, limitations = _stroke_assessment(struct_findings, struct_metrics)
        else:
            assessment, key_findings, confidence, limitations = _adhd_assessment(
                struct_findings, func_findings, struct_metrics, func_metrics
            )

        md = self._build_markdown(
            condition=condition, subject_id=subject_id,
            struct_findings=struct_findings, struct_metrics=struct_metrics,
            func_findings=func_findings, func_metrics=func_metrics,
            clinical_outputs=clinical_outputs or [],
            key_findings=key_findings, assessment=assessment,
            confidence=confidence, limitations=limitations,
            extra_context=extra_context,
        )

        html = self._build_html(
            condition=condition, subject_id=subject_id,
            struct_findings=struct_findings, struct_metrics=struct_metrics,
            func_findings=func_findings, func_metrics=func_metrics,
            clinical_outputs=clinical_outputs or [],
            key_findings=key_findings, assessment=assessment,
            confidence=confidence, limitations=limitations,
            extra_context=extra_context,
        )

        json_report = {
            "structural": {
                "n_findings": len(struct_findings),
                "global_metrics": struct_metrics,
                "findings": [f.to_dict() for f in struct_findings],
            },
            "functional": {
                "n_findings": len(func_findings),
                "global_metrics": func_metrics,
                "findings": [f.to_dict() for f in func_findings],
            },
            "clinical_knowledge": {
                "passages_retrieved": sum(len(o.findings) for o in (clinical_outputs or [])),
            },
            "processing_time_s": round(time.time() - t0, 2),
        }

        return DiagnosticReport(
            subject_id=subject_id, condition=condition,
            markdown=md, html_report=html, json_report=json_report,
            confidence=confidence, key_findings=key_findings,
            assessment=assessment, limitations=limitations,
        )

    def _build_markdown(self, condition, subject_id, struct_findings, struct_metrics,
                         func_findings, func_metrics, clinical_outputs,
                         key_findings, assessment, confidence, limitations, extra_context):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cond_title = {
            "tumor":  "Brain Tumor (BraTS-GLI)",
            "stroke": "Ischemic Stroke (ATLAS-v2)",
            "adhd":   "ADHD (ADHD-200)",
        }.get(condition, condition.upper())

        lines = [
            "# NeuroAgent Diagnostic Report", "",
            f"**Subject ID:** `{subject_id}`  ",
            f"**Condition:** {cond_title}  ",
            f"**Generated:** {now}  ",
            f"**Confidence:** {confidence:.0%}",
            "", "---", "", "## Key Findings", "",
        ]
        for kf in (key_findings or ["No significant abnormalities detected."]):
            lines.append(f"- {kf}")
        lines += ["", "---", "", "## Assessment", "", assessment, "", "---",
                  "", "## Structural MRI Findings", "",
                  f"**Structures parsed:** {struct_metrics.get('n_structures_parsed', 'N/A')}  ",
                  f"**ICV:** {struct_metrics.get('icv_mm3', 0):.0f} mm³", "",
                  _findings_table_md(struct_findings)]
        if struct_findings:
            lines += ["", "### Evidence Chains", "", _evidence_chain_md(struct_findings)]
        if func_findings or (func_metrics and "note" not in func_metrics):
            lines += ["---", "", "## Functional fMRI Findings", ""]
            if func_metrics:
                lines.append(f"**Timepoints:** {func_metrics.get('n_timepoints', 'N/A')}  "
                             f"**TR:** {func_metrics.get('t_r_s', 'N/A')} s")
            lines += ["", _findings_table_md(func_findings)]
            if func_findings:
                lines += ["", "### Evidence Chains", "", _evidence_chain_md(func_findings)]
            dmn = func_metrics.get("dmn_deactivation_failure_index")
            if dmn is not None:
                lines += ["", f"**DMN Deactivation Failure Index:** {dmn:+.3f}",
                          "_(>0 = hypercoherent DMN; consistent with ADHD task-suppression failure)_"]
        if clinical_outputs:
            lines += ["", "---", "", "## Clinical Knowledge Integration", "",
                      "_Retrieved via RAG from embedded PubMed corpus._", ""]
            for co in clinical_outputs:
                for f in co.findings[:3]:
                    lines += [f"### {f.finding_id}", f.evidence_chain[0][:500], ""]
        if extra_context:
            lines += ["---", "", "## Orchestrator Reasoning Trace", "", extra_context, ""]
        all_findings = struct_findings + func_findings
        if all_findings:
            regions = {f.region.canonical_name: f.region for f in all_findings}
            lines += ["---", "", "## Atlas-Grounded Anatomical Summary", "",
                      "| Region | Hemisphere | Lobe | Networks |",
                      "|--------|------------|------|----------|"]
            for name, reg in regions.items():
                nets = ", ".join(reg.networks[:2]) if reg.networks else "-"
                lines.append(f"| {name} | {reg.hemisphere} | {reg.lobe} | {nets} |")
            lines.append("")
        lines += ["", "---", "", "## Limitations", ""]
        for lim in limitations:
            lines.append(f"- {lim}")
        lines += ["", "---", "",
                  "_Report generated by NeuroAgent - a multi-agent neuro-imaging framework._  ",
                  "_This output is for research purposes only and does not constitute clinical advice._"]
        return "\n".join(lines)

    def _build_html(self, condition, subject_id, struct_findings, struct_metrics,
                     func_findings, func_metrics, clinical_outputs,
                     key_findings, assessment, confidence, limitations, extra_context):
        meta   = _CONDITION_META.get(condition, _CONDITION_META["adhd"])
        now    = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
        color  = meta["color"]
        light  = meta["light"]
        conf_pct = int(confidence * 100)
        conf_color = "#166534" if conf_pct >= 65 else "#854d0e" if conf_pct >= 45 else "#991b1b"

        def section(title, body, id_=""):
            id_attr = f' id="{id_}"' if id_ else ""
            return f"""
      <section class="card"{id_attr}>
        <div class="card-header">
          <h2>{_esc(title)}</h2>
        </div>
        <div class="card-body">
          {body}
        </div>
      </section>"""

        def findings_table(findings: list[Finding]) -> str:
            if not findings:
                return '<p class="muted">No significant findings.</p>'
            rows = ""
            for f in findings:
                ref_range = ""
                if f.measurement.reference_min is not None:
                    ref_range = (f"{f.measurement.reference_min:.3g} to "
                                 f"{f.measurement.reference_max:.3g} {_esc(f.measurement.units)}")
                chains = "".join(f"<li>{_esc(e)}</li>" for e in f.evidence_chain)
                sig = f.clinical_significance.replace("-", " - ").replace("\u2013", " - ")
                rows += f"""
              <tr>
                <td class="col-sev">{_sev_badge_html(f.severity)}</td>
                <td class="col-region">
                  <strong>{_esc(f.region.canonical_name)}</strong><br>
                  <small class="muted">{_esc(f.region.hemisphere)}, {_esc(f.region.lobe)}</small>
                </td>
                <td class="col-metric"><code>{_esc(f.measurement.metric)}</code></td>
                <td class="col-val num"><strong>{f.measurement.value:.4g}</strong><br><small>{_esc(f.measurement.units)}</small></td>
                <td class="col-ref">{ref_range}</td>
                <td class="col-sig">{_esc(sig[:200])}{"..." if len(sig)>200 else ""}</td>
                <td class="col-ev no-print">
                  <details>
                    <summary class="ev-toggle">Evidence</summary>
                    <ul class="chain-list">{chains}</ul>
                  </details>
                </td>
              </tr>"""
            return f"""
            <div class="table-wrap">
              <table>
                <colgroup>
                  <col class="col-sev"><col class="col-region"><col class="col-metric">
                  <col class="col-val"><col class="col-ref"><col class="col-sig"><col class="col-ev">
                </colgroup>
                <thead>
                  <tr>
                    <th>Severity</th><th>Region</th><th>Metric</th>
                    <th>Value</th><th>Ref. range</th>
                    <th>Clinical significance</th><th class="no-print">Evidence</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""

        def connectivity_section(func_metrics: dict, func_findings: list) -> str:
            if not func_metrics or "note" in func_metrics:
                return '<p class="muted">Functional fMRI not available for this condition.</p>'
            pairs = [
                ("PCC_mPFC",        "PCC / mPFC",            "DMN core",           0.62, 0.14),
                ("PCC_Precuneus",   "PCC / Precuneus",       "DMN posterior",      0.55, 0.16),
                ("ACC_R.INS",       "ACC / Right Insula",    "Salience",           0.42, 0.15),
                ("L.DLPFC_R.DLPFC", "L.DLPFC / R.DLPFC",    "Bilateral executive",0.38, 0.14),
                ("L.DLPFC_L.CAU",   "L.DLPFC / L.Caudate",  "Left frontostriatal",0.25, 0.12),
                ("R.DLPFC_R.CAU",   "R.DLPFC / R.Caudate",  "Right frontostriatal",0.25, 0.12),
                ("R.IFG_R.CAU",     "R.IFG / R.Caudate",    "Inhibition circuit", 0.20, 0.10),
            ]
            n_tp = func_metrics.get("n_timepoints", "N/A")
            t_r  = func_metrics.get("t_r_s", "N/A")
            dmn  = func_metrics.get("dmn_deactivation_failure_index")
            dmn_html = ""
            if dmn is not None:
                dmn_color = "#991b1b" if dmn > 0.5 else "#854d0e" if dmn > 0 else "#166534"
                dmn_html = f"""
              <div class="metric-pill" style="border-color:{dmn_color}">
                <span class="metric-label">DMN Deactivation Failure Index</span>
                <span class="metric-val" style="color:{dmn_color}">{dmn:+.3f}</span>
                <span class="metric-note">{'&gt;0 = hypercoherent' if dmn > 0 else 'normal suppression'}</span>
              </div>"""
            rows = ""
            for key, label, circuit, norm_mean, norm_sd in pairs:
                r_val = func_metrics.get(f"r_{key}")
                z_val = func_metrics.get(f"z_{key}")
                if r_val is None:
                    continue
                                                                                  
                bar_pct  = max(0, min(100, int((r_val + 1) / 2 * 100)))
                norm_lo  = max(0, min(100, int((norm_mean - norm_sd + 1) / 2 * 100)))
                norm_hi  = max(0, min(100, int((norm_mean + norm_sd + 1) / 2 * 100)))
                if z_val is None:
                    z_val = (r_val - norm_mean) / norm_sd
                sev = "normal" if abs(z_val) < 1 else "mild" if abs(z_val) < 1.5 else "moderate" if abs(z_val) < 2.5 else "severe"
                bar_color = _SEV_HTML[sev]["border"]
                rows += f"""
              <tr>
                <td style="white-space:nowrap"><strong>{_esc(label)}</strong><br><small class="muted">{_esc(circuit)}</small></td>
                <td class="num"><strong>{r_val:.3f}</strong></td>
                <td class="num">{z_val:+.2f}</td>
                <td>{_sev_badge_html(sev)}</td>
                <td style="min-width:180px">
                  <div class="rbar-track">
                    <div class="rbar-norm" style="left:{norm_lo}%;width:{norm_hi-norm_lo}%"></div>
                    <div class="rbar-fill" style="width:{bar_pct}%;background:{bar_color}"></div>
                    <div class="rbar-marker" style="left:{bar_pct}%"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#94a3b8;margin-top:2px">
                    <span>−1</span><span>norm: {norm_mean:.2f}±{norm_sd:.2f}</span><span>+1</span>
                  </div>
                </td>
              </tr>"""
            return f"""
            <div class="meta-pills">
              <div class="metric-pill">
                <span class="metric-label">Timepoints</span>
                <span class="metric-val">{n_tp}</span>
              </div>
              <div class="metric-pill">
                <span class="metric-label">TR</span>
                <span class="metric-val">{t_r} s</span>
              </div>
              {dmn_html}
            </div>
            <div class="table-wrap" style="margin-top:16px">
              <table>
                <thead>
                  <tr>
                    <th>Circuit</th><th>r</th><th>z-score</th>
                    <th>Severity</th><th>vs. normative range</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>
            {findings_table(func_findings)}"""

        def literature_cards(clinical_outputs: list) -> str:
            docs = []
            for co in clinical_outputs:
                for f in co.findings[:5]:
                    docs.append(f)
            if not docs:
                return '<p class="muted">No literature retrieved.</p>'
            cards = ""
            for i, f in enumerate(docs[:8], 1):
                title = f.finding_id
                body  = f.evidence_chain[0][:300] if f.evidence_chain else ""
                cards += f"""
              <div class="ref-card">
                <div class="ref-num">{i}</div>
                <div class="ref-body">
                  <p class="ref-title">{_esc(title)}</p>
                  <p class="ref-excerpt">{_esc(body)}{"…" if len(body)==300 else ""}</p>
                </div>
              </div>"""
            return cards

        def atlas_table(all_findings: list) -> str:
            if not all_findings:
                return '<p class="muted">No atlas-grounded regions.</p>'
            seen: dict = {}
            for f in all_findings:
                n = f.region.canonical_name
                if n not in seen:
                    seen[n] = f.region
            rows = ""
            for name, reg in seen.items():
                nets = ", ".join(reg.networks[:3]) if reg.networks else "-"
                rows += f"""
              <tr>
                <td><strong>{_esc(name)}</strong></td>
                <td>{_esc(reg.hemisphere)}</td>
                <td>{_esc(reg.lobe)}</td>
                <td><small>{_esc(nets)}</small></td>
              </tr>"""
            return f"""
            <div class="table-wrap">
              <table>
                <thead><tr><th>Region</th><th>Hemisphere</th><th>Lobe</th><th>Networks</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""

        def key_findings_cards(key_findings: list) -> str:
            if not key_findings:
                return '<div class="finding-card finding-normal"><p>No significant abnormalities detected.</p></div>'
            cards = ""
            for kf in key_findings:
                cards += f'<div class="finding-card"><p>{_esc(kf)}</p></div>'
            return cards

        sec_key = key_findings_cards(key_findings)

        struct_meta = ""
        if struct_metrics:
            icv  = struct_metrics.get("icv_mm3", 0)
            nprs = struct_metrics.get("n_structures_parsed", "N/A")
            icv_str = f"{icv:,.0f} mm3" if icv else "N/A"
            struct_meta = f"""
            <div class="meta-pills">
              <div class="metric-pill">
                <span class="metric-label">Structures parsed</span>
                <span class="metric-val">{nprs}</span>
              </div>
              <div class="metric-pill">
                <span class="metric-label">ICV</span>
                <span class="metric-val">{icv_str}</span>
              </div>
            </div>"""
        sec_struct = struct_meta + findings_table(struct_findings)

        sec_func = connectivity_section(func_metrics, func_findings)

        sec_lit = literature_cards(clinical_outputs)

        sec_atlas = atlas_table(struct_findings + func_findings)

        sec_assess = f"""
            <p class="assessment-text">{_esc(assessment)}</p>"""

        if extra_context:
            sec_assess += f"""
            <details style="margin-top:16px">
              <summary style="cursor:pointer;color:{color};font-weight:600">
                Orchestrator reasoning trace ▸
              </summary>
              <pre class="reasoning-trace">{_esc(extra_context)}</pre>
            </details>"""

        lim_items = "".join(f"<li>{_esc(l.replace('-', ' - '))}</li>" for l in limitations)
        sec_lim = f"<ul class='limitation-list'>{lim_items}</ul>"

        assessment_clean = assessment.replace("-", " - ").replace("\u2013", " - ")
        sec_assess = f'<p class="assessment-text">{_esc(assessment_clean)}</p>'
        if extra_context:
            sec_assess += f"""
            <details style="margin-top:16px">
              <summary class="ev-toggle">Orchestrator reasoning trace</summary>
              <pre class="reasoning-trace">{_esc(extra_context)}</pre>
            </details>"""

        raw_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroAgent Report - {_esc(subject_id)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ font-size: 15px; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: #eef2f7;
      color: #1e293b;
      line-height: 1.6;
    }}

    /* ── Page wrapper ── */
    .page {{
      max-width: 1140px;
      margin: 0 auto;
      background: #eef2f7;
      min-height: 100vh;
    }}

    /* ── Header ── */
    .report-header {{
      background: #0f172a;
      color: #fff;
      border-left: 6px solid {color};
    }}
    .header-inner {{
      padding: 26px 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }}
    .header-wordmark {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .wordmark-top {{
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: {color};
    }}
    .wordmark-title {{
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #f1f5f9;
      line-height: 1.1;
    }}
    .wordmark-sub {{
      font-size: 0.75rem;
      color: #64748b;
      letter-spacing: 0.04em;
      margin-top: 3px;
    }}
    .header-meta {{
      text-align: right;
      font-size: 0.8rem;
      color: #64748b;
      line-height: 2;
    }}
    .header-meta strong {{ color: #cbd5e1; font-weight: 600; }}
    .condition-badge {{
      display: inline-block;
      margin-top: 6px;
      padding: 4px 14px;
      border-radius: 4px;
      background: {color};
      color: #fff;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }}

    /* ── Confidence strip ── */
    .confidence-strip {{
      background: #1a2540;
      padding: 14px 36px;
      display: flex;
      align-items: center;
      gap: 18px;
      border-left: 6px solid {color};
    }}
    .conf-label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #64748b;
      white-space: nowrap;
      font-weight: 700;
      min-width: 160px;
    }}
    .conf-track {{
      flex: 1;
      height: 6px;
      background: #2d3f5e;
      border-radius: 999px;
      overflow: hidden;
    }}
    .conf-fill {{
      height: 100%;
      background: {conf_color};
      border-radius: 999px;
      width: {conf_pct}%;
      print-color-adjust: exact;
      -webkit-print-color-adjust: exact;
    }}
    .conf-pct {{
      font-size: 1.15rem;
      font-weight: 800;
      color: {conf_color};
      white-space: nowrap;
      min-width: 48px;
      text-align: right;
      letter-spacing: -0.02em;
    }}
    .conf-note {{
      font-size: 0.68rem;
      color: #475569;
      white-space: nowrap;
      font-style: italic;
    }}

    /* ── Main content ── */
    .main-content {{
      padding: 24px 32px 48px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    /* ── Cards ── */
    .card {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      overflow: hidden;
      border: 1px solid #dde3ed;
    }}
    .card-header {{
      padding: 12px 22px;
      background: #f6f8fb;
      border-bottom: 1px solid #dde3ed;
      display: flex;
      align-items: center;
    }}
    .card-header h2 {{
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #64748b;
    }}
    .card-header h2::before {{
      content: "";
      display: inline-block;
      width: 3px;
      height: 13px;
      background: {color};
      border-radius: 2px;
      margin-right: 10px;
      vertical-align: middle;
    }}
    .card-body {{ padding: 20px 22px; }}

    /* ── Key findings ── */
    .findings-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 10px;
    }}
    .finding-card {{
      border-left: 4px solid {color};
      background: {light};
      border-radius: 5px;
      padding: 11px 15px;
      font-size: 0.86rem;
      line-height: 1.55;
      color: #1e293b;
    }}
    .finding-normal {{
      border-left-color: #16a34a;
      background: #f0fdf4;
    }}

    /* ── Tables ── */
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.83rem;
      table-layout: fixed;
    }}
    col.col-sev    {{ width: 90px; }}
    col.col-region {{ width: 150px; }}
    col.col-metric {{ width: 110px; }}
    col.col-val    {{ width: 90px; }}
    col.col-ref    {{ width: 120px; }}
    col.col-sig    {{ width: auto; }}
    col.col-ev     {{ width: 90px; }}
    thead tr {{ background: #f6f8fb; }}
    th {{
      text-align: left;
      padding: 9px 11px;
      font-size: 0.68rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #64748b;
      border-bottom: 2px solid #dde3ed;
    }}
    td {{
      padding: 8px 11px;
      border-bottom: 1px solid #f0f4f8;
      vertical-align: top;
      word-break: break-word;
      overflow-wrap: break-word;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tbody tr:nth-child(even) td {{ background: #fafbfd; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .col-sig {{ font-size: 0.81rem; line-height: 1.5; }}
    code {{
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
      background: #eef2f7;
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 0.78em;
      color: #475569;
      word-break: break-all;
    }}

    /* ── Evidence toggle ── */
    .ev-toggle {{
      cursor: pointer;
      color: {color};
      font-size: 0.76em;
      font-weight: 700;
      letter-spacing: 0.04em;
      user-select: none;
    }}
    .chain-list {{
      margin: 7px 0 3px 14px;
      font-size: 0.79em;
      color: #475569;
      line-height: 1.55;
    }}

    /* ── Connectivity bars ── */
    .rbar-track {{
      position: relative;
      height: 9px;
      background: #eef2f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .rbar-norm {{
      position: absolute;
      top: 0; bottom: 0;
      background: rgba(100,116,139,.15);
    }}
    .rbar-fill {{
      position: absolute;
      top: 0; bottom: 0; left: 0;
      border-radius: 999px;
      opacity: 0.8;
      print-color-adjust: exact;
      -webkit-print-color-adjust: exact;
    }}
    .rbar-marker {{
      position: absolute;
      top: -2px; bottom: -2px;
      width: 2px;
      background: #0f172a;
      border-radius: 999px;
      transform: translateX(-50%);
    }}

    /* ── Metric pills ── */
    .meta-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .metric-pill {{
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      background: #f6f8fb;
      border: 1px solid #dde3ed;
      border-radius: 6px;
      padding: 7px 13px;
    }}
    .metric-label {{
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #64748b;
      font-weight: 700;
    }}
    .metric-val {{
      font-size: 0.97rem;
      font-weight: 800;
      color: #0f172a;
      font-variant-numeric: tabular-nums;
    }}
    .metric-note {{
      font-size: 0.7rem;
      color: #94a3b8;
      font-style: italic;
    }}

    /* ── Literature ── */
    .ref-card {{
      display: flex;
      gap: 13px;
      padding: 11px 0;
      border-bottom: 1px solid #f0f4f8;
    }}
    .ref-card:last-child {{ border-bottom: none; }}
    .ref-num {{
      flex-shrink: 0;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: {color};
      color: #fff;
      font-size: 0.7rem;
      font-weight: 800;
      display: flex;
      align-items: center;
      justify-content: center;
      print-color-adjust: exact;
      -webkit-print-color-adjust: exact;
    }}
    .ref-title {{
      font-size: 0.86rem;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 3px;
      line-height: 1.4;
    }}
    .ref-excerpt {{
      font-size: 0.78rem;
      color: #64748b;
      line-height: 1.5;
    }}

    /* ── Assessment ── */
    .assessment-text {{
      font-size: 0.91rem;
      line-height: 1.85;
      color: #1e293b;
      background: #f6f8fb;
      border-left: 4px solid {color};
      padding: 15px 20px;
      border-radius: 0 6px 6px 0;
    }}
    .reasoning-trace {{
      margin-top: 12px;
      background: #f6f8fb;
      border: 1px solid #dde3ed;
      border-radius: 5px;
      padding: 11px 15px;
      font-size: 0.76rem;
      color: #475569;
      white-space: pre-wrap;
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    }}

    /* ── Limitations ── */
    .limitation-list {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }}
    .limitation-list li {{
      font-size: 0.85rem;
      color: #475569;
      padding: 9px 14px 9px 36px;
      background: #fffbf5;
      border-radius: 5px;
      border: 1px solid #fde8c8;
      position: relative;
      line-height: 1.55;
    }}
    .limitation-list li::before {{
      content: "i";
      position: absolute;
      left: 12px;
      top: 9px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #f59e0b;
      color: #fff;
      font-size: 0.65rem;
      font-weight: 800;
      font-style: italic;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      line-height: 16px;
      print-color-adjust: exact;
      -webkit-print-color-adjust: exact;
    }}

    /* ── Footer ── */
    .report-footer {{
      background: #0f172a;
      color: #475569;
      padding: 18px 36px;
      font-size: 0.76rem;
      line-height: 1.6;
      border-left: 6px solid {color};
    }}
    .report-footer strong {{ color: #94a3b8; }}
    .muted {{ color: #94a3b8; font-style: italic; font-size: 0.83rem; }}
    .no-print {{ }}

    /* ── Print / PDF ── */
    @media print {{
      @page {{ size: A4; margin: 12mm 14mm 14mm 14mm; }}
      * {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
      html {{ font-size: 11px; }}
      body {{ background: #fff; }}
      .page {{ max-width: none; background: #fff; }}
      .report-header {{
        background: #fff !important;
        color: #0f172a !important;
        border-left-color: {color} !important;
        border-bottom: 2px solid {color};
        -webkit-print-color-adjust: exact;
      }}
      .wordmark-top {{ color: {color} !important; }}
      .wordmark-title {{ color: #0f172a !important; }}
      .wordmark-sub {{ color: #555 !important; }}
      .header-meta {{ color: #444 !important; }}
      .header-meta strong {{ color: #111 !important; }}
      .condition-badge {{ background: {color} !important; color: #fff !important; }}
      .confidence-strip {{
        background: #f6f8fb !important;
        border-bottom: 1px solid #dde3ed;
        border-left-color: {color} !important;
      }}
      .conf-label, .conf-note {{ color: #555 !important; }}
      .conf-pct {{ color: {conf_color} !important; }}
      .main-content {{ padding: 14px 0 20px; gap: 12px; }}
      .card {{
        box-shadow: none;
        border: 1px solid #c8d0dd;
        break-inside: avoid-page;
        margin-bottom: 4px;
      }}
      .card-header {{ background: #f0f3f8 !important; }}
      .report-footer {{
        background: #f6f8fb !important;
        color: #555 !important;
        border-top: 1px solid #c8d0dd;
        border-left-color: {color} !important;
      }}
      .report-footer strong {{ color: #333 !important; }}
      table {{ font-size: 9.5px; }}
      th {{ padding: 6px 8px; font-size: 7.5px; }}
      td {{ padding: 5px 8px; }}
      .no-print {{ display: none !important; }}
      .meta-pills {{ margin-bottom: 10px; }}
      .metric-pill {{ padding: 5px 10px; }}
      .assessment-text {{ padding: 11px 15px; font-size: 0.88rem; }}
      .finding-card {{ padding: 9px 12px; font-size: 0.84rem; }}
      .limitation-list li {{ padding: 7px 12px 7px 32px; font-size: 0.82rem; }}
    }}
  </style>
</head>
<body>
<div class="page">

  <header class="report-header">
    <div class="header-inner">
      <div class="header-wordmark">
        <div class="wordmark-top">Neuroimaging Analysis</div>
        <div class="wordmark-title">NeuroAgent</div>
        <div class="wordmark-sub">Multi-Agent Diagnostic Report</div>
      </div>
      <div class="header-meta">
        <div><strong>Subject ID</strong>&nbsp; {_esc(subject_id)}</div>
        <div><strong>Generated</strong>&nbsp; {_esc(now)}</div>
        <div><span class="condition-badge">{_esc(meta["label"])}</span></div>
      </div>
    </div>
  </header>

  <div class="confidence-strip">
    <span class="conf-label">Diagnostic Confidence</span>
    <div class="conf-track"><div class="conf-fill"></div></div>
    <span class="conf-pct">{conf_pct}%</span>
    <span class="conf-note">research estimate, not a clinical diagnosis</span>
  </div>

  <main class="main-content">
    {section("Key Findings", f'<div class="findings-grid">{sec_key}</div>', "key-findings")}
    {section("Structural MRI Findings", sec_struct, "structural")}
    {section("Functional Connectivity (Resting-State fMRI)", sec_func, "functional")}
    {section("Clinical Knowledge Integration", sec_lit, "literature")}
    {section("Atlas-Grounded Anatomical Summary", sec_atlas, "atlas")}
    {section("Evidence-Based Assessment", sec_assess, "assessment")}
    {section("Limitations and Caveats", sec_lim, "limitations")}
  </main>

  <footer class="report-footer">
    <strong>Disclaimer:</strong>
    This report was generated by NeuroAgent, a research-grade multi-agent AI framework.
    It is intended to support, not replace, clinical decision-making.
    All findings must be reviewed and confirmed by a qualified clinician before any diagnostic
    or therapeutic action is taken. NeuroAgent outputs do not constitute a medical diagnosis.
  </footer>

</div>
</body>
</html>"""
                                                            
        return raw_html.replace("\u2014", " - ").replace("\u2013", " - ")

    def _tumor_clinical_notes(self):
        return ["", "---", "", "## Tumor-Specific Clinical Notes", "",
                "### WHO 2021 Grade Indicators",
                "- **Grade 2 (IDH-mutant):** T2/FLAIR hyperintense, NO enhancement, slow growth",
                "- **Grade 3 (anaplastic):** Patchy enhancement, higher Cho/Cr",
                "- **Grade 4 (GBM):** Ring enhancement + central necrosis + mass effect + edema",
                "", "### Eloquent Cortex Risk",
                "- Motor cortex (precentral): contralateral hemiplegia if resected",
                "- Broca's area (left IFG): expressive aphasia",
                "- Wernicke's area (left pSTG): receptive aphasia"]

    def _stroke_clinical_notes(self):
        return ["", "---", "", "## Stroke-Specific Clinical Notes", "",
                "### Predicted Deficits by Territory",
                "- **Left MCA superior:** Right hemiplegia + Broca's aphasia",
                "- **Left MCA inferior:** Wernicke's aphasia + right superior visual field defect",
                "- **Right MCA:** Left hemiplegia + hemispatial neglect",
                "- **ACA:** Leg > arm weakness + abulia + personality change",
                "- **PCA:** Homonymous hemianopia + amnesia (hippocampal)"]

    def _adhd_clinical_notes(self, func_metrics):
        lines = ["", "---", "", "## ADHD-Specific Clinical Notes", "",
                 "### Neurobiological Model",
                 "- **Frontostriatal circuit:** PFC → Caudate → GPi → Thalamus → PFC",
                 "- **Key structural biomarkers:** Caudate, putamen, accumbens, cerebellum",
                 "- **Key functional biomarkers:** PCC-mPFC (DMN), right IFG-caudate (inhibition)",
                 "", "### DSM-5 Reminder",
                 "- ≥5 inattentive OR hyperactive-impulsive symptoms in ≥2 settings",
                 "- Onset before age 12; present for ≥6 months; cause functional impairment",
                 "- Neuroimaging findings ALONE cannot diagnose ADHD"]
        if func_metrics:
            r_pcc = func_metrics.get("r_PCC_mPFC")
            r_inh = func_metrics.get("r_R.IFG_R.CAU")
            if r_pcc is not None:
                lines.append(f"\n_Subject PCC-mPFC r = {r_pcc:.3f} (norm = 0.62 ± 0.14)_")
            if r_inh is not None:
                lines.append(f"_Subject R.IFG-R.Caudate r = {r_inh:.3f} (norm = 0.20 ± 0.10)_")
        return lines

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuroAgent Report Generator")
    parser.add_argument("--condition", required=True, choices=["tumor", "stroke", "adhd"])
    parser.add_argument("--subject",   required=True)
    parser.add_argument("--out-dir",   default=".", help="Output directory")
    args = parser.parse_args()

    generator = ReportGeneratorAgent()
    report    = generator.generate(condition=args.condition, subject_id=args.subject)
    md_path, json_path, html_path = report.save(args.out_dir)
    print(f"Report saved:\n  Markdown : {md_path}\n  JSON     : {json_path}\n  HTML     : {html_path}")
