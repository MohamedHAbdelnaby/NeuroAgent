from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Literal

Condition = Literal["tumor", "stroke", "adhd"]
Severity  = Literal["normal", "mild", "moderate", "severe", "unknown"]
FindingType = Literal[
    "volume_asymmetry",
    "atrophy",
    "mass_effect",
    "lesion",
    "connectivity_deficit",
    "connectivity_excess",
    "network_abnormality",
    "structural_anomaly",
    "clinical_context",
]

@dataclass
class AtlasRegion:
    canonical_name: str
    hemisphere:     str                                                      
    lobe:           str                                                             
                                                                                       
    atlas_label:    str                           
    mni_centroid:   tuple                      
    functional_role: str = ""
    tumor_relevance: str = ""                                                    
    stroke_territory: str = ""                                              
    adhd_relevance:  str = ""                                        
    networks:        list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Measurement:
    metric:          str                                                                            
    value:           float
    units:           str                                                 
    reference_min:   Optional[float] = None                             
    reference_max:   Optional[float] = None                             
    normative_source: str = ""                                                         

    @property
    def is_abnormal(self) -> bool:
        if self.reference_min is not None and self.value < self.reference_min:
            return True
        if self.reference_max is not None and self.value > self.reference_max:
            return True
        return False

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Finding:
    finding_id:          str
    finding_type:        FindingType
    severity:            Severity
    region:              AtlasRegion
    measurement:         Measurement
    clinical_significance: str                                                           
    evidence_chain:      list[str]                                                        

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary_line(self) -> str:
        ref = ""
        if self.measurement.reference_min is not None:
            ref = f" (ref {self.measurement.reference_min}-{self.measurement.reference_max} {self.measurement.units})"
        return (
            f"[{self.severity.upper()}] {self.region.canonical_name}: "
            f"{self.measurement.metric}={self.measurement.value:.3g} "
            f"{self.measurement.units}{ref} - {self.clinical_significance}"
        )

@dataclass
class AgentOutput:
    agent_name:    str                                                                        
                                                                        
    condition:     Condition
    subject_id:    str
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status:        str = "success"                                     
    findings:      list[Finding] = field(default_factory=list)
    global_metrics: dict = field(default_factory=dict)                               
    warnings:      list[str] = field(default_factory=list)
    errors:        list[str] = field(default_factory=list)
    metadata:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def finding_summary(self) -> str:
        if not self.findings:
            return "No findings."
        lines = [f.summary_line() for f in self.findings]
        return "\n".join(lines)

    @classmethod
    def failed(cls, agent_name: str, condition: Condition,
               subject_id: str, error: str) -> "AgentOutput":
        return cls(
            agent_name=agent_name,
            condition=condition,
            subject_id=subject_id,
            status="failed",
            errors=[error],
        )

def make_finding(
    idx: int,
    finding_type: FindingType,
    severity: Severity,
    region: AtlasRegion,
    measurement: Measurement,
    clinical_significance: str,
    evidence_chain: list[str],
) -> Finding:
    finding_id = f"F{idx:03d}_{region.canonical_name.replace(' ', '_')}"
    return Finding(
        finding_id=finding_id,
        finding_type=finding_type,
        severity=severity,
        region=region,
        measurement=measurement,
        clinical_significance=clinical_significance,
        evidence_chain=evidence_chain,
    )

@dataclass
class OrchestratorOutput:
    subject_id:         str
    condition:          Condition
    timestamp:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_outputs:      list[AgentOutput] = field(default_factory=list)
    diagnostic_summary: str = ""
    reasoning_trace:    list[str] = field(default_factory=list)
    confidence:         float = 0.0        
    report_markdown:    str = ""
    report_html:        str = ""
                                                                      
    predicted_label:    Optional[int] = None                                      
    predicted_class:    str = ""                                            
    extraction_reason:  str = ""                                        

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
