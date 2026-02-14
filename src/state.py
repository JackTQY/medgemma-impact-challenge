"""
State schema for the clinical workflow.
Defines what agents "remember" and pass between nodes (Scribe → Auditor → Verifier).
Uses Pydantic for validation and a clear, auditable structure.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ----- Nested models: extracted entities and risks -----


class ExtractedMedicalEntities(BaseModel):
    """Structured medical entities extracted from raw EHR by the Scribe agent."""

    diagnoses: list[str] = Field(default_factory=list, description="Active/resolved diagnoses")
    medications: list[str] = Field(default_factory=list, description="Current medications (names or IDs)")
    lab_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Key lab findings, e.g. [{'name': 'HbA1c', 'value': '7.2', 'unit': '%', 'date': '...'}]",
    )
    procedures: list[str] = Field(default_factory=list, description="Procedures or interventions")
    allergies: list[str] = Field(default_factory=list, description="Known allergies")
    vitals: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Vital signs, e.g. [{'name': 'BP', 'value': '120/80', 'unit': 'mmHg'}]",
    )
    notes: dict[str, Any] = Field(default_factory=dict, description="Other free-form extracted fields")


class RiskSeverity(str, Enum):
    """Severity of an identified clinical risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClinicalRisk(BaseModel):
    """A single identified clinical risk or conflict from the Auditor."""

    description: str = Field(..., description="Human-readable description of the risk")
    severity: RiskSeverity = Field(default=RiskSeverity.MEDIUM)
    category: str = Field(
        default="general",
        description="E.g. 'drug_interaction', 'diagnostic_conflict', 'guideline_deviation'",
    )
    source_guideline: str | None = Field(default=None, description="Guideline or source that flagged this")
    related_entities: list[str] = Field(default_factory=list, description="Relevant diagnoses, meds, or labs")


class VerificationStatus(BaseModel):
    """Final verification outcome from the Verifier agent."""

    passed: bool = Field(..., description="Whether the audit passed verification")
    summary: str = Field(default="", description="Final verified clinical summary")
    notes: str = Field(default="", description="Verifier comments or caveats")
    verified_at: str | None = Field(default=None, description="ISO timestamp when verification ran (optional)")


# ----- Main workflow state -----


class ClinicalState(BaseModel):
    """
    Shared state across the Clinical Council workflow.
    Tracks raw input, extracted entities, identified risks, and final verification status.
    """

    model_config = {"extra": "allow"}  # Allow extra keys for LangGraph/dict compatibility

    # --- Raw input ---
    raw_ehr: str = Field(default="", description="Raw EHR text or note content")
    patient_id: str = Field(default="", description="Patient or encounter identifier")

    # --- Scribe output: extracted medical entities ---
    extracted_entities: ExtractedMedicalEntities = Field(
        default_factory=ExtractedMedicalEntities,
        description="Structured entities extracted from raw_ehr",
    )
    scribe_summary: str = Field(default="", description="Scribe's free-text summary")

    # --- Auditor output: identified clinical risks ---
    clinical_risks: list[ClinicalRisk] = Field(
        default_factory=list,
        description="List of identified clinical risks or conflicts",
    )
    guideline_checks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of guideline lookups performed",
    )
    auditor_notes: str = Field(default="", description="Auditor's commentary")

    # --- Verifier output: final verification status ---
    verification_status: VerificationStatus = Field(
        default_factory=lambda: VerificationStatus(passed=False, summary=""),
        description="Final verification result and summary",
    )
    # Legacy/alias fields for backward compatibility with existing agents
    verified_summary: str = Field(default="", description="Alias for verification_status.summary")
    verification_passed: bool = Field(default=False, description="Alias for verification_status.passed")
    final_notes: str = Field(default="", description="Alias for verification_status.notes")

    def sync_verification_aliases(self) -> None:
        """Keep verified_summary, verification_passed, final_notes in sync with verification_status."""
        self.verified_summary = self.verification_status.summary
        self.verification_passed = self.verification_status.passed
        self.final_notes = self.verification_status.notes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalState":
        """Build ClinicalState from a dict (e.g. workflow initial state or node output)."""
        # Normalize nested dicts into Pydantic models when needed
        if "extracted_entities" in data and isinstance(data["extracted_entities"], dict):
            data = {**data, "extracted_entities": ExtractedMedicalEntities.model_validate(data["extracted_entities"])}
        if "clinical_risks" in data:
            data = {
                **data,
                "clinical_risks": [ClinicalRisk.model_validate(r) if isinstance(r, dict) else r for r in data["clinical_risks"]],
            }
        if "verification_status" in data and isinstance(data["verification_status"], dict):
            data = {**data, "verification_status": VerificationStatus.model_validate(data["verification_status"])}
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Export to a plain dict for LangGraph or serialization (e.g. JSON)."""
        d = self.model_dump(mode="python")
        # Ensure enums are strings in dict
        for i, r in enumerate(d.get("clinical_risks", [])):
            if isinstance(r, dict) and "severity" in r and hasattr(r["severity"], "value"):
                d["clinical_risks"][i] = {**r, "severity": r["severity"].value}
        return d
