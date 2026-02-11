"""
Agent 2: Risk / Conflict Detection (Clinical Auditor).
Compares extracted data against medical guidelines (e.g. RAG/Vector DB) to flag risks.
"""

from src.state import ClinicalState


def auditor_node(state: ClinicalState) -> ClinicalState:
    """
    Use tools (medical_db, drug_api) to audit scribe output; set flagged_risks, guideline_checks.
    """
    # TODO: RAG lookup via tools.medical_db; drug interaction via tools.drug_api
    state = dict(state)
    state.setdefault("clinical_risks", [])
    state.setdefault("guideline_checks", [])
    state.setdefault("auditor_notes", "Audit pending implementation.")
    return state
