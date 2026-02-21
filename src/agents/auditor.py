"""
Agent 2: Risk / Conflict Detection (Clinical Auditor).
Compares extracted data against medical guidelines (e.g. RAG/Vector DB) to flag risks.
"""

from src.state import ClinicalState, RiskSeverity
from src.tools.medical_db import lookup_guidelines
from src.tools.drug_api import check_interactions


def _entities_get(state: dict, key: str, default: list | None = None) -> list:
    """Get list from extracted_entities (dict or object)."""
    default = default or []
    entities = state.get("extracted_entities")
    if entities is None:
        return default
    if isinstance(entities, dict):
        return list(entities.get(key, default) or [])
    return list(getattr(entities, key, default) or [])


def auditor_node(state: dict) -> dict:
    """
    Use tools (medical_db, drug_api) to audit scribe output.
    Sets clinical_risks, guideline_checks, auditor_notes.
    """
    state = dict(state)
    medications = _entities_get(state, "medications")
    diagnoses = _entities_get(state, "diagnoses")
    scribe_summary = (state.get("scribe_summary") or "").strip()

    guideline_checks = []
    # Build a query from diagnoses + medications for guideline lookup
    query_parts = [p for p in diagnoses + medications if p]
    guideline_query = " ".join(query_parts) if query_parts else scribe_summary[:200]
    if guideline_query:
        results = lookup_guidelines(guideline_query, top_k=5)
        guideline_checks.append({"query": guideline_query[:100], "results": results})

    # Drug interaction check
    interaction_result = check_interactions(medications, diagnoses)
    interactions = interaction_result.get("interactions", [])
    warnings = interaction_result.get("warnings", [])

    clinical_risks = []
    for w in warnings:
        clinical_risks.append({
            "description": w,
            "severity": RiskSeverity.MEDIUM.value,
            "category": "drug_safety",
            "source_guideline": "drug_api",
            "related_entities": medications + diagnoses,
        })
    for i in interactions:
        if isinstance(i, dict):
            desc = i.get("description") or i.get("message") or str(i)
        else:
            desc = str(i)
        clinical_risks.append({
            "description": desc,
            "severity": RiskSeverity.HIGH.value,
            "category": "drug_interaction",
            "source_guideline": "drug_api",
            "related_entities": medications + diagnoses,
        })

    state["clinical_risks"] = clinical_risks
    state["guideline_checks"] = guideline_checks
    state["auditor_notes"] = (
        f"Audited {len(guideline_checks)} guideline check(s); "
        f"checked drug interactions for {len(medications)} medication(s). "
        f"Flagged {len(clinical_risks)} risk(s)."
    )
    return state
