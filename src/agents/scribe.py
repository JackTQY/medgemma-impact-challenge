"""
Agent 1: Clinical Data Extraction (Intake Scribe).
Extracts core clinical entities from raw EHR notes.
"""

from src.state import ClinicalState


def scribe_node(state: ClinicalState) -> ClinicalState:
    """
    Process raw EHR and populate extracted_entities + scribe_summary.
    """
    # TODO: Call MedGemma / LLM to extract entities; integrate with tools if needed
    state = dict(state)
    state.setdefault("extracted_entities", {"diagnoses": [], "medications": [], "lab_results": []})
    state.setdefault("scribe_summary", state.get("raw_ehr", "")[:500] + "...")
    return state
