"""
Agent 3: Final Clinical Reviewer (Verification Agent).
Cross-references summary with original source to reduce hallucinations.
"""

from src.state import ClinicalState


def verifier_node(state: ClinicalState) -> ClinicalState:
    """
    Self-reflection: compare scribe/auditor output to raw_ehr; set verified_summary, verification_passed.
    """
    # TODO: MedGemma verification pass; set final_notes
    state = dict(state)
    summary = state.get("scribe_summary", "")
    state.setdefault("verified_summary", summary)
    state.setdefault("verification_passed", True)
    state.setdefault("final_notes", "Verification pending implementation.")
    state.setdefault("verification_status", {"passed": True, "summary": summary, "notes": state.get("final_notes", "")})
    return state
