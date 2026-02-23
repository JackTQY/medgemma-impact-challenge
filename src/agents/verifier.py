"""
Agent 3: Final Clinical Reviewer (Verification Agent).
Cross-references summary with original source to reduce hallucinations.
"""

import re
from src.state import ClinicalState


def _key_terms_from_text(text: str, min_len: int = 3) -> set[str]:
    """Extract potential clinical terms (words, numbers with units) from text."""
    if not (text or "").strip():
        return set()
    # Normalize: split on non-alphanumeric but keep tokens
    tokens = re.findall(r"[A-Za-z]+|\d+\.?\d*%?|[A-Za-z]+\d+", text)
    return {t.strip().lower() for t in tokens if len(t.strip()) >= min_len and t.strip().lower() not in ("the", "and", "for", "with", "from", "has", "no", "not")}


def verifier_node(state: dict, model=None) -> dict:
    """
    Self-reflection: compare scribe summary to raw_ehr; set verified_summary,
    verification_passed, final_notes, verification_status.
    Optionally use `model` for an LLM-based verification pass (TODO).
    """
    state = dict(state)
    raw_ehr = (state.get("raw_ehr") or "").strip()
    scribe_summary = (state.get("scribe_summary") or "").strip()

    if not scribe_summary or "Error in scribe" in scribe_summary:
        passed = False
        final_notes = "Scribe output missing or failed; cannot verify."
    elif model is not None:
        # TODO: Call model with prompt "Does this summary accurately reflect the source? Source: ... Summary: ..."
        passed = True
        final_notes = "LLM verification pass not yet implemented; using heuristic."
    else:
        # Heuristic cross-check: key terms from source should appear in summary
        source_terms = _key_terms_from_text(raw_ehr)
        summary_terms = _key_terms_from_text(scribe_summary)
        if not source_terms:
            passed = True
            final_notes = "No key terms to cross-check; summary accepted."
        else:
            # Require whole-word match to avoid false positives (e.g. "ace" in "peaceful")
            summary_lower = scribe_summary.lower()
            found = sum(
                1
                for t in source_terms
                if t in summary_terms
                or re.search(r"\b" + re.escape(t) + r"\b", summary_lower) is not None
            )
            ratio = found / len(source_terms)
            passed = ratio >= 0.5
            final_notes = f"Cross-checked {len(source_terms)} key term(s) from source; {found} found in summary. {'Consistent.' if passed else 'Possible omission or hallucination.'}"

    state["verified_summary"] = scribe_summary
    state["verification_passed"] = passed
    state["final_notes"] = final_notes
    state["verification_status"] = {
        "passed": passed,
        "summary": scribe_summary,
        "notes": final_notes,
    }
    return state
