"""Tests for the clinical workflow (medical reliability / structure)."""

import pytest

from src.state import ClinicalState
from src.graphs.clinical_workflow import run_workflow


def test_workflow_runs():
    """Workflow runs end-to-end and returns state with expected keys."""
    initial: ClinicalState = {"raw_ehr": "Sample note.", "patient_id": "test-1"}
    result = run_workflow(initial)
    assert "scribe_summary" in result
    assert "auditor_notes" in result
    assert "verified_summary" in result
    assert "verification_passed" in result


def test_scribe_populates_state():
    """Scribe node adds extracted_entities and scribe_summary."""
    from src.agents.scribe import scribe_node

    state: ClinicalState = {"raw_ehr": "Patient has hypertension."}
    out = scribe_node(state)
    assert "scribe_summary" in out
    assert "extracted_entities" in out


def test_auditor_populates_state():
    """Auditor node uses tools and sets clinical_risks, guideline_checks, auditor_notes."""
    from src.agents.auditor import auditor_node

    state = {
        "scribe_summary": "Patient on lisinopril for HTN.",
        "extracted_entities": {"diagnoses": ["HTN"], "medications": ["lisinopril 10mg"]},
    }
    out = auditor_node(state)
    assert "clinical_risks" in out
    assert "guideline_checks" in out
    assert "auditor_notes" in out
    assert isinstance(out["clinical_risks"], list)
    assert isinstance(out["guideline_checks"], list)
    # With mock tools, we expect at least one guideline check and optional risks from drug_api
    assert len(out["guideline_checks"]) >= 1
    assert "Audited" in out["auditor_notes"] and "risk" in out["auditor_notes"].lower()


def test_verifier_populates_state():
    """Verifier node cross-checks summary vs source and sets verification fields."""
    from src.agents.verifier import verifier_node

    state = {
        "raw_ehr": "65yo M, HTN. On lisinopril 10mg. HbA1c 7.2%.",
        "scribe_summary": "Diagnoses: HTN. Medications: lisinopril 10mg. Lab: HbA1c 7.2%.",
    }
    out = verifier_node(state)
    assert "verified_summary" in out
    assert "verification_passed" in out
    assert "final_notes" in out
    assert "verification_status" in out
    assert out["verified_summary"] == state["scribe_summary"]
    assert isinstance(out["verification_passed"], bool)
    assert "Cross-checked" in out["final_notes"] or "key term" in out["final_notes"].lower()


def test_verifier_fails_when_scribe_errors():
    """Verifier sets verification_passed=False when scribe summary is an error."""
    from src.agents.verifier import verifier_node

    state = {
        "raw_ehr": "Patient has HTN.",
        "scribe_summary": "Error in scribe extraction: credentials not found.",
    }
    out = verifier_node(state)
    assert out["verification_passed"] is False
    assert "failed" in out["final_notes"].lower() or "missing" in out["final_notes"].lower()
