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
