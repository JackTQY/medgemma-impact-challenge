"""Tests for src.state logic. Run: pytest tests/test_state.py -v"""

import pytest

from src.state import ClinicalState


def test_from_dict_and_round_trip():
    """ClinicalState.from_dict() loads data; to_dict() returns a dict with expected keys."""
    sample_data = {"raw_ehr": "Patient has high blood pressure.", "patient_id": "123"}
    state = ClinicalState.from_dict(sample_data)

    # Verify raw input
    assert state.patient_id == "123"
    assert "high blood pressure" in state.raw_ehr

    # Verify defaults from nested models
    assert state.extracted_entities.diagnoses == []
    assert state.extracted_entities.medications == []
    assert state.clinical_risks == []
    assert state.verification_passed is False

    # Round-trip to dict (e.g. for LangGraph)
    d = state.to_dict()
    assert "patient_id" in d
    assert "extracted_entities" in d
    assert "clinical_risks" in d
    assert "verification_status" in d
    assert d["patient_id"] == "123"


def test_from_dict_with_nested_entities():
    """from_dict accepts nested dicts and validates them into Pydantic models."""
    sample_data = {
        "raw_ehr": "Hypertension, on lisinopril.",
        "patient_id": "456",
        "extracted_entities": {
            "diagnoses": ["Hypertension"],
            "medications": ["lisinopril"],
        },
    }
    state = ClinicalState.from_dict(sample_data)
    assert state.patient_id == "456"
    assert state.extracted_entities.diagnoses == ["Hypertension"]
    assert state.extracted_entities.medications == ["lisinopril"]
