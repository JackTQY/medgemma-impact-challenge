"""
Agent 1: Clinical Data Extraction (Intake Scribe).
Extracts core clinical entities from raw EHR notes.
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ClinicalState, ExtractedMedicalEntities

# The Scribe "Persona"
SCRIBE_SYSTEM_PROMPT = """
You are a Senior Clinical Scribe. Your task is to extract structured medical information 
from raw physician notes. 
Focus on accuracy. If a piece of information is not present, return an empty list.
Extract: Diagnoses, Medications, Lab Results, Procedures, Allergies, and Vitals.
Format your response as a valid JSON object with keys: Diagnoses, Medications, Lab Results, Procedures, Allergies, Vitals (each an array of strings or objects for labs/vitals).
"""


def _parse_scribe_json(text: str) -> dict | None:
    """Try to extract a JSON object from scribe summary (e.g. inside ```json ... ```)."""
    if not (text or "").strip():
        return None
    # Unwrap markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = match.group(1).strip() if match else text.strip()
    # Try to find first { ... } if there's extra prose
    if "{" in raw:
        start = raw.index("{")
        depth = 0
        for i, c in enumerate(raw[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    raw = raw[start : i + 1]
                    break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _scribe_json_to_entities(data: dict) -> dict:
    """Map LLM JSON keys (any casing) to ExtractedMedicalEntities-style dict."""
    key_aliases = {
        "diagnoses": ["diagnoses", "Diagnoses"],
        "medications": ["medications", "Medications"],
        "lab_results": ["lab_results", "Lab Results", "LabResults"],
        "procedures": ["procedures", "Procedures"],
        "allergies": ["allergies", "Allergies"],
        "vitals": ["vitals", "Vitals"],
    }
    out = {
        "diagnoses": [],
        "medications": [],
        "lab_results": [],
        "procedures": [],
        "allergies": [],
        "vitals": [],
        "notes": {},
    }
    for our_key, aliases in key_aliases.items():
        for alias in aliases:
            val = data.get(alias)
            if val is None:
                continue
            if isinstance(val, list):
                if our_key in ("lab_results", "vitals"):
                    out[our_key] = [x if isinstance(x, dict) else {"value": str(x)} for x in val]
                else:
                    out[our_key] = [str(x) for x in val]
            break
    return out


def scribe_node(state: dict, model=None):
    """
    The node function for LangGraph.
    Takes current state (dict), optionally uses `model` to extract entities, and returns updated state.
    When `model` is None (e.g. in tests or stub runs), populates state with placeholders.
    """
    state = dict(state)
    raw_ehr = state.get("raw_ehr", "")

    if model is not None:
        messages = [
            SystemMessage(content=SCRIBE_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract entities from this EHR note:\n\n{raw_ehr}"),
        ]
        call_log = state.setdefault("__llm_call_log", [])
        call_log.append({
            "call": len(call_log) + 1,
            "node": "scribe_node",
            "method": "invoke",
            "purpose": "entity extraction from raw EHR (SystemMessage + HumanMessage → LLM)",
        })
        try:
            response = model.invoke(messages)
            from src.models import _LAST_LLM_CALL_TOKENS
            if _LAST_LLM_CALL_TOKENS and call_log:
                call_log[-1].update(_LAST_LLM_CALL_TOKENS)
            content = getattr(response, "content", str(response)) or "Successfully extracted structured clinical data."
            state["scribe_summary"] = content
            parsed = _parse_scribe_json(content)
            if parsed:
                entity_dict = _scribe_json_to_entities(parsed)
                state["extracted_entities"] = ExtractedMedicalEntities.model_validate(entity_dict)
        except Exception as e:
            state["scribe_summary"] = f"Error in scribe extraction: {str(e)}"
    else:
        state.setdefault("scribe_summary", (raw_ehr[:500] + "...") if raw_ehr else "No EHR provided.")
        state.setdefault("extracted_entities", {"diagnoses": [], "medications": [], "lab_results": [], "procedures": [], "allergies": [], "vitals": [], "notes": {}})

    return state