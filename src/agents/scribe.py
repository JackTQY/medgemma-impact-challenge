"""
Agent 1: Clinical Data Extraction (Intake Scribe).
Extracts core clinical entities from raw EHR notes.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ClinicalState, ExtractedMedicalEntities

# The Scribe "Persona"
SCRIBE_SYSTEM_PROMPT = """
You are a Senior Clinical Scribe. Your task is to extract structured medical information 
from raw physician notes. 
Focus on accuracy. If a piece of information is not present, return an empty list.
Extract: Diagnoses, Medications, Lab Results, Procedures, Allergies, and Vitals.
Format your response as a valid JSON object matching the requested schema.
"""

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
        try:
            response = model.invoke(messages)
            state["scribe_summary"] = getattr(response, "content", str(response)) or "Successfully extracted structured clinical data."
            # TODO: parse response into ExtractedMedicalEntities and set state["extracted_entities"]
        except Exception as e:
            state["scribe_summary"] = f"Error in scribe extraction: {str(e)}"
    else:
        state.setdefault("scribe_summary", (raw_ehr[:500] + "...") if raw_ehr else "No EHR provided.")
        state.setdefault("extracted_entities", {"diagnoses": [], "medications": [], "lab_results": [], "procedures": [], "allergies": [], "vitals": [], "notes": {}})

    return state