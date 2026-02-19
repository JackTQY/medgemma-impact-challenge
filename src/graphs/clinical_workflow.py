"""
The "Nodes and Edges" definition for the Clinical Council workflow.
Orchestrates: Scribe → Auditor → Verifier (LangGraph when available).
"""

from src.state import ClinicalState
from src.agents import scribe_node, auditor_node, verifier_node


def run_workflow(initial_state: ClinicalState, model=None) -> ClinicalState:
    """
    Run the linear workflow: scribe → auditor → verifier.
    Pass a LangChain chat model (e.g. from src.models.get_medgemma_model()) to use real MedGemma/Gemma.
    """
    state = dict(initial_state)
    state = scribe_node(state, model=model)
    state = auditor_node(state)
    state = verifier_node(state)
    return state
