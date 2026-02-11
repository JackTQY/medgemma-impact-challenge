"""
The "Nodes and Edges" definition for the Clinical Council workflow.
Orchestrates: Scribe → Auditor → Verifier (LangGraph when available).
"""

from src.state import ClinicalState
from src.agents import scribe_node, auditor_node, verifier_node


def run_workflow(initial_state: ClinicalState) -> ClinicalState:
    """
    Run the linear workflow: scribe → auditor → verifier.
    TODO: Replace with LangGraph StateGraph when langgraph is in use:
      - add_node("scribe", scribe_node), add_node("auditor", auditor_node), add_node("verifier", verifier_node)
      - add_edge("scribe", "auditor"), add_edge("auditor", "verifier")
      - compile() and invoke(initial_state)
    """
    state = dict(initial_state)
    state = scribe_node(state)
    state = auditor_node(state)
    state = verifier_node(state)
    return state
