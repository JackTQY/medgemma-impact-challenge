"""
Clinical Council workflow as a LangGraph StateGraph.
Orchestrates: Scribe → Auditor → Verifier with explicit state, conditional edges (self-correction loop), and optional checkpointer.

Compared with a simple linear pipeline (scribe → auditor → verifier in plain Python), this adds:

- **Explicit state machine:** StateGraph(ClinicalCouncilState). State is a TypedDict; each node returns a partial update that is merged in.
- **Cyclical flow:** After Verifier, a conditional edge either goes to END or back to Scribe (self-correction loop, once).
- **Checkpointing:** MemorySaver stores state after every node so execution can be resumed or inspected (time-travel).
- **Optional human-in-the-loop:** Compile with interrupt_before=["verifier"] to pause before verification and resume later.

Sophisticated agentic behaviour: (1) Decision gate — Verifier outcome chooses next step (end vs retry).
(2) Self-correction — if verification fails, the graph routes back to Scribe once (retry_count + retry hint in prompt).
(3) Durable state — state is first-class and checkpointed. (4) Optional HIL — execution can stop before Verifier for human approval.

ASCII diagram (see also assets/langgraph-workflow.png):

                    +----------------------------------------------------------+
                    |              LangGraph StateGraph                        |
                    |  State: ClinicalCouncilState (checkpointed each step)    |
                    +----------------------------------------------------------+
                                                      |
                                                      v
    +----------------+     +----------------+     +----------------+
    |     SCRIBE      |---->|    AUDITOR     |---->|    VERIFIER    |
    | (LLM extract)   |     | (tools/risk)   |     | (cross-check)  |
    +--------^-------+     +----------------+     +--------+-------+
             |                                            |
             |         conditional edge                   |
             |    (verification_passed? retry_count<=1?)   |
             |                                            |
             |  +-----------------------------------------+
             |  |                                         |
             |  |  "scribe" (retry)                 "end" v
             +--+                                +--------+
      (once; then "end")                         |  END   |
                                                 +--------+
"""

# Human-readable diagram and description for docs/CLI (single source for text form).
WORKFLOW_DIAGRAM = r"""
    +----------------+     +----------------+     +----------------+
    |     SCRIBE      |---->|    AUDITOR     |---->|    VERIFIER    |
    | (LLM extract)   |     | (tools/risk)   |     | (cross-check)  |
    +--------^-------+     +----------------+     +--------+-------+
             |                                            |
             |  conditional: passed? end : retry? scribe   |
             +---------------------------------------------+
"""

WORKFLOW_DESCRIPTION = (
    "LangGraph StateGraph: explicit state (ClinicalCouncilState), checkpointed (MemorySaver). "
    "Nodes: Scribe -> Auditor -> Verifier. Conditional edge: if verification fails, loop back to Scribe once (self-correction), then END. "
    "Optional interrupt_before_verifier for human-in-the-loop."
)

from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents import auditor_node, scribe_node, verifier_node


# ----- Explicit state schema (LangGraph: persistent, checkpointed) -----
class ClinicalCouncilState(TypedDict, total=False):
    """Global state shared by all nodes; saved after every step for recovery/time-travel."""

    raw_ehr: str
    patient_id: str
    scribe_summary: str
    extracted_entities: Any
    clinical_risks: list
    guideline_checks: list
    auditor_notes: str
    verified_summary: str
    verification_passed: bool
    final_notes: str
    verification_status: dict
    __llm_call_log: list
    retry_count: int


# Max Scribe retries when Verifier fails (self-correction loop); allows up to MAX+1 Scribe runs (initial + retries)
MAX_VERIFICATION_RETRIES = 2


def _graph_node_scribe(state: ClinicalCouncilState, *, model: Any) -> dict:
    """LangGraph node: run Scribe; return state update."""
    out = scribe_node(dict(state), model=model)
    return {k: out[k] for k in ("scribe_summary", "extracted_entities", "__llm_call_log") if k in out}


def _graph_node_auditor(state: ClinicalCouncilState) -> dict:
    """LangGraph node: run Auditor; return state update."""
    out = auditor_node(dict(state))
    return {k: out[k] for k in ("clinical_risks", "guideline_checks", "auditor_notes") if k in out}


def _graph_node_verifier(state: ClinicalCouncilState) -> dict:
    """LangGraph node: run Verifier; return state update (includes retry_count for conditional edge)."""
    out = verifier_node(dict(state))
    retry = state.get("retry_count", 0) + 1
    return {
        **{k: out[k] for k in ("verified_summary", "verification_passed", "final_notes", "verification_status") if k in out},
        "retry_count": retry,
    }


def _after_verifier_route(state: ClinicalCouncilState) -> Literal["scribe", "end"]:
    """Conditional edge: loop back to Scribe for up to MAX_VERIFICATION_RETRIES retries if verification failed, else END."""
    passed = state.get("verification_passed", False)
    retry_count = state.get("retry_count", 0)
    if passed:
        return "end"
    if retry_count <= MAX_VERIFICATION_RETRIES:
        return "scribe"
    return "end"


def build_graph(model: Any):
    """
    Build the Clinical Council StateGraph.
    Nodes: scribe → auditor → verifier; conditional edge from verifier → scribe (retry) or END.
    """
    workflow = StateGraph(ClinicalCouncilState)

    workflow.add_node("scribe", lambda s: _graph_node_scribe(s, model=model))
    workflow.add_node("auditor", _graph_node_auditor)
    workflow.add_node("verifier", _graph_node_verifier)

    workflow.add_edge("scribe", "auditor")
    workflow.add_edge("auditor", "verifier")
    workflow.add_conditional_edges("verifier", _after_verifier_route, {"scribe": "scribe", "end": END})

    workflow.set_entry_point("scribe")
    return workflow


def run_workflow(
    initial_state: dict,
    model: Any = None,
    *,
    use_checkpointer: bool = True,
    interrupt_before_verifier: bool = False,
) -> dict:
    """
    Run the Clinical Council workflow via LangGraph.
    - State is explicit (ClinicalCouncilState); persisted after each node when use_checkpointer=True.
    - Conditional edge: if Verifier fails, graph loops back to Scribe once (self-correction), then exits.
    - Optional interrupt_before_verifier: pause for human-in-the-loop before verification (resume with .invoke).

    model: LangChain-compatible chat model (see get_medgemma_model in src.models). Actual types:
      (a) local 4B: _LocalChatWrapper (our wrapper in models.py)
      (b) Hugging Face: langchain_huggingface.ChatHuggingFace
      (c) Vertex AI: langchain_google_genai.ChatGoogleGenerativeAI
      (d) local GGUF: langchain_community.chat_models.ChatLlamaCpp
    All expose .invoke(messages: list) -> object with .content (str); contract from LangChain Runnable/LCEL.
    """
    graph = build_graph(model)
    checkpointer = MemorySaver() if use_checkpointer else None
    compile_kw: dict = {"checkpointer": checkpointer} if checkpointer else {}
    if interrupt_before_verifier:
        compile_kw["interrupt_before"] = ["verifier"]

    compiled = graph.compile(**compile_kw)

    config = {"configurable": {"thread_id": "clinical-council-1"}} if checkpointer else {}
    result = compiled.invoke(
        {**initial_state, "retry_count": initial_state.get("retry_count", 0)},
        config=config,
    )
    return result
