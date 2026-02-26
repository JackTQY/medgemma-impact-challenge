"""
Entry point to run the clinical workflow.
Usage (from project root): python -m src.main
With real MedGemma: set USE_MEDGEMMA=1 and USE_MEDGEMMA_BACKEND=vertex, huggingface, local, or local_gguf in .env
"""

import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so it works regardless of cwd (e.g. python -m src.main)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# Suppress Google Auth "quota project" warning when using ADC (e.g. gcloud auth application-default login)
warnings.filterwarnings("ignore", message=".*without a quota project.*", category=UserWarning)

from src.graphs.clinical_workflow import run_workflow
from src.models import get_medgemma_model


def _one_line(s: str, max_len: int = 200) -> str:
    """Collapse newlines and extra spaces so JSON-like text fits on one line for display."""
    if not s:
        return ""
    t = re.sub(r"\s+", " ", s.strip())
    return (t[:max_len] + "...") if len(t) > max_len else t


def _format_entities(entities: dict | None) -> str:
    """Turn extracted_entities dict into a few readable lines."""
    if not entities:
        return "  (none extracted yet)"
    lines = []
    for key in ("diagnoses", "medications", "allergies", "lab_results", "procedures", "vitals"):
        val = entities.get(key)
        if val:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines) if lines else "  (empty)"


def print_langchain_usage(result: dict) -> None:
    """Print how many LangChain LLM calls were made and how they were orchestrated."""
    log = result.get("__llm_call_log") or []
    if not log:
        return
    n = len(log)
    sep = "-" * 60
    print(sep)
    print("  LANGCHAIN LLM USAGE (agentic workflow)")
    print(sep)
    print(f"  Total LLM API calls this run: {n}")
    print()
    print("  How it was orchestrated:")
    print("    • State (ClinicalState dict) is passed through all nodes; each node returns updated state.")
    print("    • Scribe: builds [SystemMessage, HumanMessage], calls model.invoke(messages), parses JSON into state.")
    print("    • Auditor: reads state (scribe_summary, extracted_entities), uses tools (stub), returns risks + notes.")
    print("    • Verifier: cross-checks summary vs raw_ehr, sets verification_passed and final_notes.")
    print("    Flow:  [Scribe] --invoke--> [Auditor] --> [Verifier]  (single LLM call at Scribe)")
    print()
    print("  Call log:")
    for entry in log:
        parts = [f"    #{entry.get('call', '?')} {entry.get('node', '?')}.{entry.get('method', 'invoke')} — {entry.get('purpose', '')}"]

        if entry.get("n_prompt_tokens") is not None and entry.get("n_output_tokens") is not None:
            parts.append(f" ({entry['n_prompt_tokens']} → {entry['n_output_tokens']} tokens)")
        print("".join(parts))
    print(sep)
    print()


def print_workflow_result(result: dict) -> None:
    """Print workflow result in a human-readable, stage-by-stage format."""
    raw = result.get("raw_ehr", "") or "(no input)"
    pid = result.get("patient_id", "?")
    scribe = result.get("scribe_summary", "")
    entities = result.get("extracted_entities") or {}
    if hasattr(entities, "model_dump"):
        entities = entities.model_dump()
    elif not isinstance(entities, dict):
        entities = {}
    risks = result.get("clinical_risks", [])
    auditor_notes = result.get("auditor_notes", "")
    verified = result.get("verified_summary", "")
    passed = result.get("verification_passed", False)
    final_notes = result.get("final_notes", "")

    sep = "-" * 60
    print()
    print("  CLINICAL COUNCIL WORKFLOW - Result")
    print(sep)
    print("  1. RAW INPUT")
    print(f"     Patient ID: {pid}")
    print(f"     Raw EHR:    {raw[:200]}{'...' if len(raw) > 200 else ''}")
    print()
    print("  2. SCRIBE (Intake) - value: structured extraction from free text")
    print(f"     Summary:    {_one_line(scribe)}")
    print("     Extracted entities:")
    print(_format_entities(entities))
    print()
    print("  3. AUDITOR - value: risk/conflict detection vs guidelines")
    print(f"     Notes:      {_one_line(auditor_notes)}")
    if risks:
        for i, r in enumerate(risks, 1):
            desc = r.get("description", r) if isinstance(r, dict) else getattr(r, "description", r)
            print(f"     Risk {i}:    {desc}")
    else:
        print("     Flagged risks: (none)")
    print()
    print("  4. VERIFIER - value: final check vs source, reduce hallucinations")
    print(f"     Verified summary: {_one_line(verified)}")
    print(f"     Passed:     {passed}")
    print(f"     Final notes: {_one_line(final_notes)}")
    print()
    print(sep)
    print("  VALUE OF AGENTIC WORKFLOW")
    print("  - Scribe turns unstructured notes into structured entities.")
    print("  - Auditor compares against guidelines and flags risks.")
    print("  - Verifier cross-checks the summary to the source for safety.")
    print()


def main() -> None:
    """Run the Clinical Council workflow (Scribe → Auditor → Verifier)."""
    initial_state = {
        "raw_ehr": "65yo M, HTN. On lisinopril 10mg. Last HbA1c 7.2%. No known allergies.",
        "patient_id": "sample-001",
        "__llm_call_log": [],
    }
    model = None
    backend = os.getenv("USE_MEDGEMMA_BACKEND", "").strip().lower()
    if os.getenv("USE_MEDGEMMA", "").strip().lower() in ("1", "true", "yes") or backend:
        try:
            model = get_medgemma_model(backend or None)
            if model:
                print("Using real model (backend=%s). Running workflow...\n" % (backend or "env"))
        except Exception as e:
            print("Could not load MedGemma model: %s. Running with stub.\n" % e)
    result = run_workflow(initial_state, model=model)
    print_workflow_result(result)
    print_langchain_usage(result)


if __name__ == "__main__":
    main()
