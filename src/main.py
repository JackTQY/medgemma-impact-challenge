"""
Entry point to run the clinical workflow.
Usage (from project root): python -m src.main
"""

from src.graphs.clinical_workflow import run_workflow


def main() -> None:
    """Run the Clinical Council workflow (Scribe → Auditor → Verifier)."""
    # Example: run with sample input; replace with CLI/env config as needed
    initial_state = {
        "raw_ehr": "",
        "patient_id": "sample",
    }
    result = run_workflow(initial_state)
    print("Workflow result:", result)


if __name__ == "__main__":
    main()
