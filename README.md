# MedGemma Impact Challenge: Multi-Agent Clinical Workflow

**Entry for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)** (Agentic Workflow Prize). A three-agent pipeline (Scribe → Auditor → Verifier) that extracts structured clinical data from EHR text, audits it against guidelines and drug interactions, and verifies the summary against the source to reduce hallucinations.

**Developed by [JackTQY](https://github.com/JackTQY) · [Actualizer Systems LLC](https://github.com/actualizer-systems)** · [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## For judges: run in under a minute

No API keys or cloud setup required. From a clean clone:

```bash
git clone https://github.com/JackTQY/medgemma-impact-challenge.git
cd medgemma-impact-challenge
pip install -r requirements.txt
python -m src.main
pytest
```

You should see the full pipeline output (raw EHR → Scribe → Auditor → Verifier) and **8 passing tests**. This is stub mode: same code path, no external APIs.

---

## Problem and novelty

- **Problem:** After-hours documentation (“pajama time”) and fragmented EHRs lead to **missed contradictions**—e.g. new labs vs. historical diagnoses, or medication risks that slip through.
- **Novel task:** Clinical conflict detection (Auditor flags risks and guideline deviations from extracted entities).
- **Agentic workflow:** Multi-step pipeline with **tool use** (guideline lookup, drug-interaction check) and **self-reflection** (Verifier cross-checks summary vs. source).

---

## Architecture: “Clinical Council”

| Agent | Role | Implementation |
|-------|------|----------------|
| **Scribe** | Extract diagnoses, medications, labs, procedures, allergies from raw EHR | Optional LLM (Vertex AI, Hugging Face, or **local** MedGemma); without model, stub summary + placeholder entities. Output is `scribe_summary` (and stub `extracted_entities` when no model). |
| **Auditor** | Compare extracted data to guidelines; flag drug interactions and risks | Calls `lookup_guidelines(query)` and `check_interactions(medications, conditions)`. Tools are **stub implementations** (mock returns) so the flow is testable without external APIs. Writes `clinical_risks`, `guideline_checks`, `auditor_notes`. |
| **Verifier** | Reduce hallucinations by checking summary against source | **Implemented.** Heuristic: extract key terms from `raw_ehr` and require whole-word matches in `scribe_summary` (≥50% term overlap to pass). Sets `verified_summary`, `verification_passed`, `final_notes`, `verification_status`. |

State is a single Pydantic `ClinicalState` passed through all nodes; workflow is linear (no LangGraph graph file yet—sequential function calls).

**Note:** When using the real LLM, Scribe returns JSON in `scribe_summary`; the pipeline does **not** yet parse that JSON into `state["extracted_entities"]`, so in real runs the Auditor’s entity lists may be empty unless pre-filled. Stub mode pre-fills placeholder entities.

---

## What’s real vs stub

| Component | Default (no `.env` / no `USE_MEDGEMMA`) | With `USE_MEDGEMMA=1` (Vertex / HF / local) |
|-----------|------------------------------------------|-------------------------------------------|
| **Scribe** | Stub: no LLM; placeholder summary and `extracted_entities` | Real: Vertex AI (Gemini), Hugging Face API, or **local** MedGemma for entity extraction |
| **Auditor** | Full logic; tools return mock data | Same; tools still stub (no live RAG or drug API) |
| **Verifier** | Full heuristic (key-term cross-check) | Same |
| **Workflow & tests** | Full pipeline, 8 tests | Same; Scribe uses real model when configured |

---

## Run with real LLM (optional)

**Vertex AI (recommended):** The default when `USE_MEDGEMMA=1`. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), run `gcloud auth application-default login` and `gcloud config set project YOUR_PROJECT_ID`. Copy `.env.example` to `.env` and set `GOOGLE_CLOUD_PROJECT=your-project-id`, `USE_MEDGEMMA=1`, `USE_MEDGEMMA_BACKEND=vertex`. Run `python -m src.main`.

**Hugging Face (cloud API):** Set `USE_MEDGEMMA_BACKEND=huggingface` and `HF_TOKEN` in `.env`. See `.env.example`.

**Local MedGemma (optional, for future use):** The smallest Google MedGemma is 4B (~10GB disk, significant RAM/GPU). Local backend code is in place; set `USE_MEDGEMMA_BACKEND=local`, install `requirements-local.txt`, and point `MEDGEMMA_LOCAL_MODEL` at a local model folder or HF id. Prefer Vertex for typical use.

---

## Repo layout

- `src/main.py` — Entry point; builds initial state, runs workflow, prints result.
- `src/graphs/clinical_workflow.py` — `run_workflow(initial_state, model)` (Scribe → Auditor → Verifier).
- `src/agents/` — `scribe_node`, `auditor_node`, `verifier_node`.
- `src/state.py` — `ClinicalState`, `ExtractedMedicalEntities`, `ClinicalRisk`, `VerificationStatus`.
- `src/tools/` — `medical_db.lookup_guidelines`, `drug_api.check_interactions` (stubs).
- `src/models.py` — `get_medgemma_model(backend)` for Vertex, Hugging Face, or local.
- `tests/` — Workflow, scribe, auditor, verifier (including whole-word match), state tests.

---

## Tech stack

- **Orchestration:** LangChain-style nodes and state dict; model via `get_medgemma_model()`.
- **Model:** Gemini on Vertex AI (default when `USE_MEDGEMMA=1`), Hugging Face API, or **local** MedGemma (4B on CPU/GPU); optional.
- **Config:** `python-dotenv` from project root; `.env` optional (stub runs without it).
