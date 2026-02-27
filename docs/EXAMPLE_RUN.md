# Example run: real MedGemma (local) through the Clinical Council

This document captures **illustrative output** from running the full pipeline with a real LLM (local MedGemma 4B) using the **complex** sample EHR. It shows hiring managers and reviewers the sophistication of the code path, the quality of the test case, and the end-to-end behaviour: structured extraction → audit → verification.

**Two samples:** You can run a **simple** sample (~73s with local MedGemma) or a **complex** sample (~524s). The CLI prompts you to choose; default is simple if you don’t answer within 5 seconds. The web app defaults to the simple sample and offers **Load simple sample** / **Load complex sample** buttons.

**How to reproduce (complex run below):** Set `USE_MEDGEMMA=1` and `USE_MEDGEMMA_BACKEND=local` in `.env`, ensure the local model is downloaded (see README), run `python -m src.main`, then choose **2** for the complex sample.

---

## 1. Model load

```
Loading weights: 100%|████████████████████████████████████████████| 883/883 [00:05<00:00, 155.66it/s, ...]
Local model loaded on CPU | parameters: 4.30B | text layers: 34 | vision layers: 27 | attention heads: 8 | hidden size: 2560 | vocab size: 262,208
Using real model (backend=local). Running workflow...
```

**What this shows:** The pipeline supports multiple backends (Vertex, Hugging Face, local full 4B, local GGUF). This run uses the full MedGemma 4B on CPU; the same code path runs on cloud or quantized models.

---

## 2. LLM call (Scribe): prompt and response

The Scribe agent sends a **system prompt** (role + instructions + output format) and a **user message** (the raw EHR). The model returns a single structured JSON block.

### Input (trimmed)

- **Prompt size:** 1285 chars, 401 prompt tokens.
- **System instructions:** Senior Clinical Scribe; extract Diagnoses, Medications, Lab Results, Procedures, Allergies, Vitals; output valid JSON with those keys (arrays of strings or objects for labs/vitals); if not present, return empty list.
- **User content:** “Extract entities from this EHR note:” followed by the full clinical note (68yo M, PMH, meds, labs, procedures, allergies, vitals, plan).

### Output (model response)

The model returns **valid JSON** with the requested schema:

```json
{
  "Diagnoses": [
    "Hypertension",
    "Type 2 diabetes mellitus",
    "CKD stage 3a",
    "atrial fibrillation (rate-controlled)",
    "GERD",
    "hyperlipidemia"
  ],
  "Medications": [
    "Lisinopril 10mg daily",
    "Metformin 1000mg BID",
    "Apixaban 5mg BID",
    "Omeprazole 20mg daily",
    "Atorvastatin 40mg nightly",
    "Metoprolol 25mg BID"
  ],
  "Lab Results": [
    { "test": "HbA1c", "value": "7.8%", "units": "%", "date": "3mo ago" },
    { "test": "eGFR", "value": "52", "units": "mL/min/1.73m2" },
    { "test": "K+", "value": "4.2" },
    { "test": "Na", "value": "138" },
    { "test": "TSH", "value": "2.1" },
    { "test": "LDL", "value": "98", "units": "mg/dL" },
    { "test": "creatinine", "value": "1.4", "units": "mg/dL" },
    { "test": "troponin", "value": "negative" },
    { "test": "Echocardiogram", "value": "EF 55%", "units": "%", "notes": "no significant valve disease" }
  ],
  "Procedures": [ "Colonoscopy 2024: benign" ],
  "Allergies": [ "Penicillin (rash)", "Sulfa (hives)" ],
  "Vitals": [
    { "item": "BP", "value": "142", "units": "mmHg", "systolic": "142", "diastolic": "88" },
    { "item": "HR", "value": "78", "units": "bpm" },
    { "item": "weight", "value": "89", "units": "kg" },
    { "item": "BMI", "value": "28" },
    { "item": "chest pain", "value": "no" },
    { "item": "orthopnea", "value": "no" }
  ]
}
```

**What this shows:** The Scribe prompt engineering and schema are sufficient for the model to produce **consistent, parseable JSON** from a dense clinical note. The test EHR is deliberately complex (multiple diagnoses, 6 meds, many labs, procedures, allergies, vitals) to exercise extraction quality and downstream verification.

### Timing

```
LLM timing: tokenize 0.02s | generate 524.20s | decode 0.00s | total 524.22s | (703 tokens, ~1.3 tok/s)
```

---

## 3. Workflow result (all four stages)

### 1. Raw input

- **Patient ID:** sample-001  
- **Raw EHR:** 68yo M, routine follow-up. PMH: Hypertension, Type 2 diabetes mellitus, CKD stage 3a, atrial fibrillation (rate-controlled), GERD, hyperlipidemia. Meds: Lisinopril 10mg daily, Metformin 1000mg BID, Apixaban 5mg BID, … (full note in repo/default in `src/main.py`).

### 2. Scribe (intake)

- **Value:** Structured extraction from free text.
- **Summary:** The JSON string from the model (stored in `scribe_summary`).
- **Extracted entities:** Normalized into `diagnoses`, `medications`, `allergies`, `lab_results`, `procedures`, `vitals` for the Auditor (e.g. 6 diagnoses, 6 medications, 2 allergies, 9 lab results, 1 procedure, 6 vitals entries).

**What this shows:** The pipeline normalizes the Scribe output into a shared state shape used by the Auditor and Verifier.

### 3. Auditor

- **Value:** Risk/conflict detection vs guidelines.
- **Notes:** “Audited 1 guideline check(s); checked drug interactions for 6 medication(s). Flagged 1 risk(s).”
- **Risk 1:** “Mock: verify dose and renal function for listed medications.”

**What this shows:** The Auditor runs **tool calls** (guideline lookup, drug-interaction check) over the extracted entities. In this run the tools are stubs (mock returns); the same code path is used with real APIs. The logic demonstrates multi-agent tool use and risk flagging.

### 4. Verifier

- **Value:** Final check vs source to reduce hallucinations.
- **Verified summary:** Same as Scribe summary (passed through).
- **Passed:** True  
- **Final notes:** “Cross-checked 74 key term(s) from source; 57 found in summary. Consistent.”

**What this shows:** The Verifier extracts key terms from `raw_ehr` and requires whole-word matches in `scribe_summary`. With a complex note and a good extraction, term overlap exceeds the threshold (e.g. ≥55%) and the run completes in one pass. If it had failed, the **LangGraph conditional edge** would route back to Scribe once (self-correction) with a retry hint in the prompt.

---

## 4. Value of the agentic workflow (printed summary)

```
VALUE OF AGENTIC WORKFLOW
- Scribe turns unstructured notes into structured entities.
- Auditor compares against guidelines and flags risks.
- Verifier cross-checks the summary to the source for safety.
```

---

## What hiring managers can take away

| Aspect | Evidence in this run |
|--------|----------------------|
| **Structured extraction** | Single LLM call yields valid JSON (diagnoses, meds, labs, procedures, allergies, vitals) from a long, dense note. |
| **Agentic design** | Three distinct agents (Scribe, Auditor, Verifier) with clear roles; state flows through a LangGraph StateGraph with checkpointing and conditional edges. |
| **Tool use** | Auditor calls `lookup_guidelines` and `check_interactions`; stubs allow tests without external APIs; same interface for real backends. |
| **Verification & self-correction** | Verifier checks summary vs source (term overlap). On failure, graph routes back to Scribe with a retry hint (documented in README and code). |
| **Testability** | Complex sample EHR and stub tools let the full pipeline run and be tested (e.g. `pytest`) without API keys or model download. |
| **Observability** | Run logs prompt size, token counts, timing, and a clear four-stage result summary. |

This example run is **one snapshot** of the same code path that runs in stub mode (no LLM), with Vertex/HF, or with local/GGUF models—demonstrating consistent architecture and test coverage across environments.
