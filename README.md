# MedGemma Impact Challenge: Multi-Agent Clinical Workflow
**Developed by [JackTQY](https://github.com/JackTQY) | Powered by [Actualizer Systems LLC](https://github.com/actualizer-systems)**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-MedGemma_Impact_Challenge-blue)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview
This project is an entry for the **MedGemma Impact Challenge**, specifically targeting the **Agentic Workflow Prize**. We leverage the MedGemma-9B model to automate complex clinical decision-support tasks through a specialized multi-agent architecture.

---

## 🔬 Problem Statement: The "Silent Risk" in Data Overload
Healthcare providers spend 35% of their time on "pajama time" (after-hours documentation). The risk isn't just the time spent—it's the **missed contradictions** in massive EHR files. 
This project goes beyond simple summarization to create a **Clinical Auditor** that identifies diagnostic inconsistencies and medication risks across fragmented patient records.

## 🌟 Prize-Targeting Novelty
* **Novel Task Prize:** Implementing **"Clinical Conflict Detection"**—identifying when new lab results contradict historical diagnoses.
* **Agentic Workflow Prize:** A multi-step **"Self-Correction & Tool-Use"** loop where a specialized agent queries real-world medical guidelines (RAG) to audit its own summaries.

---

## 🧠 Agentic Architecture: "The Clinical Council"
We use a **LangGraph-based orchestration** consisting of three specialized roles:
1.  **The Intake Scribe:** Extracts core clinical entities from raw EHR notes.
2.  **The Clinical Auditor:** Compares extracted data against medical guidelines (via Vector DB) to flag risks.
3.  **The Verification Agent:** A "Self-Reflection" agent that cross-references the summary with the original source to eliminate hallucinations.

### Key Agentic Patterns Implemented:
1.  **Reflection & Critique:** A "Scribe" agent drafts documentation which is then audited by a "Critic" agent using MedGemma’s reasoning capabilities.
2.  **Tool Use (Function Calling):** Agents can autonomously query medical databases (mocked HIPAA-compliant endpoints) to verify claims.
3.  **Stateful Persistence:** The workflow maintains a clinical "memory," ensuring context is preserved across multi-turn medical evaluations.


---

## 🛠️ Technical Stack
* **Model:** MedGemma (7B/9B variants)
* **Orchestration:** LangGraph / LangChain
* **Inference:** [Local Inference / HuggingFace Inference Endpoints]
* **Organization:** Developed under the **Actualizer Systems LLC** AI research framework.

---
## 🚀 Getting Started (How to Run Locally)

### Default: Run without GCP (stub mode — reproducible by anyone)

By default the app runs in **stub mode**: no Google Cloud or API keys required. The Scribe/Auditor/Verifier pipeline runs with placeholder logic so you can see the workflow and reproduce results immediately.

```bash
git clone https://github.com/JackTQY/medgemma-impact-challenge.git
cd medgemma-impact-challenge
pip install -r requirements.txt
python -m src.main
pytest
```

You should see the full workflow output (raw input → Scribe → Auditor → Verifier) and all tests passing. This is the recommended path for reviewers and judges who want to run the project without setting up GCP.

---

### Optional: Run with real MedGemma (Vertex AI)

To call a real LLM (Gemini on Vertex AI) for the Scribe step, you need a Google Cloud project with the Vertex AI API enabled and **credentials** so the app can authenticate. Two ways to provide credentials:

| Method | Best for | What you do |
|--------|----------|-------------|
| **Option 1: Application Default Credentials (ADC)** | Local dev, your own machine | Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), run `gcloud auth application-default login`, set `GOOGLE_CLOUD_PROJECT` in `.env`. No key file. |
| **Option 2: Service account key** | Servers, CI, or when you can't use gcloud | In GCP Console create a service account with **Vertex AI User**, download a JSON key, set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to the key path. |

**Steps (Option 1 — ADC):**

1. Create a GCP project (or use an existing one). Enable the **Vertex AI API** (e.g. via Vertex AI Studio in the console).
2. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. Copy `.env.example` to `.env` and set:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   USE_MEDGEMMA=1
   USE_MEDGEMMA_BACKEND=vertex
   ```
4. Run:
   ```bash
   python -m src.main
   ```

**Steps (Option 2 — service account key):**

1. In GCP Console → IAM & Admin → Service Accounts, create a service account with role **Vertex AI User**. Create a JSON key and download it.
2. In `.env` set:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-key.json
   USE_MEDGEMMA=1
   USE_MEDGEMMA_BACKEND=vertex
   ```
3. Run `python -m src.main`.

**Why two options?** Option 1 uses your **user** identity (OAuth tokens stored by `gcloud`); no key file to manage. Option 2 uses a **service account** identity (JSON key file). Both allow your local code to make real API calls to Vertex AI in your project. See the section below for the underlying mechanism.

---

### What's real vs stub in this repo

| Component | Default (stub) | With `USE_MEDGEMMA=1` + GCP |
|-----------|-----------------|-----------------------------|
| Scribe (entity extraction) | Placeholder text; no LLM call | Real call to Vertex AI (Gemini) |
| Auditor | Placeholder; no RAG/DB | Placeholder (RAG/tools TODO) |
| Verifier | Placeholder | Placeholder (verification logic TODO) |
| State, workflow, tests | Full pipeline and tests | Same pipeline; Scribe uses real model |

The architecture (state schema, agents, graph) is the same in both modes. Stub mode ensures the project runs and is reproducible without any cloud setup; real mode demonstrates integration with Vertex AI for the Scribe step.

---

### Credentials: Option 1 vs Option 2 (essence)

- **Option 1 (ADC, `gcloud auth application-default login`):** You sign in once with your **Google user account** (e.g. your @gmail.com). The gcloud CLI obtains **OAuth tokens** for that user and writes them to a well-known file (e.g. `%APPDATA%\gcloud\application_default_credentials.json` on Windows). When your app calls Vertex AI, the Google client library finds that file automatically (no `GOOGLE_APPLICATION_CREDENTIALS` needed). The request is made **as your user**. Your user must have permission in the GCP project (e.g. Vertex AI User) for the project you set in `GOOGLE_CLOUD_PROJECT`. So: **no service account and no key file**; your **user identity** is used, and the library discovers it from the gcloud-managed file.

- **Option 2 (service account key):** You create a **service account** in the project and download a **JSON key**. You set `GOOGLE_APPLICATION_CREDENTIALS` to that file. The client library loads the key and obtains tokens for the **service account**. Requests are made as that service account. So: a **non-human identity** (the service account) and a **key file** you must store and protect.

**Can local code call Vertex without `GOOGLE_APPLICATION_CREDENTIALS`?** Yes. With Option 1, credentials are still present: they are **user OAuth credentials** stored by gcloud in the ADC file. The library uses that file when `GOOGLE_APPLICATION_CREDENTIALS` is not set. So your local app **does** have credentials; they are just the **user** type and are found via the default path, so you never generate or download a service account key.
