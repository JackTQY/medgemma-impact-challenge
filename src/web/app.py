"""
Local web app for the Clinical Council agentic workflow.
Run from project root: uvicorn src.web.app:app --reload
Then open http://127.0.0.1:8000
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")
warnings.filterwarnings("ignore", message=".*without a quota project.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.graphs.clinical_workflow import run_workflow
from src.models import get_medgemma_model


app = FastAPI(
    title="Clinical Council",
    description="Agentic workflow: Scribe → Auditor → Verifier with MedGemma",
    version="1.0",
)


class RunRequest(BaseModel):
    raw_ehr: str = Field(..., min_length=1, description="Raw EHR or clinical note text")
    patient_id: str = Field(default="web-001", description="Patient or encounter ID")


def _to_jsonable(obj):
    """Recursively turn workflow state into JSON-serializable dict (Pydantic → dict, enum → value)."""
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump(mode="python"))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


# Load model once at startup (optional; can be None for stub mode)
_model = None


def _get_model():
    global _model
    if _model is None:
        backend = os.getenv("USE_MEDGEMMA_BACKEND", "").strip().lower()
        if os.getenv("USE_MEDGEMMA", "").strip().lower() in ("1", "true", "yes") or backend:
            try:
                _model = get_medgemma_model(backend or None)
            except Exception:
                _model = None
    return _model


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=500, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/run")
def run(req: RunRequest) -> JSONResponse:
    """Run the workflow (Scribe → Auditor → Verifier) and return full state as JSON."""
    initial_state = {
        "raw_ehr": req.raw_ehr.strip(),
        "patient_id": req.patient_id.strip() or "web-001",
        "__llm_call_log": [],
    }
    model = _get_model()
    try:
        result = run_workflow(initial_state, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content=_to_jsonable(result))
