"""
Load a real MedGemma (or Gemma) model for the workflow.
Supports: Vertex AI (GCP), Hugging Face Inference API.
Set USE_MEDGEMMA=1 and backend in .env to enable.
"""

import os
from typing import Any

# Backend: "vertex" | "huggingface" | None (no real model)
MEDGEMMA_BACKEND = os.getenv("USE_MEDGEMMA_BACKEND", "").strip().lower() or None
if os.getenv("USE_MEDGEMMA", "").strip().lower() in ("1", "true", "yes"):
    MEDGEMMA_BACKEND = MEDGEMMA_BACKEND or "vertex"


def get_medgemma_model(backend: str | None = None) -> Any:
    """
    Return a LangChain-compatible chat model (has .invoke(messages) -> response with .content).
    backend: "vertex" | "huggingface" | None (use env USE_MEDGEMMA_BACKEND).
    Returns None if backend is None or loading fails (caller can run without model).
    """
    backend = (backend or MEDGEMMA_BACKEND or "").strip().lower()
    if not backend:
        return None

    if backend == "vertex":
        return _get_vertex_model()
    if backend == "huggingface":
        return _get_huggingface_model()
    return None


def _get_vertex_model() -> Any:
    """Vertex AI: ChatGoogleGenerativeAI (langchain-google-genai) with Vertex backend."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("Install: pip install langchain-google-genai") from None

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if "GOOGLE_CLOUD_LOCATION" not in os.environ:
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
    model_id = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.0-flash-exp")
    kwargs = dict(
        model=model_id,
        temperature=0.2,
        max_output_tokens=2048,
        vertexai=True,
    )
    if project:
        kwargs["project"] = project
    return ChatGoogleGenerativeAI(**kwargs)


def _get_huggingface_model() -> Any:
    """Hugging Face: MedGemma via Inference API. Requires HF_TOKEN."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in .env for Hugging Face backend")

    model_id = os.getenv("MEDGEMMA_HF_MODEL", "google/medgemma-2-4b-it")
    try:
        from langchain_huggingface import ChatHuggingFace
        return ChatHuggingFace(
            model_id=model_id,
            huggingfacehub_api_token=token,
            temperature=0.2,
            max_new_tokens=2048,
        )
    except ImportError:
        raise ImportError(
            "For Hugging Face backend: pip install langchain-huggingface. "
            "Then set HF_TOKEN in .env. Or use Vertex: USE_MEDGEMMA_BACKEND=vertex"
        ) from None
