"""
Load a real MedGemma (or Gemma) model for the workflow.
Supports: Vertex AI (GCP), Hugging Face Inference API, or local (downloaded model on CPU/GPU).
Set USE_MEDGEMMA=1 and backend in .env to enable.
"""

import os
from typing import Any

# Backend: "vertex" | "huggingface" | "local" | None (no real model)
MEDGEMMA_BACKEND = os.getenv("USE_MEDGEMMA_BACKEND", "").strip().lower() or None
if os.getenv("USE_MEDGEMMA", "").strip().lower() in ("1", "true", "yes"):
    MEDGEMMA_BACKEND = MEDGEMMA_BACKEND or "vertex"


def get_medgemma_model(backend: str | None = None) -> Any:
    """
    Return a LangChain-compatible chat model (has .invoke(messages) -> response with .content).
    backend: "vertex" | "huggingface" | "local" | None (use env USE_MEDGEMMA_BACKEND).
    Returns None if backend is None or loading fails (caller can run without model).
    """
    backend = (backend or MEDGEMMA_BACKEND or "").strip().lower()
    if not backend:
        return None

    if backend == "vertex":
        return _get_vertex_model()
    if backend == "huggingface":
        return _get_huggingface_model()
    if backend == "local":
        return _get_local_model()
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


class _LocalChatWrapper:
    """Thin wrapper so local model matches .invoke(messages) -> response with .content."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str, max_new_tokens: int) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

    def invoke(self, messages: list) -> Any:
        import torch
        # Convert LangChain messages to HF chat format
        hf_messages = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
            content = getattr(m, "content", None) or str(m)
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                )
            if role == "system":
                hf_messages.append({"role": "user", "content": f"System: {content}"})
            elif role in ("user", "human"):
                hf_messages.append({"role": "user", "content": content})
        if not hf_messages:
            return _Response(content="")
        # Use tokenizer chat template when available
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                hf_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: Gemma-style
            parts = [
                f"<start_of_turn>user\n{m['content']}<end_of_turn>" for m in hf_messages
            ]
            prompt = "\n".join(parts) + "\n<start_of_turn>model\n"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return _Response(content=text)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


def _is_local_model_path(value: str) -> bool:
    """True if value is an existing directory (load from disk, no Hugging Face)."""
    if not value or not isinstance(value, str):
        return False
    return os.path.isdir(value.strip())


def _get_local_model() -> Any:
    """
    Local MedGemma: load from a local folder (no Hugging Face at runtime) or from HF.
    Set MEDGEMMA_LOCAL_MODEL to a path like ./models/medgemma-2-4b-it or C:\\path\\to\\model.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "For local backend install: pip install transformers torch accelerate. "
            "See README 'Run MedGemma locally' and requirements-local.txt."
        ) from None

    model_id = (os.getenv("MEDGEMMA_LOCAL_MODEL") or "google/medgemma-2-4b-it").strip()
    device = os.getenv("MEDGEMMA_LOCAL_DEVICE", "").strip().lower()
    if device not in ("cuda", "cpu", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = int(os.getenv("MEDGEMMA_LOCAL_MAX_TOKENS", "2048"))

    use_local_files = _is_local_model_path(model_id)
    if use_local_files:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            device_map=device if device == "auto" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token or True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token or True,
            device_map=device if device == "auto" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    if device in ("cuda", "cpu"):
        model = model.to(device)

    return _LocalChatWrapper(model, tokenizer, model_id, max_new_tokens)
