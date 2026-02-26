"""
Load a real MedGemma (or Gemma) model for the workflow.
Supports: Vertex AI (GCP), Hugging Face Inference API, or local (downloaded model on CPU/GPU).
Set USE_MEDGEMMA=1 and backend in .env to enable.
"""

import os
import time
from typing import Any

# Backend: "vertex" | "huggingface" | "local" | "local_gguf" | None (no real model)
MEDGEMMA_BACKEND = os.getenv("USE_MEDGEMMA_BACKEND", "").strip().lower() or None

# Set by _LocalChatWrapper.invoke() after each call so callers can attach token counts to their log
_LAST_LLM_CALL_TOKENS: dict | None = None
if os.getenv("USE_MEDGEMMA", "").strip().lower() in ("1", "true", "yes"):
    MEDGEMMA_BACKEND = MEDGEMMA_BACKEND or "vertex"


def get_medgemma_model(backend: str | None = None) -> Any:
    """
    Return a LangChain-compatible chat model (has .invoke(messages) -> response with .content).
    backend: "vertex" | "huggingface" | "local" | "local_gguf" | None (use env USE_MEDGEMMA_BACKEND).
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
    if backend == "local_gguf":
        return _get_local_gguf_model()
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

    model_id = os.getenv("MEDGEMMA_HF_MODEL", "google/medgemma-4b-it")
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
        global _LAST_LLM_CALL_TOKENS
        import torch
        _LAST_LLM_CALL_TOKENS = None
        # Convert LangChain messages to HF chat format (must alternate user/assistant)
        system_parts = []
        user_parts = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
            content = getattr(m, "content", None) or str(m)
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                )
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "human"):
                user_parts.append(content)
        # Single user turn: system as prefix then user content (avoids user/user alternation error)
        user_content = "\n\n".join(system_parts + user_parts) if system_parts else "\n\n".join(user_parts)
        if not user_content.strip():
            return _Response(content="")
        hf_messages = [{"role": "user", "content": user_content}]
        t0 = time.perf_counter()
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
        t_tokenize = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        t_generate = time.perf_counter() - t1
        t2 = time.perf_counter()
        # Decode only the new tokens
        new_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        t_decode = time.perf_counter() - t2
        n_out = new_ids.shape[1]
        n_prompt = inputs["input_ids"].shape[1]
        _print_llm_call_summary(
            method_name="invoke",
            prompt=prompt,
            n_prompt_tokens=n_prompt,
            response_text=text,
            n_output_tokens=n_out,
            t_tokenize=t_tokenize,
            t_generate=t_generate,
            t_decode=t_decode,
        )
        _LAST_LLM_CALL_TOKENS = {"n_prompt_tokens": n_prompt, "n_output_tokens": n_out}
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
    Set MEDGEMMA_LOCAL_MODEL to a path like ./models/medgemma-1.5-4b-it or C:\\path\\to\\model.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "For local backend install: pip install transformers torch accelerate. "
            "See README 'Run MedGemma locally' and requirements-local.txt."
        ) from None

    model_id = (os.getenv("MEDGEMMA_LOCAL_MODEL") or "models/medgemma-1.5-4b-it").strip()
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
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token or True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token or True,
            device_map=device if device == "auto" else None,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    if device in ("cuda", "cpu"):
        model = model.to(device)

    _print_local_model_summary(model, device)
    return _LocalChatWrapper(model, tokenizer, model_id, max_new_tokens)


def _print_local_model_summary(model: Any, device: str) -> None:
    """Print a short summary of the loaded model (layers, heads, size, etc.)."""
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return
        # Handle Gemma3/MedGemma: text_config + optional vision_config
        text_cfg = getattr(cfg, "text_config", cfg)
        n_layers = getattr(text_cfg, "num_hidden_layers", None)
        n_heads = getattr(text_cfg, "num_attention_heads", None)
        hidden = getattr(text_cfg, "hidden_size", None)
        vocab = getattr(text_cfg, "vocab_size", getattr(cfg, "vocab_size", None))
        vision_cfg = getattr(cfg, "vision_config", None)
        n_vision_layers = getattr(vision_cfg, "num_hidden_layers", None) if vision_cfg else None
        n_params = sum(p.numel() for p in model.parameters())
        parts = [f"Local model loaded on {device.upper()}"]
        if n_params:
            parts.append(f"parameters: {n_params / 1e9:.2f}B")
        if n_layers is not None:
            parts.append(f"text layers: {n_layers}")
        if n_vision_layers is not None:
            parts.append(f"vision layers: {n_vision_layers}")
        if n_heads is not None:
            parts.append(f"attention heads: {n_heads}")
        if hidden is not None:
            parts.append(f"hidden size: {hidden}")
        if vocab is not None:
            parts.append(f"vocab size: {vocab:,}")
        print(" | ".join(parts))
    except Exception:
        pass


# Max characters to show for input/output in LLM call summary (avoid flooding console)
_LLM_SUMMARY_INPUT_MAX_CHARS = 1024
_LLM_SUMMARY_OUTPUT_MAX_CHARS = 2048


def _truncate(s: str, max_chars: int) -> str:
    """Return s or s[:max_chars] + '...' if longer."""
    if not s or len(s) <= max_chars:
        return s or ""
    return s[:max_chars].rstrip() + "..."


def _print_llm_call_summary(
    method_name: str,
    prompt: str,
    n_prompt_tokens: int,
    response_text: str,
    n_output_tokens: int,
    t_tokenize: float,
    t_generate: float,
    t_decode: float,
) -> None:
    """Print method name, input summary, output summary, then timing (after output, wrapped in #######)."""
    total = t_tokenize + t_generate + t_decode
    n_chars_in = len(prompt)
    n_chars_out = len(response_text)
    print(f"LLM call: {method_name}")
    # Input (demarcated)
    print(f"  input ({n_chars_in} chars, {n_prompt_tokens} prompt tokens):")
    if prompt.strip():
        to_show = prompt if len(prompt) <= _LLM_SUMMARY_INPUT_MAX_CHARS else _truncate(prompt, _LLM_SUMMARY_INPUT_MAX_CHARS)
        print('  """')
        for line in to_show.splitlines():
            print("  " + line)
        print('  """')
    else:
        print('  ""')
    # Output (demarcated)
    print(f"  output ({n_chars_out} chars, {n_output_tokens} tokens):")
    if response_text.strip():
        to_show = response_text if len(response_text) <= _LLM_SUMMARY_OUTPUT_MAX_CHARS else _truncate(response_text, _LLM_SUMMARY_OUTPUT_MAX_CHARS)
        print('  """')
        for line in to_show.splitlines():
            print("  " + line)
        print('  """')
    else:
        print('  ""')
    # Timing after result, conspicuous
    parts = [
        f"tokenize {t_tokenize:.2f}s",
        f"generate {t_generate:.2f}s",
        f"decode {t_decode:.2f}s",
        f"total {total:.2f}s",
    ]
    if n_output_tokens > 0 and t_generate > 0:
        parts.append(f"({n_output_tokens} tokens, ~{n_output_tokens / t_generate:.1f} tok/s)")
    timing_line = "LLM timing: " + " | ".join(parts)
    print()
    print("####### " + timing_line + " #######")
    print()


def _get_local_gguf_model() -> Any:
    """
    Local MedGemma via GGUF (quantized, ~1.8–2.6 GB download).
    Set MEDGEMMA_LOCAL_GGUF to the path to a .gguf file (e.g. models/medgemma-4b-it-gguf/medgemma-4b-it-Q4_K_M.gguf).
    Use scripts/download_medgemma_gguf.py to download. Requires: pip install -r requirements-gguf.txt
    """
    try:
        from langchain_community.chat_models import ChatLlamaCpp
    except ImportError:
        raise ImportError(
            "For local_gguf backend install: pip install llama-cpp-python langchain-community. "
            "See requirements-gguf.txt and README 'Quantized (GGUF)'."
        ) from None

    model_path = (os.getenv("MEDGEMMA_LOCAL_GGUF") or "").strip()
    if not model_path:
        raise ValueError(
            "Set MEDGEMMA_LOCAL_GGUF to the path to your .gguf file (e.g. models/medgemma-4b-it-gguf/medgemma-4b-it-Q4_K_M.gguf). "
            "Run: python scripts/download_medgemma_gguf.py"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MEDGEMMA_LOCAL_GGUF file not found: {model_path}")

    n_ctx = int(os.getenv("MEDGEMMA_GGUF_N_CTX", "4096"))
    n_gpu_layers = int(os.getenv("MEDGEMMA_GGUF_N_GPU_LAYERS", "-1"))  # -1 = all on GPU if available
    return ChatLlamaCpp(
        model_path=model_path,
        temperature=0.2,
        max_tokens=int(os.getenv("MEDGEMMA_LOCAL_MAX_TOKENS", "2048")),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
