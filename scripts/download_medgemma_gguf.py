"""
Download a single quantized MedGemma GGUF file (~1.8–2.6 GB) for the local_gguf backend.
Run from project root: python scripts/download_medgemma_gguf.py

Requires: pip install -r requirements-local.txt (or at least huggingface_hub).
Set HF_TOKEN in .env if the model is gated.

Quantization options (MEDGEMMA_GGUF_QUANT env, default Q4_K_M):
  Q2_K     ~1.8 GB  (smallest, lower quality)
  Q4_K_S   ~2.5 GB
  Q4_K_M   ~2.6 GB  (recommended)
  Q5_K_M   ~3.2 GB
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "unsloth/medgemma-4b-it-GGUF"
# Filename pattern on the repo (unsloth uses medgemma-4b-it-<QUANT>.gguf)
QUANT_TO_FILENAME = {
    "Q2_K": "medgemma-4b-it-Q2_K.gguf",
    "Q3_K_S": "medgemma-4b-it-Q3_K_S.gguf",
    "Q3_K_M": "medgemma-4b-it-Q3_K_M.gguf",
    "Q4_K_S": "medgemma-4b-it-Q4_K_S.gguf",
    "Q4_K_M": "medgemma-4b-it-Q4_K_M.gguf",
    "Q5_K_M": "medgemma-4b-it-Q5_K_M.gguf",
    "Q8_0": "medgemma-4b-it-Q8_0.gguf",
}
DEFAULT_QUANT = "Q4_K_M"
OUTPUT_DIR = PROJECT_ROOT / "models" / "medgemma-4b-it-gguf"


def main() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    quant = (os.getenv("MEDGEMMA_GGUF_QUANT") or DEFAULT_QUANT).strip()
    filename = QUANT_TO_FILENAME.get(quant) or f"medgemma-4b-it-{quant}.gguf"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub first: pip install -r requirements-local.txt")
        raise SystemExit(1) from None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_file = OUTPUT_DIR / filename
    print(f"Downloading {REPO_ID} / {filename} to {local_file} ...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(OUTPUT_DIR),
        token=token or None,
    )
    env_value = (OUTPUT_DIR / filename).relative_to(PROJECT_ROOT).as_posix()
    print("Done. To use this model:")
    print("  1. pip install llama-cpp-python langchain-community  # see requirements-gguf.txt")
    print("  2. In .env set: USE_MEDGEMMA=1, USE_MEDGEMMA_BACKEND=local_gguf")
    print(f"  3. Set: MEDGEMMA_LOCAL_GGUF={env_value}")
    print("  4. Run from project root: python -m src.main")


if __name__ == "__main__":
    main()
