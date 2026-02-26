"""
Download the smallest official MedGemma (4B instruction-tuned) into the project.
Run from project root: python scripts/download_medgemma_local.py

Requires: pip install -r requirements-local.txt (or at least huggingface_hub).
For gated models, set HF_TOKEN in .env or in the environment.
"""
from pathlib import Path
import os

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/medgemma-4b-it"
LOCAL_DIR = PROJECT_ROOT / "models" / "medgemma-1.5-4b-it"


def main() -> None:
    # Load .env so HF_TOKEN is available
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first: pip install -r requirements-local.txt")
        raise SystemExit(1) from None

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
    snapshot_download(
        MODEL_ID,
        local_dir=str(LOCAL_DIR),
        token=token or None,
    )
    # Prefer path relative to project root so .env works from any drive
    rel = LOCAL_DIR.relative_to(PROJECT_ROOT)
    env_value = rel.as_posix()  # e.g. models/medgemma-1.5-4b-it
    print("Done. To use this model locally:")
    print("  1. In .env set: USE_MEDGEMMA=1, USE_MEDGEMMA_BACKEND=local")
    print(f"  2. Set: MEDGEMMA_LOCAL_MODEL={env_value}")
    print("  3. Run from project root: python -m src.main")


if __name__ == "__main__":
    main()
