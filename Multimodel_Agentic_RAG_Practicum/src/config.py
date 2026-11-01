"""Central configuration for the USFR Agentic RAG system.

All tunables live here and can be overridden through environment variables
(loaded from `.env`). Includes helpers to validate and persist API keys to `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, set_key

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_store"))

UNIFIED_CSV = PROCESSED_DIR / "net_cost_unified.csv"
UNIFIED_MD = PROCESSED_DIR / "net_cost_readable.md"

for _d in (RAW_DATA_DIR, PROCESSED_DIR, UPLOAD_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM & API Key Management
# ---------------------------------------------------------------------------
_PLACEHOLDER_KEYS = {
    "",
    "your-gemini-api-key-here",
    "your_gemini_api_key_here",
    "your-api-key-here",
    "<your-gemini-api-key>",
}


def _clean_key(raw: str | None) -> str | None:
    if not raw:
        return None
    val = raw.strip().strip('"').strip("'")
    return None if val.lower() in _PLACEHOLDER_KEYS else val


# "gemini" (development default) or "ollama" (fully open-source, local).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = _clean_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))


def save_llm_settings(
    *,
    provider: str = "gemini",
    api_key: str | None = None,
    gemini_model: str | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    persist_to_env: bool = True,
    env_path: Path | None = None,
) -> Path:
    """Update runtime LLM settings and optionally persist them to `.env`."""
    global LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL

    target_env = env_path or ENV_FILE
    if persist_to_env and not target_env.exists():
        example = PROJECT_ROOT / ".env.example"
        target_env.write_text(
            example.read_text(encoding="utf-8") if example.exists() else "",
            encoding="utf-8",
        )

    LLM_PROVIDER = provider.strip().lower()
    os.environ["LLM_PROVIDER"] = LLM_PROVIDER
    if persist_to_env:
        set_key(str(target_env), "LLM_PROVIDER", LLM_PROVIDER, quote_mode="never")

    if api_key is not None:
        cleaned = _clean_key(api_key)
        GEMINI_API_KEY = cleaned
        if cleaned:
            os.environ["GEMINI_API_KEY"] = cleaned
            os.environ["GOOGLE_API_KEY"] = cleaned
            if persist_to_env:
                set_key(str(target_env), "GEMINI_API_KEY", cleaned, quote_mode="never")

    if gemini_model:
        GEMINI_MODEL = gemini_model.strip()
        os.environ["GEMINI_MODEL"] = GEMINI_MODEL
        if persist_to_env:
            set_key(str(target_env), "GEMINI_MODEL", GEMINI_MODEL, quote_mode="never")

    if ollama_model:
        OLLAMA_MODEL = ollama_model.strip()
        os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
        if persist_to_env:
            set_key(str(target_env), "OLLAMA_MODEL", OLLAMA_MODEL, quote_mode="never")

    if ollama_base_url:
        OLLAMA_BASE_URL = ollama_base_url.strip()
        os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
        if persist_to_env:
            set_key(str(target_env), "OLLAMA_BASE_URL", OLLAMA_BASE_URL, quote_mode="never")

    return target_env


# ---------------------------------------------------------------------------
# Embeddings / Vector store (open source, local)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "usfr_net_cost")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "8"))

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_VALIDATION_RETRIES = int(os.getenv("MAX_VALIDATION_RETRIES", "1"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls", ".json", ".xml"}

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASET_NAME = "Financial Report of the U.S. Government — Statements of Net Cost"
DATASET_URL = (
    "https://fiscaldata.treasury.gov/datasets/u-s-government-financial-report/"
    "statements-of-net-cost"
)
