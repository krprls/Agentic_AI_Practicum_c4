"""LLM factory — Gemini for development, Ollama for a 100% open-source deployment.

Includes automatic model fallback across current Gemini Flash/Pro models so new API
keys work seamlessly even when older model aliases are deprecated.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src import config

GEMINI_FALLBACK_MODELS = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
)


class LLMConfigError(RuntimeError):
    """Raised when the configured LLM provider cannot be initialised."""


def _build_gemini_models(primary_model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    ordered = [primary_model] + [m for m in GEMINI_FALLBACK_MODELS if m != primary_model]
    return [
        ChatGoogleGenerativeAI(
            model=m,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_retries=1,
            timeout=60,
        )
        for m in ordered
    ]


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None) -> Any:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Create one at https://aistudio.google.com/app/apikey "
                "and add it to your .env file (or Streamlit sidebar)."
            )
        models = _build_gemini_models(model or config.GEMINI_MODEL)
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
    raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'ollama'.")


def get_structured_llm(schema: Any, provider: str | None = None, model: str | None = None) -> Any:
    """Return a structured-output runnable with model fallbacks."""
    base = get_llm(provider=provider, model=model)
    if hasattr(base, "with_structured_output"):
        return base.with_structured_output(schema)
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "gemini":
        models = [m.with_structured_output(schema) for m in _build_gemini_models(model or config.GEMINI_MODEL)]
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]
    return base


def llm_status() -> tuple[bool, str]:
    """(ok, human-readable description) for the UI."""
    p = config.LLM_PROVIDER
    if p == "gemini":
        return (
            bool(config.GEMINI_API_KEY),
            f"Gemini · {config.GEMINI_MODEL}" + ("" if config.GEMINI_API_KEY else " (API key missing)"),
        )
    if p == "ollama":
        return True, f"Ollama · {config.OLLAMA_MODEL}"
    return False, f"Unknown provider '{p}'"
