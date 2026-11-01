"""Configure and save your LLM API key to `.env`.

Usage:
    python setup_key.py                           # interactive prompt (hidden input)
    python setup_key.py --key AIzaSy...           # pass Gemini key directly
    python setup_key.py --provider ollama         # switch to local Ollama (no API key needed)
"""
from __future__ import annotations

import argparse
import getpass

from src import config
from src.llm import get_llm, llm_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Save LLM provider and API key to .env")
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default=config.LLM_PROVIDER or "gemini",
        help="LLM provider ('gemini' or 'ollama')",
    )
    parser.add_argument("--key", help="Google Gemini API key (saved to .env)")
    parser.add_argument("--model", help="Override model name (e.g. gemini-3-flash-preview or llama3.1:8b)")
    args = parser.parse_args()

    provider = args.provider.lower()
    if provider == "gemini":
        api_key = args.key
        if not api_key:
            print("🔑 Enter your Google Gemini API Key (get one at https://aistudio.google.com/app/apikey)")
            api_key = getpass.getpass("GEMINI_API_KEY (input hidden, or press Enter to type visibly): ").strip()
            if not api_key:
                api_key = input("GEMINI_API_KEY: ").strip()
        if not api_key:
            print("⚠️  No API key entered. Exiting without changes.")
            return
        env_path = config.save_llm_settings(
            provider="gemini",
            api_key=api_key,
            gemini_model=args.model or config.GEMINI_MODEL,
            persist_to_env=True,
        )
    else:
        env_path = config.save_llm_settings(
            provider="ollama",
            ollama_model=args.model or config.OLLAMA_MODEL,
            persist_to_env=True,
        )

    get_llm.cache_clear()
    ok, desc = llm_status()
    print(f"✅ Saved settings to {env_path}")
    print(f"📡 Active LLM status: {desc} (ready={ok})")


if __name__ == "__main__":
    main()
