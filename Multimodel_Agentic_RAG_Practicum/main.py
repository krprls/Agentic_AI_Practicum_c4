"""Terminal REPL for the USFR agentic RAG assistant. Type 'quit' to exit."""
from __future__ import annotations

import getpass

from src import config
from src.agents.graph import run_agent
from src.llm import LLMConfigError, get_llm, llm_status


def _ensure_api_key_interactive() -> None:
    """Prompt the user for a Gemini API key if provider=='gemini' and no key is configured."""
    ok, desc = llm_status()
    if ok:
        print(f"📡 Active LLM: {desc}\n")
        return

    print("⚠️  GEMINI_API_KEY is not configured in .env.")
    print("   Get a free key at: https://aistudio.google.com/app/apikey")
    choice = input("   Paste your GEMINI_API_KEY now (or type 'ollama' to use local Ollama, or Enter to skip): ").strip()
    if not choice:
        try:
            choice = getpass.getpass("   Hidden GEMINI_API_KEY (or Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""
    if not choice:
        return
    if choice.lower() == "ollama":
        config.save_llm_settings(provider="ollama", persist_to_env=True)
        get_llm.cache_clear()
        print(f"✅ Switched to Ollama ({config.OLLAMA_MODEL}) and saved to .env!\n")
    else:
        config.save_llm_settings(provider="gemini", api_key=choice, persist_to_env=True)
        get_llm.cache_clear()
        print("✅ Saved GEMINI_API_KEY to .env!\n")


def main() -> None:
    print("🇺🇸 U.S. Government Financial Report — Agentic RAG (type 'quit' to exit, 'key' to update API key)\n")
    _ensure_api_key_interactive()
    history: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q.lower() in {"key", "set-key", "apikey"}:
            new_key = input("Enter new GEMINI_API_KEY: ").strip()
            if new_key:
                config.save_llm_settings(provider="gemini", api_key=new_key, persist_to_env=True)
                get_llm.cache_clear()
                print("✅ Updated GEMINI_API_KEY in .env!\n")
            continue
        if not q:
            continue
        try:
            result = run_agent(q, history)
        except LLMConfigError as exc:
            print(f"⚠️  {exc}")
            print("💡 Tip: Type 'key' to enter and save your GEMINI_API_KEY to .env.\n")
            continue
        answer = result.get("answer", "")
        print(f"\nAssistant: {answer}\n")
        for s in result.get("sources", []):
            print(f"   [{s['id']}] {s['title']} — {s['source']}")
        print("   trace: " + " → ".join(f"{t['node']}({t['status']})" for t in result.get("trace", [])) + "\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
