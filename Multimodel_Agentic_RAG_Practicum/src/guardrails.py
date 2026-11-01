"""Reliability & safety controls (input + output guardrails)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src import config

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now (?!asking)",
    r"reveal (your|the) (system )?prompt",
    r"act as (an? )?(unfiltered|jailbroken|dan)",
    r"\bjailbreak\b",
    r"print (your|the) (api|secret) key",
]
UNSAFE_PATTERNS = [
    r"\b(build|make|synthesi[sz]e) (a )?(bomb|explosive|bioweapon|nerve agent)\b",
    r"\b(hack|exploit|ddos)\b.*\b(treasury|government|irs)\b",
]
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit card": r"\b(?:\d[ -]?){13,16}\b",
}


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    cleaned: str = ""


def check_input(question: str) -> GuardResult:
    q = (question or "").strip()
    if not q:
        return GuardResult(False, "Please enter a question.")
    if len(q) > config.MAX_QUESTION_CHARS:
        return GuardResult(False, f"Question is too long ({len(q)} chars). "
                                  f"Please keep it under {config.MAX_QUESTION_CHARS} characters.")
    low = q.lower()
    if any(re.search(p, low) for p in INJECTION_PATTERNS):
        return GuardResult(False, "This request looks like a prompt-injection attempt and was blocked. "
                                  "Please ask a question about the U.S. Government financial data.")
    if any(re.search(p, low) for p in UNSAFE_PATTERNS):
        return GuardResult(False, "This request is outside the safe-use policy of this assistant.")
    for label, pat in PII_PATTERNS.items():
        if re.search(pat, q):
            return GuardResult(False, f"Your question appears to contain sensitive personal data ({label}). "
                                      "Please remove it and try again.")
    # strip control characters
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", q)
    return GuardResult(True, cleaned=cleaned)


# ---------------------------------------------------------------------------
# Output grounding check (deterministic, no LLM needed)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(trillion|billion|tn|bn|t\b|b\b|%)?",
                  re.IGNORECASE)


def _numbers(text: str) -> list[tuple[float, str, bool]]:
    """Extract (value, unit, has_decimal) tuples, skipping years and citation markers."""
    text = re.sub(r"\[S\d+\]", " ", text)
    text = re.sub(r"\bFY\s?\d{2,4}\b", " ", text, flags=re.IGNORECASE)
    out = []
    for m in _NUM.finditer(text):
        whole, dec, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        val = float((whole + dec).replace(",", ""))
        if not dec and 1900 <= val <= 2100 and unit != "%":   # a year
            continue
        out.append((val, unit, bool(dec)))
    return out


def ungrounded_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that cannot be found (within rounding) in the context."""
    ctx_vals = [abs(v) for v, _, _ in _numbers(context)]
    bad = []
    for val, unit, has_dec in _numbers(answer):
        v = abs(val)
        if not has_dec and v < 50 and unit not in {"%"}:   # counts / ranks ("top 5")
            continue
        candidates = [v]
        if unit in {"trillion", "tn", "t"}:
            candidates.append(v * 1000)                     # context is in billions
        ok = False
        for c in candidates:
            tol = max(0.051, 0.03 * c) if c != v else 0.051
            if any(abs(c - x) <= tol for x in ctx_vals):
                ok = True
                break
        if not ok:
            bad.append(f"{val:g}{(' ' + unit) if unit else ''}")
    return bad


def has_citations(answer: str) -> bool:
    return bool(re.search(r"\[S\d+\]", answer))
