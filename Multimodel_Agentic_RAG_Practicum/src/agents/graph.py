"""LangGraph multi-agent workflow.

    START → input_guard → planner → retriever → reasoner → validator → finalize → END
                 │            │                      ▲          │
                 └─blocked────┴─out-of-scope─────────┼──────────┤
                                                     └─retry────┘

Agents / nodes
--------------
* input_guard – safety & validation of the user question (no LLM)
* planner     – LLM: rewrites follow-ups, classifies intent, extracts agencies/years/metric,
                proposes semantic search queries (structured output, heuristic fallback)
* retriever   – executes tools (vector search + structured analytics) per the plan
* reasoner    – LLM: writes a grounded, cited answer from the numbered context only
* validator   – deterministic grounding check (numbers + citations); triggers one rewrite
* finalize    – attaches sources, caveats and the agent trace
"""
from __future__ import annotations

import logging
import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config
from src.agents import tools as T
from src.data_store import get_store
from src.guardrails import check_input, has_citations, ungrounded_numbers
from src.llm import LLMConfigError, get_llm, get_structured_llm

log = logging.getLogger(__name__)

Intent = Literal["lookup", "ranking", "comparison", "trend", "restatement",
                 "definition", "general", "greeting", "out_of_scope"]
Metric = Literal["net_cost", "gross_cost", "earned_revenue", "subtotal", "change_in_assumptions"]


# ---------------------------------------------------------------------------
# State & schemas
# ---------------------------------------------------------------------------
class Plan(BaseModel):
    """Structured plan produced by the Planner agent."""

    intent: Intent = Field(description="Type of question")
    standalone_question: str = Field(description="The question rewritten to be self-contained "
                                                 "using the chat history")
    agencies: list[str] = Field(default_factory=list, description="Agency names or abbreviations "
                                "mentioned or implied, e.g. ['Department of Defense']")
    fiscal_years: list[int] = Field(default_factory=list, description="Fiscal years referenced")
    metric: Metric = Field(default="net_cost", description="Primary financial metric")
    top_n: int = Field(default=10, description="How many items for ranking questions")
    ascending: bool = Field(default=False, description="True when the user wants the lowest/smallest")
    include_restated: bool = Field(default=False, description="True if user asks about restated/revised figures")
    search_queries: list[str] = Field(default_factory=list, description="1-3 semantic search queries")
    reasoning: str = Field(default="", description="One sentence on how to answer")


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    plan: dict
    contexts: list[dict]
    answer: str
    validation: dict
    retries: int
    blocked: bool
    direct_answer: str
    sources: list[dict]
    trace: Annotated[list[dict], operator.add]


def _text(msg: Any) -> str:
    """Normalise LLM message content (str or list-of-parts) to plain text."""
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ---------------------------------------------------------------------------
# Node: input guard
# ---------------------------------------------------------------------------
def input_guard(state: AgentState) -> AgentState:
    res = check_input(state["question"])
    if not res.ok:
        return {"blocked": True, "direct_answer": f"🛡️ {res.reason}",
                "trace": [{"node": "input_guard", "status": "blocked", "detail": res.reason}]}
    return {"question": res.cleaned, "blocked": False, "retries": 0,
            "trace": [{"node": "input_guard", "status": "passed",
                       "detail": "Length, prompt-injection, unsafe-content and PII checks passed"}]}


# ---------------------------------------------------------------------------
# Node: planner
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are the Planner agent of a question-answering system over the
"Financial Report of the U.S. Government — Statements of Net Cost" dataset
(fiscal years available: {years}; amounts in billions USD; fiscal year ends Sep 30).
Known entities: {agencies}.

Analyse the user's latest question (use chat history to resolve follow-ups like "what about 2024?")
and output a plan. Intent guide:
- lookup: specific figure(s) for agency/year(s)       - ranking: top/bottom/largest/smallest
- comparison: change between years or between agencies - trend: evolution over all years
- restatement: original vs restated/revised figures    - definition: meaning of a term/concept
- general: other questions answerable from the documents (incl. uploaded files)
- greeting: hello/thanks/chit-chat
- out_of_scope: unrelated to U.S. federal finances or the uploaded documents
For the whole government use agency "Total". Map abbreviations (DoD, VA, HHS, SSA...) to full names."""


def _heuristic_plan(question: str) -> Plan:
    store = get_store()
    q = question.lower()
    years = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", question)})
    if re.search(r"\b(top|largest|highest|biggest|most|lowest|smallest|least|rank)", q):
        intent = "ranking"
    elif re.search(r"\b(restat|revis)", q):
        intent = "restatement"
    elif re.search(r"\b(change|compare|vs|versus|differ|increase|decrease|grow)", q):
        intent = "comparison"
    elif re.search(r"\b(trend|over time|over the years|history)", q):
        intent = "trend"
    elif re.search(r"\b(what is|what does|define|meaning|explain)\b", q):
        intent = "definition"
    elif re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you)[!. ]*", q):
        intent = "greeting"
    else:
        intent = "lookup"
    metric = ("gross_cost" if "gross" in q else "earned_revenue" if "revenue" in q
              else "change_in_assumptions" if "assumption" in q else "net_cost")
    m = re.search(r"\btop\s+(\d+)", q)
    return Plan(intent=intent, standalone_question=question,
                agencies=store.find_agencies_in_text(question), fiscal_years=years, metric=metric,
                top_n=int(m.group(1)) if m else 10,
                ascending=bool(re.search(r"\b(lowest|smallest|least|negative)\b", q)),
                include_restated=intent == "restatement", search_queries=[question],
                reasoning="Heuristic plan (LLM planner unavailable)")


def planner(state: AgentState) -> AgentState:
    store = get_store()
    question = state["question"]
    history = state.get("history", [])[-6:]
    try:
        llm = get_structured_llm(Plan)
        msgs = [SystemMessage(PLANNER_SYSTEM.format(years=store.years,
                                                    agencies="; ".join(store.agencies)))]
        for h in history:
            msgs.append((HumanMessage if h["role"] == "user" else AIMessage)(h["content"][:1500]))
        msgs.append(HumanMessage(question))
        plan = llm.invoke(msgs)
        if plan is None:
            raise ValueError("empty plan")
        source = "llm"
    except LLMConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("Planner LLM failed (%s); using heuristic plan", exc)
        plan, source = _heuristic_plan(question), "heuristic"

    # Resolve agency names against the actual data + scan the question text.
    resolved: list[str] = []
    for a in plan.agencies + store.find_agencies_in_text(plan.standalone_question):
        r = store.resolve_agency(a)
        if r and r not in resolved:
            resolved.append(r)
    plan.agencies = resolved
    if not plan.search_queries:
        plan.search_queries = [plan.standalone_question]
    plan.top_n = max(1, min(plan.top_n, 40))

    detail = (f"intent={plan.intent}, agencies={plan.agencies or '—'}, years={plan.fiscal_years or '—'}, "
              f"metric={plan.metric} [{source}]")
    out: AgentState = {"plan": plan.model_dump(),
                       "trace": [{"node": "planner", "status": "done", "detail": detail,
                                  "reasoning": plan.reasoning}]}
    if plan.intent == "greeting":
        out["direct_answer"] = ("Hello! 👋 Ask me anything about U.S. Government net cost by agency for "
                                f"FY{min(store.years, default='')}–FY{max(store.years, default='')}, "
                                "or upload your own documents in the sidebar.")
    elif plan.intent == "out_of_scope":
        out["direct_answer"] = ("I can only answer questions about the Financial Report of the U.S. "
                                "Government (Statements of Net Cost) and documents you've uploaded. "
                                "Try asking e.g. *“Which agency had the highest net cost in FY2025?”*")
    return out


# ---------------------------------------------------------------------------
# Node: retriever (tool executor)
# ---------------------------------------------------------------------------
def retriever(state: AgentState) -> AgentState:
    plan = Plan(**state["plan"])
    store = get_store()
    contexts: list[dict] = []
    calls: list[str] = []

    def add(kind: str, title: str, content: str, source: str, score: float | None = None):
        contexts.append({"id": f"S{len(contexts) + 1}", "kind": kind, "title": title,
                         "content": content, "source": source, "score": score})

    avail = store.years
    years = [y for y in plan.fiscal_years if y in avail]
    missing = [y for y in plan.fiscal_years if y not in avail]
    if missing:
        add("structured", "Data availability",
            f"The dataset contains fiscal years {avail}. No data exists for FY{', FY'.join(map(str, missing))}.",
            "net_cost_unified.csv")

    agencies = plan.agencies
    if not store.empty:
        if plan.intent == "ranking":
            for y in (years or [max(avail)]):
                args = {"fiscal_year": y, "metric": plan.metric, "top_n": plan.top_n,
                        "ascending": plan.ascending}
                add("structured", f"Ranking FY{y}", T.rank_agencies.invoke(args), "net_cost_unified.csv")
                calls.append(f"rank_agencies({args})")
        if plan.intent in {"comparison", "trend"}:
            targets = agencies or ["Total"]
            ys = sorted(years) if len(years) >= 2 else avail
            pairs = list(zip(ys, ys[1:]))
            if plan.intent == "comparison" and len(ys) > 2 and len(years) >= 2:
                pairs.append((ys[0], ys[-1]))
            for a in targets[:6]:
                for ya, yb in pairs:
                    args = {"agency": a, "year_a": ya, "year_b": yb, "metric": plan.metric}
                    add("structured", f"{a}: FY{ya}→FY{yb}", T.compare_fiscal_years.invoke(args),
                        "net_cost_unified.csv")
                    calls.append(f"compare_fiscal_years({a}, {ya}, {yb})")
        if plan.intent == "restatement":
            for y in (years or [y for y in avail if y < max(avail)]):
                add("structured", f"Restatements FY{y}",
                    T.restatement_differences.invoke({"fiscal_year": y}), "net_cost_unified.csv")
                calls.append(f"restatement_differences({y})")
        if plan.intent in {"lookup", "comparison", "trend", "restatement"} or (agencies and plan.intent != "ranking"):
            look_agencies = agencies or (["Total"] if plan.intent in {"lookup", "comparison", "trend"} else [])
            if look_agencies:
                args = {"agencies": look_agencies[:8], "fiscal_years": years or None,
                        "include_restated": plan.include_restated or plan.intent == "restatement"}
                add("structured", "Financial figures", T.lookup_financials.invoke(args), "net_cost_unified.csv")
                calls.append(f"lookup_financials({look_agencies[:8]}, {years or 'all years'})")

    # Semantic search (always) — dedupe across sub-queries.
    seen = set()
    k_each = max(3, config.TOP_K // max(1, len(plan.search_queries[:3])))
    vec_budget = config.TOP_K if plan.intent in {"definition", "general"} else max(4, config.TOP_K // 2)
    n_vec = 0
    for q in plan.search_queries[:3]:
        try:
            hits = T.semantic_search.invoke({"query": q, "k": k_each})
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector search failed: %s", exc)
            hits = []
        calls.append(f"semantic_search('{q[:60]}', k={k_each})")
        for h in hits:
            cid = h["metadata"].get("chunk_id") or h["content"][:50]
            if cid in seen or n_vec >= vec_budget:
                continue
            seen.add(cid)
            n_vec += 1
            md = h["metadata"]
            title = md.get("doc_type", "chunk") + (f" · {md['agency']}" if md.get("agency") else "") + \
                (f" · FY{md['fiscal_year']}" if md.get("fiscal_year") else "") + \
                (f" · p.{md['page']}" if md.get("page") else "")
            add("vector", title, h["content"], md.get("source_file", "?"), h["score"])

    n_struct = sum(c["kind"] == "structured" for c in contexts)
    return {"contexts": contexts,
            "trace": [{"node": "retriever", "status": "done",
                       "detail": f"{n_struct} structured + {n_vec} vector context blocks",
                       "tool_calls": calls}]}


# ---------------------------------------------------------------------------
# Node: reasoner
# ---------------------------------------------------------------------------
REASONER_SYSTEM = """You are a meticulous U.S. federal financial analyst assistant.
Answer the user's question using ONLY the numbered context blocks below.

Rules:
1. Every factual sentence must end with citation(s) like [S1] or [S2][S4] that point to the context.
2. Copy numbers exactly as they appear in the context (amounts are in billions USD, e.g. $1,886.3B).
   Do NOT perform your own arithmetic — use the pre-computed changes/percentages from the context.
   You may additionally express large amounts in trillions (e.g. $1,886.3B ≈ $1.9 trillion).
3. If the context does not contain the answer, say clearly: "The available data does not contain
   that information." Never use outside knowledge or guess.
4. Default to originally reported figures; mention restated values only when relevant or asked.
5. Be concise: a direct answer first, then supporting detail (a small markdown table is welcome
   for multi-row data). Explain negative net cost or changes-in-assumptions effects when relevant.
6. Do not add a references list — sources are appended automatically.

Context:
{context}"""


def _format_context(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] ({c['kind']} · {c['title']} · source: {c['source']})\n{c['content']}"
                       for c in contexts) or "(no context retrieved)"


def reasoner(state: AgentState) -> AgentState:
    plan = state["plan"]
    contexts = state.get("contexts", [])
    msgs = [SystemMessage(REASONER_SYSTEM.format(context=_format_context(contexts)))]
    for h in state.get("history", [])[-4:]:
        msgs.append((HumanMessage if h["role"] == "user" else AIMessage)(h["content"][:1500]))
    msgs.append(HumanMessage(plan.get("standalone_question") or state["question"]))
    feedback = state.get("validation", {}).get("feedback")
    if feedback:
        msgs.append(HumanMessage(f"Your previous draft failed validation: {feedback} "
                                 "Rewrite the answer fixing these issues, strictly following the rules."))
    answer = _text(get_llm().invoke(msgs)).strip()
    attempt = state.get("retries", 0) + 1
    return {"answer": answer,
            "trace": [{"node": "reasoner", "status": "done",
                       "detail": f"Draft #{attempt} generated ({len(answer)} chars)"}]}


# ---------------------------------------------------------------------------
# Node: validator
# ---------------------------------------------------------------------------
NO_DATA_PHRASES = ("does not contain", "not available", "no data", "cannot find", "not in the")


def validator(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    context = _format_context(state.get("contexts", []))
    issues = []
    bad_nums = ungrounded_numbers(answer, context)
    if bad_nums:
        issues.append(f"these figures are not in the context: {', '.join(bad_nums[:8])}")
    is_no_data = any(p in answer.lower() for p in NO_DATA_PHRASES)
    if not has_citations(answer) and not is_no_data:
        issues.append("the answer has no [S#] citations")
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", answer)}
    invalid = [f"S{n}" for n in cited if n > len(state.get("contexts", []))]
    if invalid:
        issues.append(f"citations {invalid} do not exist")
    passed = not issues
    retries = state.get("retries", 0) + (0 if passed else 1)
    return {"validation": {"passed": passed, "issues": issues,
                           "feedback": "; ".join(issues) if issues else ""},
            "retries": retries,
            "trace": [{"node": "validator", "status": "passed" if passed else "failed",
                       "detail": "Grounding & citation checks passed" if passed else "; ".join(issues)}]}


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------
def finalize(state: AgentState) -> AgentState:
    if state.get("direct_answer"):
        return {"answer": state["direct_answer"], "sources": [],
                "trace": [{"node": "finalize", "status": "done", "detail": "Direct response"}]}
    answer = state.get("answer", "")
    val = state.get("validation", {})
    if val and not val.get("passed"):
        answer += ("\n\n> ⚠️ **Verification note:** the validator could not fully verify this answer "
                   f"against the source data ({val.get('feedback')}). Please check the cited sources.")
    cited = {f"S{n}" for n in re.findall(r"\[S(\d+)\]", answer)}
    sources = [c for c in state.get("contexts", []) if c["id"] in cited] or state.get("contexts", [])[:3]
    return {"answer": answer, "sources": sources,
            "trace": [{"node": "finalize", "status": "done",
                       "detail": f"{len(sources)} sources attached"}]}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
def _after_guard(state: AgentState) -> str:
    return "finalize" if state.get("blocked") else "planner"


def _after_planner(state: AgentState) -> str:
    return "finalize" if state.get("direct_answer") else "retriever"


def _after_validator(state: AgentState) -> str:
    if state["validation"]["passed"] or state.get("retries", 0) > config.MAX_VALIDATION_RETRIES:
        return "finalize"
    return "reasoner"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard)
    g.add_node("planner", planner)
    g.add_node("retriever", retriever)
    g.add_node("reasoner", reasoner)
    g.add_node("validator", validator)
    g.add_node("finalize", finalize)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", _after_guard, {"planner": "planner", "finalize": "finalize"})
    g.add_conditional_edges("planner", _after_planner, {"retriever": "retriever", "finalize": "finalize"})
    g.add_edge("retriever", "reasoner")
    g.add_edge("reasoner", "validator")
    g.add_conditional_edges("validator", _after_validator, {"reasoner": "reasoner", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(question: str, history: list[dict] | None = None) -> AgentState:
    return get_graph().invoke({"question": question, "history": history or [], "trace": []})


def stream_agent(question: str, history: list[dict] | None = None):
    """Yield (node_name, partial_state) as each agent finishes — used for live UI status."""
    yield from ((node, upd) for chunk in get_graph().stream(
        {"question": question, "history": history or [], "trace": []}, stream_mode="updates")
        for node, upd in chunk.items())
