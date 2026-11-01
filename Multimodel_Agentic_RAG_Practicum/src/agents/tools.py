"""Agent tools.

Each tool is a LangChain ``@tool`` so it is self-describing and can be bound to
an LLM for function-calling. The Retriever agent invokes them deterministically
based on the Planner agent's structured plan, which is far more reliable for
numeric financial questions than free-form ReAct loops.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src import config
from src.data_store import METRIC_LABELS, METRICS, fmt_b, get_store


@tool
def semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Semantic similarity search over all ingested document chunks in the Chroma vector DB.

    Use for definitions, narrative/contextual questions, uploaded PDFs/TXT, or to find
    which agencies/years are relevant to a fuzzy question.
    """
    from src.vectorstore import get_vectorstore

    hits = get_vectorstore().similarity_search_with_relevance_scores(query, k=k)
    return [{"content": d.page_content, "metadata": d.metadata, "score": round(float(s), 3)}
            for d, s in hits]


@tool
def lookup_financials(agencies: list[str] | None = None, fiscal_years: list[int] | None = None,
                      include_restated: bool = False) -> str:
    """Exact Statement of Net Cost figures (gross cost, earned revenue, changes in assumptions,
    net cost) for the given agencies and fiscal years, from the unified structured table."""
    store = get_store()
    df = store.lookup(agencies or None, fiscal_years or None, include_restated)
    if df.empty:
        return "No matching rows in the structured table."
    return store.to_markdown(df)


@tool
def rank_agencies(fiscal_year: int, metric: str = "net_cost", top_n: int = 10,
                  ascending: bool = False) -> str:
    """Rank agencies for a fiscal year by a metric (net_cost, gross_cost, earned_revenue,
    subtotal, change_in_assumptions). ascending=True returns the lowest values."""
    store = get_store()
    col = METRICS.get(metric, "net_cost_bil")
    df = store.rank(fiscal_year, col, top_n, ascending)
    if df.empty:
        return f"No data for FY{fiscal_year}."
    order = "lowest" if ascending else "highest"
    rows = [f"{i}. {r.agency_name}: {fmt_b(getattr(r, col))}"
            for i, r in enumerate(df.itertuples(index=False), start=1)]
    total = store.lookup(["Total"], [fiscal_year])
    tot_line = (f"\nGovernment-wide total {METRIC_LABELS[col]} FY{fiscal_year}: "
                f"{fmt_b(total.iloc[0][col])}" if not total.empty else "")
    src = ", ".join(sorted(df["source_file"].unique()))
    return (f"Top {len(df)} agencies with the {order} {METRIC_LABELS[col]} in FY{fiscal_year} "
            f"(excludes totals/reconciling lines):\n" + "\n".join(rows) + tot_line + f"\nSource: {src}")


@tool
def compare_fiscal_years(agency: str, year_a: int, year_b: int, metric: str = "net_cost") -> str:
    """Pre-computed change and % change of a metric for one agency between two fiscal years."""
    store = get_store()
    col = METRICS.get(metric, "net_cost_bil")
    res = store.compare(agency, year_a, year_b, col)
    if not res:
        return f"Comparison unavailable for {agency} FY{year_a} vs FY{year_b} (missing data)."
    direction = "increased" if res["change"] > 0 else "decreased" if res["change"] < 0 else "was unchanged"
    pct = f" ({res['pct_change']:+.1f}%)" if res["pct_change"] is not None else ""
    return (f"{agency} {METRIC_LABELS[col]}: FY{year_a} {fmt_b(res['value_a'])} → FY{year_b} "
            f"{fmt_b(res['value_b'])}; {direction} by {fmt_b(abs(res['change']))}{pct}. "
            f"Sources: {', '.join(res['sources'])}")


@tool
def restatement_differences(fiscal_year: int) -> str:
    """Agencies whose net cost for a fiscal year was revised between the original report and
    the following year's restatement."""
    store = get_store()
    df = store.restatement_diff(fiscal_year)
    if df.empty:
        return f"No restatement differences found for FY{fiscal_year} (or no original + restated pair)."
    lines = [f"- {agency}: original {fmt_b(r.original)}, restated {fmt_b(r.restated)}, "
             f"revision {fmt_b(r.difference)}" for agency, r in df.iterrows()]
    return f"FY{fiscal_year} net cost restatements (original vs restated):\n" + "\n".join(lines)


ALL_TOOLS = [semantic_search, lookup_financials, rank_agencies, compare_fiscal_years,
             restatement_differences]
