"""Turn loaded files into embedding-ready LangChain Documents (chunks).

Chunking strategy
-----------------
Tabular financial data is NOT split by characters (that would cut rows in
half and separate numbers from their labels). Instead we build *semantic
chunks* at four granularities:

1. ``record``         – one natural-language sentence block per table row
2. ``agency_trend``   – one chunk per agency spanning all fiscal years (with YoY deltas)
3. ``year_summary``   – one chunk per fiscal year (totals, top/bottom agencies)
4. ``overview``       – dataset description + glossary of accounting terms

Free text (PDF/TXT/MD) and non-USFR tables use RecursiveCharacterTextSplitter.
"""
from __future__ import annotations

import hashlib

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.data_store import NON_AGENCY_ROWS, FinancialDataStore, fmt_b
from src.ingestion.loaders import LoadedFile

OVERVIEW_TEXT = f"""Dataset: {config.DATASET_NAME}
Source: U.S. Department of the Treasury, Fiscal Data ({config.DATASET_URL}).

The Financial Report of the U.S. Government presents the federal government's financial
position and condition, including its financial activity and results, for each fiscal year.
It provides a comprehensive view of the federal government's finances, accounting for its
revenues and costs, assets and liabilities, and other obligations and commitments. The
Financial Report dataset has a table for each of the consolidated financial statements of the
U.S. government, prepared using GAAP (Generally Accepted Accounting Principles). Each year's
Financial Report also includes restatements of the previous fiscal year's figures, enabling
users to connect prior-year ending balances with current-year starting balances. The Financial
Report is produced by Treasury in coordination with OMB (Office of Management and Budget),
part of the Executive Office of the President.

The Statement of Net Cost shows how much it costs to operate each federal agency/entity.
The U.S. federal fiscal year runs from October 1 to September 30 (e.g. FY2025 = Oct 1 2024 – Sep 30 2025).
All amounts are in billions of U.S. dollars.

Glossary:
- Gross Cost: total cost incurred by the entity (program costs, salaries, benefits, interest, etc.).
- Earned Revenue: exchange revenue the entity earns from the public or other entities for goods/services
  (e.g. postage for USPS, premiums, fees). It is subtracted from gross cost.
- Subtotal: Gross Cost minus Earned Revenue.
- (Gain)/Loss from Changes in Assumptions: actuarial gains or losses from changes in assumptions used to
  estimate long-term liabilities such as federal employee & veteran pensions and benefits (mostly VA,
  OPM and DoD). A positive number increases net cost; a negative number decreases it.
- Net Cost: Subtotal plus (Gain)/Loss from Changes in Assumptions. Negative net cost means earned revenue
  and gains exceeded costs.
- Restatement Flag: 'N' = originally reported figure for that fiscal year; 'Y' = figure restated in the
  following year's report (prior-year comparative).
- Interest on Treasury Securities held by the public: interest cost on federal debt, reported as its own line.
"""


def _doc_id(text: str, meta: dict) -> str:
    raw = text + "|" + "|".join(f"{k}={meta[k]}" for k in sorted(meta))
    return hashlib.sha1(raw.encode()).hexdigest()


def _with_id(doc: Document) -> Document:
    doc.id = _doc_id(doc.page_content, doc.metadata)
    doc.metadata["chunk_id"] = doc.id[:12]
    return doc


# ---------------------------------------------------------------------------
# Financial table → semantic chunks
# ---------------------------------------------------------------------------
def record_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for r in df.itertuples(index=False):
        status = ("restated figure (published in the FY%d report)" % r.report_fiscal_year
                  if r.restatement_flag == "Y" else "originally reported figure")
        text = (
            f"{r.agency_name} — Statement of Net Cost, fiscal year {r.statement_fiscal_year} "
            f"({status}; record date {r.record_date}).\n"
            f"Gross cost: {fmt_b(r.gross_cost_bil)}. Earned revenue: {fmt_b(r.earned_revenue_bil)}. "
            f"Subtotal (gross cost less earned revenue): {fmt_b(r.subtotal_bil)}. "
            f"(Gain)/loss from changes in assumptions: {fmt_b(r.change_in_assumptions_bil)}. "
            f"Net cost: {fmt_b(r.net_cost_bil)}.\n"
            f"Source: {r.source_file} ({r.source_format.upper()}), line {r.source_line_number}."
        )
        docs.append(_with_id(Document(page_content=text, metadata={
            "doc_type": "record", "agency": r.agency_name,
            "fiscal_year": int(r.statement_fiscal_year), "restated": r.restatement_flag == "Y",
            "source_file": r.source_file, "source_format": r.source_format,
        })))
    return docs


def agency_trend_documents(store: FinancialDataStore) -> list[Document]:
    docs = []
    pref = store.preferred()
    for agency, g in pref.groupby("agency_name"):
        g = g.sort_values("statement_fiscal_year")
        lines = [f"{agency} — multi-year Statement of Net Cost trend "
                 f"(originally reported figures where available; all amounts in billions USD)."]
        prev = None
        for r in g.itertuples(index=False):
            tag = " [restated]" if r.restatement_flag == "Y" else ""
            line = (f"FY{r.statement_fiscal_year}{tag}: net cost {fmt_b(r.net_cost_bil)}, "
                    f"gross cost {fmt_b(r.gross_cost_bil)}, earned revenue {fmt_b(r.earned_revenue_bil)}, "
                    f"changes in assumptions {fmt_b(r.change_in_assumptions_bil)}")
            if prev is not None and pd.notna(prev) and pd.notna(r.net_cost_bil):
                delta = round(r.net_cost_bil - prev, 1)
                pct = f", {delta / abs(prev) * 100:+.1f}%" if prev else ""
                line += f" (change in net cost vs prior year: {fmt_b(delta)}{pct})"
            lines.append(line + f". Source: {r.source_file}.")
            prev = r.net_cost_bil
        text = "\n".join(lines)
        docs.append(_with_id(Document(page_content=text, metadata={
            "doc_type": "agency_trend", "agency": agency,
            "source_file": ", ".join(sorted(g["source_file"].unique())), "source_format": "derived",
        })))
    return docs


def year_summary_documents(store: FinancialDataStore) -> list[Document]:
    docs = []
    pref = store.preferred()
    for year, g in pref.groupby("statement_fiscal_year"):
        agencies = g[~g["agency_name"].str.lower().isin(NON_AGENCY_ROWS)]
        total = g[g["agency_name"].str.lower() == "total"]
        top = agencies.sort_values("net_cost_bil", ascending=False).head(10)
        neg = agencies[agencies["net_cost_bil"] < 0].sort_values("net_cost_bil")
        lines = [f"Fiscal year {year} — U.S. Government Statement of Net Cost summary."]
        if not total.empty:
            t = total.iloc[0]
            lines.append(f"Government-wide total: gross cost {fmt_b(t.gross_cost_bil)}, earned revenue "
                         f"{fmt_b(t.earned_revenue_bil)}, changes in assumptions "
                         f"{fmt_b(t.change_in_assumptions_bil)}, total net cost {fmt_b(t.net_cost_bil)}.")
        lines.append(f"Top {len(top)} entities by net cost in FY{year}:")
        lines += [f"  {i}. {r.agency_name}: {fmt_b(r.net_cost_bil)}"
                  for i, r in enumerate(top.itertuples(index=False), start=1)]
        if not neg.empty:
            lines.append(f"Entities with negative net cost in FY{year}: " + "; ".join(
                f"{r.agency_name} ({fmt_b(r.net_cost_bil)})" for r in neg.itertuples(index=False)))
        lines.append("Sources: " + ", ".join(sorted(g["source_file"].unique())))
        docs.append(_with_id(Document(page_content="\n".join(lines), metadata={
            "doc_type": "year_summary", "fiscal_year": int(year),
            "source_file": ", ".join(sorted(g["source_file"].unique())), "source_format": "derived",
        })))
    return docs


def overview_documents() -> list[Document]:
    return [_with_id(Document(page_content=OVERVIEW_TEXT, metadata={
        "doc_type": "overview", "source_file": "fiscaldata.treasury.gov", "source_format": "text"}))]


# ---------------------------------------------------------------------------
# Generic content → character chunks
# ---------------------------------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def text_documents(loaded: LoadedFile) -> list[Document]:
    chunks = _splitter.split_documents(loaded.text_docs)
    for c in chunks:
        c.metadata["doc_type"] = "text"
    return [_with_id(c) for c in chunks]


def generic_table_documents(loaded: LoadedFile, rows_per_chunk: int = 15) -> list[Document]:
    """Non-USFR tables: serialise rows as 'col: value' lines, grouped into chunks."""
    df = loaded.table.fillna("")
    docs = []
    cols = list(df.columns)
    for start in range(0, len(df), rows_per_chunk):
        block = df.iloc[start:start + rows_per_chunk]
        lines = [f"Table from {loaded.path.name}, rows {start + 1}-{start + len(block)}:"]
        for i, row in block.iterrows():
            lines.append(f"Row {i + 1}: " + "; ".join(f"{c}: {row[c]}" for c in cols if str(row[c]) != ""))
        docs.append(_with_id(Document(page_content="\n".join(lines), metadata={
            "doc_type": "table_rows", "source_file": loaded.path.name, "source_format": loaded.fmt})))
    return docs
