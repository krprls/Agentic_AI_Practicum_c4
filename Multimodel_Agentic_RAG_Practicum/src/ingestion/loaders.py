"""Multi-format document loaders.

Supports: CSV, JSON, XML, Excel (tabular) and PDF, TXT, Markdown (text).

Tabular files that match the Fiscal Data *Statements of Net Cost* schema are
normalised into one canonical, analysis-ready schema regardless of whether
they arrived as CSV (human-readable headers), JSON or XML (API field names).
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

log = logging.getLogger(__name__)

TABULAR_EXT = {".csv", ".json", ".xml", ".xlsx", ".xls"}
TEXT_EXT = {".pdf", ".txt", ".md"}

# ---------------------------------------------------------------------------
# Canonical schema for Statements of Net Cost
# ---------------------------------------------------------------------------
# Maps both API field names (JSON/XML) and display headers (CSV) to canonical names.
FIELD_MAP: dict[str, str] = {
    "record_date": "record_date", "Record Date": "record_date",
    "stmt_fiscal_year": "statement_fiscal_year", "Statement Fiscal Year": "statement_fiscal_year",
    "restmt_flag": "restatement_flag", "Restatement Flag": "restatement_flag",
    "agency_nm": "agency_name", "Agency Name": "agency_name",
    "gross_cost_bil_amt": "gross_cost_bil", "Gross Cost (in Billions)": "gross_cost_bil",
    "earned_revenue_bil_amt": "earned_revenue_bil", "Earned Revenue (in Billions)": "earned_revenue_bil",
    "subtotal_bil_amt": "subtotal_bil", "Subtotal (in Billions)": "subtotal_bil",
    "change_assumptions_bil_amt": "change_in_assumptions_bil",
    "Change in Assumptions (in Billions)": "change_in_assumptions_bil",
    "net_cost_bil_amt": "net_cost_bil", "Net Cost (in Billions)": "net_cost_bil",
    "src_line_nbr": "source_line_number", "Source Line Number": "source_line_number",
    "record_fiscal_year": "report_fiscal_year", "Fiscal Year": "report_fiscal_year",
    "record_fiscal_quarter": "report_fiscal_quarter", "Fiscal Quarter Number": "report_fiscal_quarter",
}
CANONICAL_COLUMNS = [
    "record_date", "report_fiscal_year", "statement_fiscal_year", "restatement_flag",
    "agency_name", "gross_cost_bil", "earned_revenue_bil", "subtotal_bil",
    "change_in_assumptions_bil", "net_cost_bil", "source_line_number",
    "source_file", "source_format",
]
NUMERIC_COLUMNS = [
    "gross_cost_bil", "earned_revenue_bil", "subtotal_bil",
    "change_in_assumptions_bil", "net_cost_bil",
]
_REQUIRED = {"agency_name", "net_cost_bil", "statement_fiscal_year"}


@dataclass
class LoadedFile:
    """Result of loading one file."""

    path: Path
    fmt: str
    table: pd.DataFrame | None = None          # raw tabular content (if any)
    net_cost: pd.DataFrame | None = None       # canonical net-cost rows (if schema matched)
    text_docs: list[Document] = field(default_factory=list)  # raw text pages (pdf/txt)
    error: str | None = None


# ---------------------------------------------------------------------------
# Raw readers
# ---------------------------------------------------------------------------
def _read_xml(path: Path) -> pd.DataFrame:
    """Read 'record-like' XML: any element whose children are all leaf nodes is a row."""
    root = ET.parse(path).getroot()
    rows = []
    for el in root.iter():
        children = list(el)
        if len(children) >= 2 and all(len(list(c)) == 0 for c in children):
            rows.append({c.tag: (c.text or "").strip() for c in children})
    if not rows:
        raise ValueError("No record-like elements found in XML")
    return pd.DataFrame(rows)


def _read_json(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "rows", "results"):
            if isinstance(payload.get(key), list):
                return pd.json_normalize(payload[key])
        return pd.json_normalize(payload)
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    raise ValueError("Unsupported JSON structure")


def read_tabular(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, dtype=str)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    if ext == ".json":
        return _read_json(path)
    if ext == ".xml":
        return _read_xml(path)
    raise ValueError(f"Not a tabular format: {ext}")


def read_text(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={
                    "source_file": path.name, "source_format": "pdf", "page": i}))
        if not docs:
            raise ValueError("PDF contains no extractable text (scanned image PDFs need OCR)")
        return docs
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("File is empty")
    return [Document(page_content=text, metadata={"source_file": path.name, "source_format": ext[1:]})]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_net_cost(df: pd.DataFrame, source_file: str, source_format: str) -> pd.DataFrame | None:
    """Return canonical net-cost rows, or None if the table isn't a Statement of Net Cost."""
    renamed = df.rename(columns={c: FIELD_MAP[c] for c in df.columns if c in FIELD_MAP})
    if not _REQUIRED.issubset(renamed.columns):
        return None
    out = renamed.copy()
    for col in NUMERIC_COLUMNS:
        if col in out:
            # Round to 1 dp: the source has float artefacts like 1.39999999999999
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
        else:
            out[col] = pd.NA
    out["statement_fiscal_year"] = pd.to_numeric(out["statement_fiscal_year"], errors="coerce").astype("Int64")
    if "report_fiscal_year" in out:
        out["report_fiscal_year"] = pd.to_numeric(out["report_fiscal_year"], errors="coerce").astype("Int64")
    else:
        out["report_fiscal_year"] = out["statement_fiscal_year"]
    out["restatement_flag"] = out.get("restatement_flag", "N").fillna("N").str.upper().str.strip()
    out["agency_name"] = out["agency_name"].astype(str).str.strip()
    out["source_line_number"] = pd.to_numeric(out.get("source_line_number"), errors="coerce").astype("Int64")
    out["source_file"] = source_file
    out["source_format"] = source_format
    for col in CANONICAL_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    return out[CANONICAL_COLUMNS].dropna(subset=["agency_name", "statement_fiscal_year"])


def load_file(path: str | Path) -> LoadedFile:
    """Load any supported file into a LoadedFile (never raises; errors are captured)."""
    path = Path(path)
    ext = path.suffix.lower()
    result = LoadedFile(path=path, fmt=ext.lstrip("."))
    try:
        if ext in TABULAR_EXT:
            table = read_tabular(path)
            if table.empty:
                raise ValueError("Table is empty")
            result.table = table
            result.net_cost = normalize_net_cost(table, path.name, result.fmt)
        elif ext in TEXT_EXT:
            result.text_docs = read_text(path)
        else:
            raise ValueError(f"Unsupported file type '{ext}'")
    except Exception as exc:  # noqa: BLE001 – surface to UI instead of crashing
        log.exception("Failed to load %s", path)
        result.error = f"{type(exc).__name__}: {exc}"
    return result
