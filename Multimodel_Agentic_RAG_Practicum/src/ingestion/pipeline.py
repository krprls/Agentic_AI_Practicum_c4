"""End-to-end ingestion: files → structured table + readable report + vector store."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.data_store import METRIC_LABELS, FinancialDataStore, fmt_b, reload_store
from src.ingestion import documents as D
from src.ingestion.loaders import CANONICAL_COLUMNS, load_file

log = logging.getLogger(__name__)
DEDUP_KEYS = ["agency_name", "statement_fiscal_year", "restatement_flag", "report_fiscal_year"]


@dataclass
class IngestReport:
    files_ok: list[str] = field(default_factory=list)
    files_failed: dict[str, str] = field(default_factory=dict)
    net_cost_rows: int = 0
    chunks_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_by_type.values())


def _write_readable_report(df: pd.DataFrame) -> None:
    """Human-readable Markdown + Excel export of the unified table."""
    store = FinancialDataStore(df)
    lines = [f"# {config.DATASET_NAME}", "",
             f"Source: {config.DATASET_URL}  ", "All amounts in **billions of USD**. "
             "Fiscal year ends **September 30**.", ""]
    for year in sorted(df["statement_fiscal_year"].unique(), reverse=True):
        for flag, label in (("N", "Originally reported"), ("Y", "Restated")):
            part = df[(df["statement_fiscal_year"] == year) & (df["restatement_flag"] == flag)]
            if part.empty:
                continue
            src = ", ".join(sorted(part["source_file"].unique()))
            lines += [f"## FY{year} — {label}", f"_Source file(s): {src}_", "",
                      store.to_markdown(part.sort_values("source_line_number")), ""]
    config.UNIFIED_MD.write_text("\n".join(lines), encoding="utf-8")

    xlsx = config.PROCESSED_DIR / "net_cost_unified.xlsx"
    pretty = df.rename(columns={**METRIC_LABELS, "agency_name": "Agency",
                                "statement_fiscal_year": "Statement FY",
                                "report_fiscal_year": "Report FY",
                                "restatement_flag": "Restated (Y/N)"})
    pretty.to_excel(xlsx, index=False, sheet_name="NetCost")


def ingest_paths(paths: list[str | Path], reset: bool = False) -> IngestReport:
    """Ingest files into the structured store and the vector store (idempotent)."""
    from src.vectorstore import get_vectorstore, reset_collection

    report = IngestReport()
    if reset:
        reset_collection()

    loaded_files = [load_file(p) for p in paths]
    new_frames, other_docs = [], []
    for lf in loaded_files:
        if lf.error:
            report.files_failed[lf.path.name] = lf.error
            continue
        report.files_ok.append(lf.path.name)
        if lf.net_cost is not None:
            new_frames.append(lf.net_cost)
        elif lf.table is not None:
            other_docs += D.generic_table_documents(lf)
        else:
            other_docs += D.text_documents(lf)

    # ---- structured table (merge with existing unless reset) ----
    frames = list(new_frames)
    if not reset and config.UNIFIED_CSV.exists():
        frames.insert(0, pd.read_csv(config.UNIFIED_CSV))
    if frames:
        unified = (pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]
                   .drop_duplicates(DEDUP_KEYS, keep="last")
                   .sort_values(["statement_fiscal_year", "restatement_flag", "source_line_number"],
                                ascending=[False, True, True]))
        unified.to_csv(config.UNIFIED_CSV, index=False)
        _write_readable_report(unified)
    store = reload_store()
    report.net_cost_rows = len(store.df)

    # ---- vector documents ----
    vs = get_vectorstore()
    docs = list(other_docs)
    if not store.empty and (new_frames or reset):
        # Derived summaries change when data changes → replace them wholesale.
        try:
            vs._collection.delete(where={"doc_type": {"$in": ["agency_trend", "year_summary"]}})
        except Exception:  # noqa: BLE001 – empty collection
            pass
        docs += D.overview_documents()
        docs += D.record_documents(store.df)
        docs += D.agency_trend_documents(store)
        docs += D.year_summary_documents(store)

    if docs:
        unique = {d.id: d for d in docs}           # de-dupe within batch
        batch = list(unique.values())
        for i in range(0, len(batch), 128):
            chunk = batch[i:i + 128]
            vs.add_documents(chunk, ids=[d.id for d in chunk])   # upsert → idempotent
        for d in batch:
            t = d.metadata.get("doc_type", "other")
            report.chunks_by_type[t] = report.chunks_by_type.get(t, 0) + 1
    log.info("Ingest complete: %s", report)
    return report


def default_source_files() -> list[Path]:
    """Raw dataset files shipped with the project (data/raw)."""
    return sorted(p for p in config.RAW_DATA_DIR.iterdir()
                  if p.suffix.lower() in config.SUPPORTED_EXTENSIONS)


def save_upload(name: str, content: bytes) -> Path:
    """Persist an uploaded file safely (sanitised name, size limit)."""
    safe = Path(name).name.replace(" ", "_")
    ext = Path(safe).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: "
                         f"{', '.join(sorted(config.SUPPORTED_EXTENSIONS))}")
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {config.MAX_UPLOAD_MB} MB limit")
    if not content:
        raise ValueError("File is empty")
    dest = config.UPLOAD_DIR / safe
    dest.write_bytes(content)
    return dest


def bootstrap_raw_data(project_root: Path = config.PROJECT_ROOT) -> None:
    """Copy the USFR_* source files from the project root into data/raw (first run)."""
    for p in project_root.glob("USFR_*"):
        target = config.RAW_DATA_DIR / p.name
        if not target.exists():
            shutil.copy2(p, target)


__all__ = ["ingest_paths", "default_source_files", "save_upload", "bootstrap_raw_data", "fmt_b"]
