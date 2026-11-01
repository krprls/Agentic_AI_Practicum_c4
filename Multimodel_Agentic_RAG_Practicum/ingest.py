"""Build / refresh the knowledge base.

Usage:
    python ingest.py                 # ingest data/raw (copies USFR_* files from project root on first run)
    python ingest.py --reset         # wipe the vector store + structured table and rebuild
    python ingest.py path/to/a.pdf path/to/b.xlsx   # ingest extra files
"""
from __future__ import annotations

import argparse
import logging

from src.ingestion.pipeline import bootstrap_raw_data, default_source_files, ingest_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="Optional extra files to ingest")
    ap.add_argument("--reset", action="store_true", help="Rebuild from scratch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    bootstrap_raw_data()
    paths = default_source_files() + list(args.files)
    print(f"Ingesting {len(paths)} file(s)...")
    report = ingest_paths(paths, reset=args.reset)
    print(f"\n✅ OK:     {report.files_ok}")
    if report.files_failed:
        print(f"❌ Failed: {report.files_failed}")
    print(f"📊 Structured rows: {report.net_cost_rows}")
    print(f"🧩 Chunks upserted: {report.total_chunks} {report.chunks_by_type}")


if __name__ == "__main__":
    main()
