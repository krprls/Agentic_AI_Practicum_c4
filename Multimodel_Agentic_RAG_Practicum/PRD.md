# Product Requirements Document (PRD) — USFR Multi-Modal Agentic RAG System

**Date:** 2026-10-31  
**Author:** Rama Kotipatruni  
**Project:** Agentic AI Capstone Project — Multi-Modal Agentic RAG System

---

## 1. Overview

The **U.S. Government Financial Report (USFR) Multi-Modal Agentic RAG System** is an AI agent–powered knowledge and decision-support application that enables analysts, auditors, policymakers, and citizens to query multi-format enterprise and federal financial documents (**XML, JSON, CSV, Excel, PDF, TXT, Markdown**) using natural language.

Built on **LangGraph**, **LangChain**, **ChromaDB**, local **HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)**, and **Google Gemini / Ollama**, the system orchestrates specialized AI agents to validate user inputs, plan retrieval strategies, execute hybrid tools (vector similarity search + deterministic `pandas` financial analytics), synthesize cited answers, and deterministically validate every numeric figure before returning the final output.

---

## 2. Problem Statement

Enterprise and government financial reporting is distributed across incompatible file formats (`XML`, `JSON`, `CSV`, `Excel`, `PDF`), spans multiple fiscal years with prior-year restatements, and contains dense accounting terminology:
- **Schema Fragmentation:** The FY2023 USFR Statement of Net Cost is published in XML, FY2024 in JSON, and FY2025 in CSV with different column naming conventions.
- **Restatement Ambiguity:** Each annual report includes restated prior-year figures alongside current-year figures, causing naive RAG pipelines to confuse or double-count numbers.
- **LLM Arithmetic Hallucinations:** Standard RAG systems split tabular rows across character chunks and rely on the LLM to perform mental arithmetic, leading to inaccurate sums, rankings, and percentage changes.
- **Lack of Verification & Auditability:** Decision-support users require traceable citations (`[S1]..[Sn]`) and verifiable numeric grounding.

---

## 3. Goals & Success Metrics

| Goal | Target Metric |
|---|---|
| **Multi-Format Ingestion** | Seamlessly ingest XML, JSON, CSV, Excel (`.xlsx`/`.xls`), PDF, TXT, and MD files with 0 unhandled exceptions |
| **Numeric Grounding** | 100% of financial figures in verified responses match the source dataset within rounding tolerance (`±0.05B`) |
| **Citation Completeness** | 100% of factual responses include valid inline `[S#]` citations mapped to expandable source blocks |
| **Safety & Guardrails** | 100% of prompt-injection, unsafe content, and PII (SSN/credit card) test cases blocked by `input_guard` |
| **Interactive UX** | End-to-end streaming agent trace visible in real time via Streamlit UI and CLI REPL |

---

## 4. Functional Requirements

### 4.1 User Interaction Layer (`app.py`, `main.py`)
- **FR-01:** Users can ask natural-language questions in a conversational Streamlit interface with multi-turn session memory.
- **FR-02:** Users can upload custom documents (`PDF`, `TXT`, `MD`, `CSV`, `XLSX`, `XLS`, `JSON`, `XML`) up to 25 MB via the sidebar and immediately query them.
- **FR-03:** Every assistant response displays expandable **Grounded Sources & Citations** and a step-by-step **Agent Workflow Trace**.
- **FR-04:** Users can explore interactive Plotly charts (Top-N agency rankings, multi-year trends) and download unified datasets (`.csv`, `.xlsx`, `.md`) in the **Financial Data Explorer** tab.
- **FR-05:** A terminal REPL (`main.py`) supports interactive CLI querying.

### 4.2 Multi-Format Ingestion & Semantic Chunking (`src/ingestion/`)
- **FR-06:** Automatically normalize `XML`, `JSON`, and `CSV` Statements of Net Cost into a 13-column canonical table (`net_cost_unified.csv`), rounding floating-point artifacts to 1 decimal place.
- **FR-07:** Construct multi-granularity semantic chunks (`record`, `agency_trend`, `year_summary`, `overview`) for financial tables and `RecursiveCharacterTextSplitter` chunks (`900` chars, `150` overlap) for free-text documents.
- **FR-08:** Generate local embeddings with `BAAI/bge-small-en-v1.5` and upsert idempotently into persistent ChromaDB (`chroma_store/`) using SHA-1 content hashes.

### 4.3 Multi-Agent Reasoning & Hybrid Retrieval (`src/agents/`)
- **FR-09:** **Input Guard Node:** Validate length, block prompt injections, filter unsafe requests, and redact/reject PII before any LLM call.
- **FR-10:** **Planner Agent:** Classify intent (`lookup`, `ranking`, `comparison`, `trend`, `restatement`, `definition`, `general`, `greeting`, `out_of_scope`), rewrite follow-ups using chat history, resolve agency aliases (e.g., `DoD` → `Department of Defense`), and fallback to a deterministic heuristic planner if the LLM fails.
- **FR-11:** **Retriever Agent:** Execute deterministic structured tools (`lookup_financials`, `rank_agencies`, `compare_fiscal_years`, `restatement_differences`) alongside `semantic_search`.
- **FR-12:** **Reasoner Agent:** Synthesize answers strictly from numbered context blocks `[S1]..[Sn]` at `temperature=0`.
- **FR-13:** **Validator Agent:** Deterministically verify all numbers (`ungrounded_numbers`) and citations (`has_citations`), routing back to the Reasoner for self-correction if any issue is found.

---

## 5. Non-Functional Requirements

- **NFR-01 (Privacy & Local Embeddings):** All vector embeddings (`BAAI/bge-small-en-v1.5`) run locally on CPU with zero external embedding API cost.
- **NFR-02 (Open-Source Optionality):** Supports both Google Gemini (`gemini-2.5-flash`) for cloud development and Ollama (`llama3.1:8b`) for a 100% local, open-source deployment.
- **NFR-03 (Idempotency & Persistence):** Both `chroma_store/` and `data/processed/net_cost_unified.csv` persist to disk and support idempotent re-ingestion.
- **NFR-04 (Security):** No API keys are hardcoded; secrets load from `.env` or runtime UI inputs. Uploaded filenames are sanitized and size-capped at 25 MB.
