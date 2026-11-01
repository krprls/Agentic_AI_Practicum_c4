# 🇺🇸 U.S. Government Financial Report — Multi-Modal Agentic RAG System

An autonomous **Multi-Agent Retrieval-Augmented Generation (RAG)** knowledge and decision-support system built with **LangGraph**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)**, **Google Gemini / Ollama**, and **Streamlit**.

Developed as an **Agentic AI Capstone Project**, this application ingests multi-format enterprise and federal financial documents (**XML, JSON, CSV, Excel, PDF, TXT, Markdown**) and answers natural-language questions with strict inline citations (`[S1]..[Sn]`) and deterministic numeric validation.

---

## ✨ Key Features

- **Multi-Format Document Ingestion (`src/ingestion/`)**:
  - Pre-loaded with the U.S. Treasury *Statements of Net Cost* across three heterogeneous formats:
    - `USFR_StmtNetCost_20230101_20231231.xml` (FY2023 + FY2022 restated)
    - `USFR_StmtNetCost_20240101_20241231.json` (FY2024 + FY2023 restated)
    - `USFR_StmtNetCost_20250101_20250930.csv` (FY2025 + FY2024 restated)
  - Supports user uploads of **PDF, TXT, MD, CSV, XLSX/XLS, JSON, and XML** via the Streamlit sidebar or CLI.
  - Normalizes financial statements into a unified 13-column dataset (`data/processed/net_cost_unified.csv`, `.xlsx`, and `.md`).
- **Multi-Granularity Semantic Chunking**:
  - Generates semantic chunks at 4 granularities (`record`, `agency_trend`, `year_summary`, `overview`) for financial data plus character-split chunks (`900` chars / `150` overlap) for free-text documents.
- **Local Open-Source Vector Store (`src/vectorstore.py`)**:
  - Uses `BAAI/bge-small-en-v1.5` locally via `sentence-transformers` and persists embeddings in **ChromaDB** (`chroma_store/`) with deterministic SHA-1 IDs for idempotent re-runs.
- **6-Node LangGraph Multi-Agent Workflow (`src/agents/graph.py`)**:
  - `input_guard` → `planner` → `retriever` → `reasoner` → `validator` → `finalize`
  - Combines dense vector search (`semantic_search`) with deterministic `pandas` tools (`lookup_financials`, `rank_agencies`, `compare_fiscal_years`, `restatement_differences`).
- **Reliability & Safety Guardrails (`src/guardrails.py`)**:
  - **Input Guard:** Blocks prompt injection, unsafe content, overly long inputs, and PII (SSNs, credit card numbers).
  - **Output Validator:** Deterministically verifies that every numeric figure in the LLM response exists in the retrieved context and every factual claim has a valid `[S#]` citation, triggering a self-correction retry loop if needed.
- **Interactive Streamlit UI (`app.py`) & CLI REPL (`main.py`)**:
  - Live streaming agent execution trace, expandable source citations, interactive **Plotly** financial explorer, and **Graphviz** architecture diagrams.

---

## 📁 Project Structure

```text
├── app.py                        # Main Streamlit multi-tab web application
├── streamlit_app.py              # Alias entrypoint for Streamlit
├── main.py                       # Interactive terminal REPL CLI
├── ingest.py                     # CLI script to build/refresh the knowledge base
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template (no secrets)
├── PRD.md                        # Product Requirements Document
├── docs/
│   └── ARCHITECTURE.md           # Detailed architecture, Mermaid diagrams, setup & limitations
├── src/
│   ├── config.py                 # Central configuration & environment variables
│   ├── llm.py                    # LLM factory (Gemini 2.5 Flash default + Ollama local OSS)
│   ├── data_store.py             # Unified financial store, fuzzy agency resolver & analytics
│   ├── vectorstore.py            # ChromaDB + local BAAI/bge-small-en-v1.5 embeddings
│   ├── guardrails.py             # Input safety checks + deterministic numeric grounding check
│   ├── diagrams.py               # Graphviz DOT diagrams rendered in the Streamlit UI
│   ├── ingestion/
│   │   ├── loaders.py            # Multi-format readers (XML, JSON, CSV, XLSX, PDF, TXT, MD)
│   │   ├── documents.py          # Multi-granularity semantic & text chunk builders
│   │   └── pipeline.py           # Idempotent end-to-end ingestion pipeline
│   └── agents/
│       ├── tools.py              # LangChain @tools (vector search + structured analytics)
│       └── graph.py              # LangGraph multi-agent state machine
└── tests/
    └── test_system.py            # Automated pytest suite
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set your `GEMINI_API_KEY` (free at https://aistudio.google.com/app/apikey), or set `LLM_PROVIDER=ollama` to run 100% locally with Ollama (`ollama pull llama3.1:8b`). You can also enter your API key directly in the Streamlit sidebar.

### 3. Ingest the Knowledge Base

```bash
python ingest.py --reset
```

Expected output:
```text
Ingesting 3 file(s)...
✅ OK:     ['USFR_StmtNetCost_20230101_20231231.xml', 'USFR_StmtNetCost_20240101_20241231.json', 'USFR_StmtNetCost_20250101_20250930.csv']
📊 Structured rows: 250
🧩 Chunks upserted: 295 {'overview': 1, 'record': 250, 'agency_trend': 40, 'year_summary': 4}
```

### 4. Launch the Streamlit Web App

```bash
streamlit run app.py
```

### 5. Run the Terminal CLI (Optional)

```bash
python main.py
```

### 6. Run Tests

```bash
pytest -v
```

---

## 📖 Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full architecture diagrams, agent specifications, data flow sequences, deployment steps, and a discussion of challenges and limitations.
