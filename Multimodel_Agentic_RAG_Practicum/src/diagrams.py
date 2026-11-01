"""Architecture & data-flow diagrams (Graphviz DOT) rendered in the Streamlit app.

Mermaid versions of the same diagrams live in docs/ARCHITECTURE.md.
"""

ARCHITECTURE_DOT = r"""
digraph G {
  rankdir=LR; bgcolor="transparent"; splines=spline; nodesep=0.35; ranksep=0.6;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#6C8CFF", fontcolor="#E6EAF5", fillcolor="#18223D", penwidth=1.4];
  edge [color="#8EA2D9", fontcolor="#B8C2E0", fontname="Helvetica", fontsize=9, penwidth=1.2];

  subgraph cluster_src { label="Data Sources"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    xml  [label="FY2023 XML\n(+FY2022 restated)"];
    json [label="FY2024 JSON\n(+FY2023 restated)"];
    csv  [label="FY2025 CSV\n(+FY2024 restated)"];
    up   [label="User uploads\nPDF · TXT · CSV · Excel\nJSON · XML"];
  }

  subgraph cluster_ing { label="Ingestion Layer"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    load [label="Multi-format Loaders\n(pandas · ElementTree · pypdf)"];
    norm [label="Schema Normaliser\n→ canonical columns"];
    chunk [label="Semantic Chunker\nrecord · agency_trend\nyear_summary · overview · text"];
  }

  subgraph cluster_store { label="Knowledge Stores"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    table [label="Structured Table\nnet_cost_unified.csv\n(+ .xlsx / .md readable)", fillcolor="#1D2B4F"];
    emb [label="Embeddings\nBAAI/bge-small-en-v1.5\n(local, open-source)"];
    chroma [label="ChromaDB\n(persistent vector DB)", shape=cylinder, fillcolor="#1D2B4F"];
  }

  subgraph cluster_agents { label="LangGraph Agentic Workflow"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    guard [label="🛡 Input Guard"];
    plan  [label="🧭 Planner Agent\n(LLM structured output)"];
    ret   [label="🔎 Retriever Agent\n(tool executor)"];
    reas  [label="🧠 Reasoner Agent\n(grounded generation)"];
    val   [label="✅ Validator Agent\n(numeric grounding)"];
  }

  llm [label="LLM\nGemini 2.5 Flash (dev)\nor Ollama Llama 3 (OSS)", shape=component, fillcolor="#2A1F4D"];
  ui  [label="Streamlit UI\nchat · upload · explorer", shape=box3d, fillcolor="#2A1F4D"];

  xml -> load; json -> load; csv -> load; up -> load;
  load -> norm -> table;
  norm -> chunk; load -> chunk [label="text / other tables"];
  chunk -> emb -> chroma;

  ui -> guard [label="question"];
  guard -> plan -> ret;
  ret -> chroma [label="semantic_search"];
  ret -> table [label="lookup · rank\ncompare · restatement"];
  ret -> reas -> val;
  val -> reas [label="retry", style=dashed, color="#E3B341"];
  val -> ui [label="answer + citations\n+ agent trace"];
  plan -> llm [style=dotted]; reas -> llm [style=dotted];
}
"""

DATAFLOW_DOT = r"""
digraph D {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10.5,
        color="#6C8CFF", fontcolor="#E6EAF5", fillcolor="#18223D"];
  edge [color="#8EA2D9", fontcolor="#B8C2E0", fontname="Helvetica", fontsize=9];

  subgraph cluster_a { label="① Offline / on-upload ingestion"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    a1 [label="Raw file (XML / JSON / CSV / XLSX / PDF / TXT)"];
    a2 [label="Validate type, size, name"];
    a3 [label="Parse → DataFrame / text pages"];
    a4 [label="Statement-of-Net-Cost schema?", shape=diamond, fillcolor="#2A1F4D"];
    a5 [label="Map fields → canonical schema\nround floats · tag restatements · source lineage"];
    a6 [label="Merge + de-duplicate → net_cost_unified.csv/.xlsx/.md"];
    a7 [label="Build semantic chunks\n(rows · agency trends · year summaries · glossary)"];
    a8 [label="Recursive character split\n(900 chars / 150 overlap)"];
    a9 [label="Embed (bge-small) → upsert ChromaDB\n(SHA1 ids ⇒ idempotent)"];
    a1 -> a2 -> a3 -> a4;
    a4 -> a5 [label="yes"]; a4 -> a8 [label="no"];
    a5 -> a6 -> a7 -> a9; a8 -> a9;
  }

  subgraph cluster_b { label="② Online question answering"; fontcolor="#E3B341"; color="#3A4466"; style="rounded,dashed";
    b1 [label="User question + chat history"];
    b2 [label="Input guard\n(length · injection · unsafe · PII)"];
    b3 [label="Planner: rewrite · intent · agencies · years · metric"];
    b4 [label="Retriever: structured tools + vector search\n→ numbered context [S1..Sn]"];
    b5 [label="Reasoner: answer from context only, cite [S#]"];
    b6 [label="Validator: numbers ∈ context? citations valid?", shape=diamond, fillcolor="#2A1F4D"];
    b7 [label="Finalize: answer + sources + trace → UI"];
    b1 -> b2 -> b3 -> b4 -> b5 -> b6;
    b6 -> b7 [label="pass / retries exhausted"];
    b6 -> b5 [label="fail → retry", style=dashed, color="#E3B341"];
  }
  a9 -> b4 [style=dotted, label="vector + table stores"];
}
"""
