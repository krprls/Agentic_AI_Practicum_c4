"""Generate the Capstone Project Technical Documentation PDF using pure Python (PDF 1.4).

Produces:
  - Capstone_Project_Documentation.pdf (project root)
  - docs/Capstone_Project_Documentation.pdf
"""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

PAGE_W = 612.0   # 8.5 in * 72 pt
PAGE_H = 792.0   # 11.0 in * 72 pt
MARGIN_L = 46.0
MARGIN_R = 46.0
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
TOP_Y = PAGE_H - 58.0
BOTTOM_Y = 52.0


def _esc(text: str) -> str:
    """Escape a string for PDF literal text (...) in WinAnsiEncoding."""
    replacements = {
        "—": "-",
        "–": "-",
        "→": "->",
        "↔": "<->",
        "•": "\x95",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "≈": "~",
        "≤": "<=",
        "≥": ">=",
        "±": "+/-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "(":
            out.append("\\(")
        elif ch == ")":
            out.append("\\)")
        elif 32 <= code <= 126 or code == 0x95:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


class PDFDocument:
    """Minimal, dependency-free multi-page PDF 1.4 writer with vector graphics and tables."""

    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.y = TOP_Y
        self.new_page()

    @property
    def cmd(self) -> list[str]:
        return self.pages[-1]

    def new_page(self) -> None:
        self.pages.append([])
        self.y = TOP_Y
        # Subtle top header bar on pages >= 2
        if len(self.pages) > 1:
            self.fill_rect(0, PAGE_H - 28, PAGE_W, 28, (0.07, 0.11, 0.22))
            self.text(
                MARGIN_L,
                PAGE_H - 18,
                "USFR Multi-Modal Agentic RAG System  |  Architecture, Setup, Agent Roles & Deployment",
                font="F2",
                size=8.5,
                color=(0.82, 0.87, 0.96),
            )
            self.y = PAGE_H - 50.0

    def ensure_space(self, needed: float) -> None:
        if self.y - needed < BOTTOM_Y:
            self.new_page()

    def fill_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill_rgb: tuple[float, float, float],
        stroke_rgb: tuple[float, float, float] | None = None,
        line_w: float = 0.8,
    ) -> None:
        r, g, b = fill_rgb
        self.cmd.append(f"q {r:.3f} {g:.3f} {b:.3f} rg")
        if stroke_rgb:
            sr, sg, sb = stroke_rgb
            self.cmd.append(f"{sr:.3f} {sg:.3f} {sb:.3f} RG {line_w:.2f} w")
            self.cmd.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re B Q")
        else:
            self.cmd.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f Q")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb: tuple[float, float, float] = (0.75, 0.78, 0.86),
        width: float = 0.7,
    ) -> None:
        r, g, b = rgb
        self.cmd.append(f"q {r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S Q")

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb: tuple[float, float, float] = (0.25, 0.38, 0.72),
        width: float = 1.1,
    ) -> None:
        import math

        self.line(x1, y1, x2, y2, rgb=rgb, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 5.0
        a1 = angle + math.pi * 0.82
        a2 = angle - math.pi * 0.82
        px1, py1 = x2 + head * math.cos(a1), y2 + head * math.sin(a1)
        px2, py2 = x2 + head * math.cos(a2), y2 + head * math.sin(a2)
        r, g, b = rgb
        self.cmd.append(
            f"q {r:.3f} {g:.3f} {b:.3f} rg {x2:.2f} {y2:.2f} m {px1:.2f} {py1:.2f} l {px2:.2f} {py2:.2f} l h f Q"
        )

    def text(
        self,
        x: float,
        y: float,
        txt: str,
        font: str = "F1",
        size: float = 10.0,
        color: tuple[float, float, float] = (0.12, 0.15, 0.22),
    ) -> None:
        r, g, b = color
        self.cmd.append(f"BT /{font} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({_esc(txt)}) Tj ET")

    # ------------------------------------------------------------------ high-level layout
    def title_banner(self) -> None:
        banner_h = 96.0
        by = PAGE_H - banner_h
        self.fill_rect(0, by, PAGE_W, banner_h, (0.06, 0.10, 0.22))
        self.fill_rect(0, by, PAGE_W, 4, (0.35, 0.52, 0.96))
        self.text(
            MARGIN_L,
            PAGE_H - 32,
            "AGENTIC AI  -  TECHNICAL ARCHITECTURE & DOCUMENTATION",
            font="F2",
            size=9.0,
            color=(0.56, 0.72, 1.0),
        )
        self.text(
            MARGIN_L,
            PAGE_H - 54,
            "U.S. Government Financial Report - Multi-Modal Agentic RAG",
            font="F2",
            size=17.5,
            color=(1.0, 1.0, 1.0),
        )
        self.text(
            MARGIN_L,
            PAGE_H - 72,
            "System Setup, Architecture, Autonomous Agent Roles, Deployment Steps, Challenges & Limitations",
            font="F1",
            size=10.2,
            color=(0.82, 0.87, 0.95),
        )
        self.text(
            MARGIN_L,
            PAGE_H - 87,
            "Author: Rama Kotipatruni   |   Dataset: U.S. Treasury Statements of Net Cost (FY2022-FY2025: XML, JSON, CSV)",
            font="F3",
            size=8.6,
            color=(0.68, 0.76, 0.90),
        )
        self.y = by - 20.0

    def section(self, num: str, title: str) -> None:
        self.ensure_space(42)
        self.y -= 6
        self.fill_rect(MARGIN_L, self.y - 16, CONTENT_W, 22, (0.92, 0.95, 1.0), stroke_rgb=(0.75, 0.83, 0.96))
        self.fill_rect(MARGIN_L, self.y - 16, 4.5, 22, (0.20, 0.40, 0.85))
        self.text(MARGIN_L + 11, self.y - 9, f"{num}. {title}", font="F2", size=12.0, color=(0.08, 0.16, 0.36))
        self.y -= 28

    def subsection(self, title: str) -> None:
        self.ensure_space(26)
        self.y -= 3
        self.text(MARGIN_L, self.y - 8, title, font="F2", size=10.5, color=(0.14, 0.26, 0.55))
        self.line(MARGIN_L, self.y - 11, MARGIN_L + CONTENT_W, self.y - 11, rgb=(0.84, 0.88, 0.95), width=0.6)
        self.y -= 21

    def paragraph(self, txt: str, size: float = 9.5, leading: float = 13.2, indent: float = 0.0) -> None:
        max_chars = int((CONTENT_W - indent) / (size * 0.50))
        lines = textwrap.wrap(txt, width=max_chars)
        self.ensure_space(len(lines) * leading + 6)
        for ln in lines:
            self.text(MARGIN_L + indent, self.y, ln, font="F1", size=size, color=(0.14, 0.17, 0.23))
            self.y -= leading
        self.y -= 3

    def bullet(self, lead_bold: str, body: str, size: float = 9.3, leading: float = 12.8) -> None:
        full = f"{lead_bold} {body}" if lead_bold else body
        max_chars = int((CONTENT_W - 16) / (size * 0.50))
        lines = textwrap.wrap(full, width=max_chars)
        self.ensure_space(len(lines) * leading + 3)
        self.text(MARGIN_L + 4, self.y, "\x95", font="F2", size=size, color=(0.20, 0.40, 0.85))
        for i, ln in enumerate(lines):
            if i == 0 and lead_bold and ln.startswith(lead_bold):
                rest = ln[len(lead_bold):]
                self.cmd.append(
                    f"BT /F2 {size:.2f} Tf 0.100 0.140 0.220 rg {MARGIN_L + 15:.2f} {self.y:.2f} Td "
                    f"({_esc(lead_bold)}) Tj /F1 {size:.2f} Tf 0.150 0.180 0.240 rg ({_esc(rest)}) Tj ET"
                )
            else:
                self.text(MARGIN_L + 15, self.y, ln, font="F1", size=size, color=(0.15, 0.18, 0.24))
            self.y -= leading
        self.y -= 1.5

    def code_block(self, lines: list[str], title: str | None = None) -> None:
        line_h = 11.2
        header_h = 15.0 if title else 0.0
        box_h = len(lines) * line_h + 12.0 + header_h
        self.ensure_space(box_h + 8)
        top = self.y
        self.fill_rect(
            MARGIN_L,
            top - box_h,
            CONTENT_W,
            box_h,
            (0.96, 0.97, 0.99),
            stroke_rgb=(0.80, 0.84, 0.92),
            line_w=0.7,
        )
        cur_y = top - 11.0
        if title:
            self.fill_rect(MARGIN_L, top - header_h, CONTENT_W, header_h, (0.89, 0.92, 0.98))
            self.text(MARGIN_L + 8, top - 10.5, title, font="F2", size=8.3, color=(0.16, 0.24, 0.44))
            cur_y = top - header_h - 11.0
        for ln in lines:
            self.text(MARGIN_L + 8, cur_y, ln, font="F4", size=8.2, color=(0.12, 0.16, 0.26))
            cur_y -= line_h
        self.y = top - box_h - 8.0

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        col_Fractions: list[float],
        font_size: float = 8.4,
    ) -> None:
        col_widths = [f * CONTENT_W for f in col_Fractions]
        line_h = 11.0

        # Pre-wrap cells
        wrapped_rows: list[list[list[str]]] = []
        row_heights: list[float] = []
        for r in rows:
            w_cells = []
            max_lines = 1
            for cell, cw in zip(r, col_widths):
                chars = max(10, int((cw - 10) / (font_size * 0.49)))
                lines = textwrap.wrap(cell, width=chars) or [""]
                w_cells.append(lines)
                max_lines = max(max_lines, len(lines))
            wrapped_rows.append(w_cells)
            row_heights.append(max_lines * line_h + 7.0)

        header_h = 18.0
        self.ensure_space(header_h + (row_heights[0] if row_heights else 20) + 10)

        def draw_header(y_top: float) -> float:
            self.fill_rect(MARGIN_L, y_top - header_h, CONTENT_W, header_h, (0.12, 0.21, 0.42))
            cx = MARGIN_L
            for h_txt, cw in zip(headers, col_widths):
                self.text(cx + 5, y_top - 12.2, h_txt, font="F2", size=font_size + 0.3, color=(1.0, 1.0, 1.0))
                cx += cw
            return y_top - header_h

        cur_y = draw_header(self.y)
        for idx, (w_cells, rh) in enumerate(zip(wrapped_rows, row_heights)):
            if cur_y - rh < BOTTOM_Y:
                self.new_page()
                cur_y = draw_header(self.y)
            bg = (0.97, 0.98, 1.0) if idx % 2 == 0 else (1.0, 1.0, 1.0)
            self.fill_rect(MARGIN_L, cur_y - rh, CONTENT_W, rh, bg, stroke_rgb=(0.85, 0.88, 0.94), line_w=0.5)
            cx = MARGIN_L
            for c_idx, (lines, cw) in enumerate(zip(w_cells, col_widths)):
                if c_idx > 0:
                    self.line(cx, cur_y, cx, cur_y - rh, rgb=(0.86, 0.89, 0.94), width=0.5)
                ty = cur_y - 10.5
                f_name = "F2" if c_idx == 0 else "F1"
                for ln in lines:
                    self.text(cx + 5, ty, ln, font=f_name, size=font_size, color=(0.13, 0.16, 0.23))
                    ty -= line_h
                cx += cw
            cur_y -= rh
        self.y = cur_y - 8.0

    def draw_architecture_diagram(self) -> None:
        """Draw a clean vector system architecture & LangGraph workflow diagram."""
        diag_h = 158.0
        self.ensure_space(diag_h + 12)
        top = self.y
        bottom = top - diag_h
        self.fill_rect(
            MARGIN_L,
            bottom,
            CONTENT_W,
            diag_h,
            (0.97, 0.98, 1.0),
            stroke_rgb=(0.76, 0.82, 0.94),
            line_w=0.8,
        )
        self.text(
            MARGIN_L + 10,
            top - 13,
            "Figure 1: End-to-End Multi-Modal Ingestion & 6-Node LangGraph Agentic RAG Workflow",
            font="F2",
            size=8.8,
            color=(0.12, 0.22, 0.48),
        )

        # Row 1: Ingestion Pipeline
        r1_y = top - 56.0
        boxes_r1 = [
            (MARGIN_L + 10, 108, "1. Multi-Format Sources", "FY23 XML | FY24 JSON", "FY25 CSV | Uploads (PDF/XLSX)"),
            (MARGIN_L + 136, 114, "2. Loaders & Normalizer", "pandas / ElementTree / pypdf", "13-Col Canonical Schema"),
            (MARGIN_L + 268, 118, "3. Semantic Chunker", "record (250) | trend (40)", "year (4) | overview | text"),
            (MARGIN_L + 404, 106, "4. Knowledge Stores", "ChromaDB (bge-small)", "+ net_cost_unified.csv"),
        ]
        for bx, bw, t1, t2, t3 in boxes_r1:
            self.fill_rect(bx, r1_y, bw, 36, (0.89, 0.93, 1.0), stroke_rgb=(0.30, 0.48, 0.85), line_w=0.9)
            self.text(bx + 5, r1_y + 25, t1, font="F2", size=7.8, color=(0.08, 0.18, 0.42))
            self.text(bx + 5, r1_y + 14, t2, font="F1", size=7.1, color=(0.20, 0.24, 0.34))
            self.text(bx + 5, r1_y + 4, t3, font="F1", size=7.1, color=(0.20, 0.24, 0.34))

        self.arrow(MARGIN_L + 118, r1_y + 18, MARGIN_L + 136, r1_y + 18)
        self.arrow(MARGIN_L + 250, r1_y + 18, MARGIN_L + 268, r1_y + 18)
        self.arrow(MARGIN_L + 386, r1_y + 18, MARGIN_L + 404, r1_y + 18)

        # Row 2: LangGraph 6-Node Pipeline
        r2_y = top - 122.0
        nodes = [
            (MARGIN_L + 10, 74, "Input Guard", "Length / PII", "Injection Check"),
            (MARGIN_L + 96, 78, "Planner Agent", "Intent / Years", "Agency Resolver"),
            (MARGIN_L + 186, 84, "Retriever Agent", "Pandas Tools +", "Chroma Vector"),
            (MARGIN_L + 282, 78, "Reasoner Agent", "Strict Context", "Citations [S#]"),
            (MARGIN_L + 372, 74, "Validator", "Numbers &", "Citation Check"),
            (MARGIN_L + 458, 52, "Finalize", "Sources", "+ Trace"),
        ]
        for nx, nw, n1, n2, n3 in nodes:
            self.fill_rect(nx, r2_y, nw, 38, (0.12, 0.20, 0.40), stroke_rgb=(0.42, 0.62, 0.98), line_w=0.9)
            self.text(nx + 4, r2_y + 26, n1, font="F2", size=7.6, color=(1.0, 1.0, 1.0))
            self.text(nx + 4, r2_y + 15, n2, font="F1", size=6.9, color=(0.82, 0.88, 0.98))
            self.text(nx + 4, r2_y + 5, n3, font="F1", size=6.9, color=(0.82, 0.88, 0.98))

        self.arrow(MARGIN_L + 84, r2_y + 19, MARGIN_L + 96, r2_y + 19)
        self.arrow(MARGIN_L + 174, r2_y + 19, MARGIN_L + 186, r2_y + 19)
        self.arrow(MARGIN_L + 270, r2_y + 19, MARGIN_L + 282, r2_y + 19)
        self.arrow(MARGIN_L + 360, r2_y + 19, MARGIN_L + 372, r2_y + 19)
        self.arrow(MARGIN_L + 446, r2_y + 19, MARGIN_L + 458, r2_y + 19)

        # Store -> Retriever orthogonal connector in the gap between Row 1 and Row 2
        mid_y = top - 70.0
        self.line(MARGIN_L + 457, r1_y, MARGIN_L + 457, mid_y, rgb=(0.18, 0.52, 0.36), width=1.0)
        self.line(MARGIN_L + 457, mid_y, MARGIN_L + 228, mid_y, rgb=(0.18, 0.52, 0.36), width=1.0)
        self.arrow(MARGIN_L + 228, mid_y, MARGIN_L + 228, r2_y + 38, rgb=(0.18, 0.52, 0.36), width=1.0)
        self.text(MARGIN_L + 292, mid_y + 3, "Hybrid Context Retrieval ([S1..Sn])", font="F2", size=7.2, color=(0.15, 0.45, 0.30))

        # Validator -> Reasoner retry loop
        retry_y = bottom + 12.0
        self.line(MARGIN_L + 409, r2_y, MARGIN_L + 409, retry_y, rgb=(0.82, 0.45, 0.12), width=1.0)
        self.line(MARGIN_L + 409, retry_y, MARGIN_L + 321, retry_y, rgb=(0.82, 0.45, 0.12), width=1.0)
        self.arrow(MARGIN_L + 321, retry_y, MARGIN_L + 321, r2_y, rgb=(0.82, 0.45, 0.12), width=1.0)
        self.text(
            MARGIN_L + 328,
            retry_y + 3,
            "Self-Correction Retry (if ungrounded figure)",
            font="F2",
            size=6.9,
            color=(0.75, 0.36, 0.08),
        )

        self.y = bottom - 10.0

    # ------------------------------------------------------------------ serialization
    def save(self, path: Path) -> None:
        total_pages = len(self.pages)
        # Add footer with page numbers to every page
        for idx, page_cmds in enumerate(self.pages, start=1):
            r, g, b = (0.45, 0.49, 0.58)
            page_cmds.append(
                f"q 0.84 0.87 0.93 RG 0.6 w {MARGIN_L:.2f} {BOTTOM_Y - 10:.2f} m "
                f"{ PAGE_W - MARGIN_R:.2f} {BOTTOM_Y - 10:.2f} l S Q"
            )
            footer_l = "USFR Multi-Modal Agentic RAG System - Technical Documentation"
            footer_r = f"Page {idx} of {total_pages}"
            page_cmds.append(
                f"BT /F1 8.0 Tf {r:.3f} {g:.3f} {b:.3f} rg {MARGIN_L:.2f} {BOTTOM_Y - 22:.2f} Td "
                f"({_esc(footer_l)}) Tj ET"
            )
            page_cmds.append(
                f"BT /F2 8.0 Tf {r:.3f} {g:.3f} {b:.3f} rg {PAGE_W - MARGIN_R - 52:.2f} {BOTTOM_Y - 22:.2f} Td "
                f"({_esc(footer_r)}) Tj ET"
            )

        objects: list[bytes] = []

        def add_obj(body: str | bytes) -> int:
            data = body.encode("latin-1") if isinstance(body, str) else body
            objects.append(data)
            return len(objects)

        # 1..4: Standard Type1 Fonts (WinAnsiEncoding)
        f1 = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        f3 = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>")
        f4 = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")

        pages_id = len(objects) + 1 + len(self.pages) * 2
        page_ids: list[int] = []

        for page_cmds in self.pages:
            stream_bytes = "\n".join(page_cmds).encode("latin-1")
            c_id = add_obj(
                f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
                + stream_bytes
                + b"\nendstream"
            )
            p_id = add_obj(
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R /F4 {f4} 0 R >> >> "
                f"/Contents {c_id} 0 R >>"
            )
            page_ids.append(p_id)

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        actual_pages_id = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
        assert actual_pages_id == pages_id
        catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("latin-1"))
            out.extend(obj)
            out.extend(b"\nendobj\n")

        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
        out.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
        )
        path.write_bytes(bytes(out))


def build_report(output_path: Path) -> None:
    pdf = PDFDocument()
    pdf.title_banner()

    # Executive Summary
    pdf.section("1", "Executive Summary & Project Overview")
    pdf.paragraph(
        "The U.S. Government Financial Report (USFR) Multi-Modal Agentic RAG System is an autonomous "
        "knowledge and decision-support application for enterprise and federal financial intelligence. "
        "It enables analysts, auditors, policymakers, and citizens to "
        "upload and query heterogeneous enterprise and federal financial documents (XML, JSON, CSV, Excel, "
        "PDF, TXT, and Markdown) using natural language."
    )
    pdf.paragraph(
        "Traditional Retrieval-Augmented Generation (RAG) systems struggle with financial statements because "
        "character-level text splitting severs numbers from their agency and fiscal-year labels, and Large "
        "Language Models (LLMs) are prone to arithmetic hallucinations when computing rankings or year-over-year "
        "changes. This system solves both problems by combining multi-format schema normalization, "
        "four-level semantic chunking in ChromaDB, deterministic pandas analytical tools, and a 6-node "
        "LangGraph multi-agent workflow with an automated numeric grounding validator."
    )

    pdf.table(
        ["Capstone Task", "Requirement Summary", "Implementation Module(s)"],
        [
            ["1. Project Foundation", "Repository layout, .env config, modular structure", "src/config.py, setup_key.py, .env.example"],
            ["2. User Interaction Layer", "Web UI + CLI for uploads and natural-language Q&A", "app.py (Streamlit 3-tab UI), main.py (REPL)"],
            ["3. Document Ingestion", "Multi-format loading (XML, JSON, CSV, XLSX, PDF, TXT)", "src/ingestion/loaders.py, ingest.py"],
            ["4. Semantic Chunking", "Multi-granularity financial chunks + recursive text split", "src/ingestion/documents.py (295 chunks)"],
            ["5. Vector Knowledge Store", "Local embeddings + persistent vector DB", "src/vectorstore.py (ChromaDB + bge-small)"],
            ["6. Intelligent Retrieval", "Hybrid similarity search + structured analytical tools", "src/agents/tools.py, src/data_store.py"],
            ["7. Grounded RAG Pipeline", "Numbered context [S1..Sn] + strict inline citations", "src/agents/graph.py (reasoner node)"],
            ["8. Agent-Based Reasoning", "Autonomous planning, tool execution, validation loop", "src/agents/graph.py (6-node LangGraph)"],
            ["9. Safety & Reliability", "Prompt-injection/PII guardrails + numeric verification", "src/guardrails.py, validator node"],
            ["10. Deployment & Docs", "Streamlit deployment, architecture diagrams & tests", "docs/ARCHITECTURE.md, tests/test_system.py"],
        ],
        [0.24, 0.42, 0.34],
    )

    # Section 2: System Setup
    pdf.section("2", "System Setup & Environment Configuration")
    pdf.subsection("2.1 Technology Stack & Open-Source Dependencies")
    pdf.bullet("Agent Orchestration:", "LangGraph (>=0.2.0) and LangChain Core/Community (>=0.3.0) for state-machine agent workflows and tool binding.")
    pdf.bullet("LLM Providers:", "Google Gemini ('gemini-3-flash-preview' with automatic fallback to 'gemini-2.5-flash-lite', 'gemini-3.1-flash-lite', and 'gemini-2.5-pro') for cloud inference, or Ollama ('llama3.1:8b') for 100% local open-source execution.")
    pdf.bullet("Local Embeddings & Vector DB:", "HuggingFace 'BAAI/bge-small-en-v1.5' (sentence-transformers, CPU-friendly, cosine-normalized) persisted in ChromaDB ('chroma_store/').")
    pdf.bullet("Multi-Format Parsing & UI:", "pandas, openpyxl, ElementTree, and pypdf for ingestion; Streamlit and Plotly for the interactive web dashboard.")

    pdf.subsection("2.2 Step-by-Step Installation & API Key Configuration")
    pdf.code_block(
        [
            "# 1. Create and activate Python 3.11+ virtual environment",
            "python3 -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements.txt",
            "",
            "# 2. Configure API key in .env (via interactive CLI, flag, or Streamlit sidebar)",
            "python setup_key.py --key AIzaSyYourGeminiKeyHere",
            "# Or switch to 100% local open-source Ollama:",
            "python setup_key.py --provider ollama --model llama3.1:8b",
            "",
            "# 3. Build / refresh the unified structured table and ChromaDB vector store",
            "python ingest.py --reset",
        ],
        title="Terminal Setup & Ingestion Commands",
    )

    # Section 3: System Architecture
    pdf.section("3", "System Architecture & Data Pipeline")
    pdf.draw_architecture_diagram()

    pdf.subsection("3.1 Multi-Format Ingestion & Canonical Schema Normalization")
    pdf.paragraph(
        "The project ships with three official U.S. Treasury Fiscal Data files spanning FY2022-FY2025 in three "
        "different formats: 'USFR_StmtNetCost_20230101_20231231.xml' (FY2023 + FY2022 restated), "
        "'USFR_StmtNetCost_20240101_20241231.json' (FY2024 + FY2023 restated), and "
        "'USFR_StmtNetCost_20250101_20250930.csv' (FY2025 + FY2024 restated). In addition, users can upload "
        "custom PDF, TXT, Markdown, CSV, Excel (.xlsx/.xls), JSON, and XML files up to 25 MB."
    )
    pdf.paragraph(
        "In 'src/ingestion/loaders.py', FIELD_MAP maps both API field names (such as 'stmt_fiscal_year', "
        "'gross_cost_bil_amt', 'restmt_flag') and human-readable CSV headers ('Statement Fiscal Year', "
        "'Gross Cost (in Billions)', 'Restatement Flag') into a unified 13-column canonical table "
        "('data/processed/net_cost_unified.csv', 250 deduplicated rows across 40 federal entities). All numeric "
        "columns are rounded to 1 decimal place to eliminate IEEE-754 floating-point artifacts."
    )

    pdf.subsection("3.2 Four-Level Semantic Chunking Strategy")
    pdf.paragraph(
        "Rather than splitting financial tables by arbitrary character counts, 'src/ingestion/documents.py' "
        "synthesizes self-contained natural-language chunks at four complementary granularities, each tagged "
        "with rich metadata and a deterministic SHA-1 ID for idempotent ChromaDB upserts:"
    )
    pdf.bullet("1. Record Chunks (250 chunks):", "One natural-language block per (agency, statement FY, report FY) row containing gross cost, earned revenue, subtotal, assumption changes, net cost, restatement status, and source file/line lineage.")
    pdf.bullet("2. Agency Trend Chunks (40 chunks):", "One multi-year longitudinal summary per agency across FY2022-FY2025 with pre-computed year-over-year dollar and percentage deltas.")
    pdf.bullet("3. Year Summary Chunks (4 chunks):", "One macro summary per fiscal year (FY2022-FY2025) listing government-wide totals, top 10 highest-cost agencies, and entities with negative net cost.")
    pdf.bullet("4. Dataset Overview & Free-Text Chunks:", "A domain glossary explaining federal accounting terms (Gross Cost, Earned Revenue, Changes in Assumptions, Restatement Flag) plus RecursiveCharacterTextSplitter chunks (900 chars, 150 overlap) for uploaded PDFs/TXT.")

    # Section 4: Agent Roles
    pdf.section("4", "Autonomous Agent Roles (LangGraph Workflow)")
    pdf.paragraph(
        "The core reasoning engine in 'src/agents/graph.py' is implemented as a compiled LangGraph StateGraph "
        "with six specialized nodes and conditional edges:"
    )
    pdf.table(
        ["Agent / Node", "Primary Responsibility", "Tools / Mechanisms Used"],
        [
            [
                "1. Input Guard\n(input_guard)",
                "Pre-LLM security and input validation gatekeeper. Blocks malicious or invalid inputs immediately.",
                "Regex checks for prompt injection ('ignore previous instructions', 'jailbreak'), unsafe queries, length (>1000 chars), and PII (SSN, credit card).",
            ],
            [
                "2. Planner Agent\n(planner)",
                "Rewrites follow-up queries using chat history, classifies intent, resolves agency abbreviations, and plans tool calls.",
                "LLM structured output (Plan Pydantic schema) + 35+ agency alias map (DoD, HHS, SSA, VA, OPM) + deterministic _heuristic_plan() fallback.",
            ],
            [
                "3. Retriever Agent\n(retriever)",
                "Executes hybrid retrieval according to the structured Plan and formats numbered context blocks [S1..Sn].",
                "lookup_financials, rank_agencies, compare_fiscal_years, restatement_differences (pandas) + semantic_search (ChromaDB cosine similarity).",
            ],
            [
                "4. Reasoner Agent\n(reasoner)",
                "Synthesizes a concise, grounded markdown response strictly from [S1..Sn] at temperature=0.",
                "Strict system prompt requiring [S#] citations on every factual claim and prohibiting unverified mental arithmetic.",
            ],
            [
                "5. Validator Agent\n(validator)",
                "Deterministically checks numeric accuracy and citation integrity; triggers self-correction retry if needed.",
                "ungrounded_numbers() (parses numbers, scales trillions<->billions, checks +/-0.05B tolerance) + has_citations() ([S#] existence check).",
            ],
            [
                "6. Finalize Node\n(finalize)",
                "Packages verified answer, attaches cited source metadata, and records full execution trace for the UI.",
                "Filters context blocks to cited [S#] IDs; appends verification note if retry budget (MAX_VALIDATION_RETRIES=1) is exhausted.",
            ],
        ],
        [0.20, 0.40, 0.40],
    )

    # Section 5: Deployment Steps
    pdf.section("5", "Deployment Steps & Verification Suite")
    pdf.subsection("5.1 Running Locally (Streamlit Web UI & CLI REPL)")
    pdf.paragraph(
        "The application supports both a browser-based multi-tab Streamlit dashboard ('app.py') and an "
        "interactive command-line REPL ('main.py'). On first launch, 'ensure_knowledge_base()' automatically "
        "bootstraps 'data/raw/' and populates 'chroma_store/' and 'data/processed/net_cost_unified.csv'."
    )
    pdf.code_block(
        [
            "# Launch the Streamlit Web UI (http://localhost:8501)",
            ".venv/bin/streamlit run app.py",
            "",
            "# Launch the Interactive Terminal REPL",
            ".venv/bin/python main.py",
            "",
            "# Run the Automated Pytest Verification Suite (10 tests)",
            ".venv/bin/pytest -v",
        ],
        title="Application Execution & Testing Commands",
    )

    pdf.subsection("5.2 Cloud & Container Deployment")
    pdf.bullet("Streamlit Community Cloud:", "Push repository to GitHub ('.env' and 'chroma_store/' are excluded via '.gitignore'). Set 'app.py' as the entrypoint and add 'GEMINI_API_KEY' in App Settings -> Secrets. The app auto-ingests the USFR dataset on cold start.")
    pdf.bullet("100% Local / Air-Gapped Deployment:", "Install Ollama, pull 'llama3.1:8b', and set 'LLM_PROVIDER=ollama' in '.env'. Combined with local HuggingFace 'BAAI/bge-small-en-v1.5' embeddings and ChromaDB, the entire stack runs offline with zero external API calls.")

    # Section 6: Challenges & Limitations
    pdf.section("6", "Challenges Faced & System Limitations")
    pdf.subsection("6.1 Key Engineering Challenges & How They Were Solved")
    pdf.table(
        ["Challenge Encountered", "Root Cause", "Engineering Solution Implemented"],
        [
            [
                "1. Heterogeneous Multi-Year File Formats",
                "FY2023 data arrives as XML with API tags, FY2024 as nested JSON, and FY2025 as CSV with display headers.",
                "Built unified readers in loaders.py and FIELD_MAP to normalize all three formats into a 13-column canonical table.",
            ],
            [
                "2. Floating-Point Artifacts in Raw Data",
                "XML and JSON files contained IEEE-754 artifacts such as 1.39999999999999 instead of 1.4 billion.",
                "Applied pd.to_numeric(...).round(1) during normalization so all figures match official published Treasury tables.",
            ],
            [
                "3. Prior-Year Restatement Double-Counting",
                "Each annual report includes restated prior-year figures (restatement_flag='Y') alongside current-year rows ('N').",
                "Implemented FinancialDataStore.preferred() to default to original figures ('N') while exposing restatement_diff() for revision queries.",
            ],
            [
                "4. LLM Arithmetic & Ranking Hallucinations",
                "LLMs frequently miscalculate multi-row rankings, sums, or percentage changes over raw text chunks.",
                "Offloaded all lookups, rankings, and YoY deltas to deterministic pandas tools and added ungrounded_numbers() validation.",
            ],
            [
                "5. Model Deprecation & API Resilience",
                "Older model aliases ('gemini-2.5-flash') return 404 NOT_FOUND for newly issued Gemini API keys.",
                "Updated default to 'gemini-3-flash-preview' with automatic multi-model fallbacks in src/llm.py plus heuristic planner fallback.",
            ],
        ],
        [0.24, 0.36, 0.40],
    )

    pdf.subsection("6.2 Current Limitations & Future Roadmap")
    pdf.bullet("Scanned Image-Only PDFs:", "The PDF loader ('pypdf') extracts digital text streams. Scanned image-only PDFs without an embedded text layer require an OCR pre-processing stage (e.g., Tesseract or Google Document AI).")
    pdf.bullet("Pre-Loaded Financial Statement Scope:", "The default Treasury dataset covers the Statements of Net Cost (FY2022-FY2025). Answering questions about federal tax revenue collections or debt balance sheets requires uploading the corresponding Treasury files via the sidebar uploader.")
    pdf.bullet("Single-Pass Validator Retry Cap:", "To maintain sub-10-second response latency, automatic self-correction between the Validator and Reasoner is capped at MAX_VALIDATION_RETRIES=1; if still unverified, a transparent warning banner is attached.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(output_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out_main = root / "Capstone_Project_Documentation.pdf"
    out_docs = root / "docs" / "Capstone_Project_Documentation.pdf"
    build_report(out_main)
    out_docs.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_main, out_docs)
    print(f"Generated {out_main} ({out_main.stat().st_size:,} bytes)")
    print(f"Copied to {out_docs}")
