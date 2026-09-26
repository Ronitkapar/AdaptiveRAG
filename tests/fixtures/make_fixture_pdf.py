"""
make_fixture_pdf.py
-------------------
Generates a deterministic synthetic 2-page academic paper PDF using ReportLab.
Used exclusively in automated tests for ingestion, chunking, and normalizer verification.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_synthetic_paper_pdf(dest_path: Path) -> Path:
    """Create a 2-page deterministic academic-style paper PDF."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(dest_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,  # Center
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "PaperH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "PaperH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=8,
    )

    caption_style = ParagraphStyle(
        "PaperCaption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=11,
        spaceBefore=4,
        spaceAfter=8,
    )

    story = []

    # Page 1: Title + Abstract + Section 1
    story.append(Paragraph("A Synthetic Study on Retrieval Architectures", title_style))
    story.append(
        Paragraph("Abstract. This paper evaluates dense retrieval pipelines.", body_style)
    )

    story.append(Paragraph("1. Introduction", h1_style))
    story.append(
        Paragraph(
            "Dense retrieval maps queries and documents into continuous vector spaces. "
            "Bi-encoders encode queries and passages independently, enabling sub-linear MIPS search.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Earlier lexical approaches like BM25 rely on inverted indices and exact term matching.",
            body_style,
        )
    )

    story.append(Paragraph("1.1 Motivation and Scope", h2_style))
    story.append(
        Paragraph(
            "We examine whether fixed dense baseline performance suffices across diverse queries.",
            body_style,
        )
    )

    # Page break to Page 2
    story.append(PageBreak())

    # Page 2: Section 2 with Table and Equation
    story.append(Paragraph("2. Experimental Results", h1_style))
    story.append(
        Paragraph(
            "We report top-k accuracy across different retrieval settings below.",
            body_style,
        )
    )

    # Table
    table_data = [
        ["Method", "Recall@5", "MRR@10"],
        ["BM25", "0.64", "0.52"],
        ["Dense (ours)", "0.82", "0.71"],
    ]
    t = Table(table_data, colWidths=[150, 100, 100])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(
        Paragraph(
            "Table 1: Retrieval performance comparison across benchmarks.",
            caption_style,
        )
    )

    # Display Equation-like paragraph
    story.append(
        Paragraph(
            "Score(q, d) = cos(E_q(q), E_d(d)) = (E_q(q) · E_d(d)) / (||E_q(q)|| ||E_d(d)||)",
            body_style,
        )
    )

    # Conclusion
    story.append(Paragraph("3. Conclusion", h1_style))
    story.append(
        Paragraph(
            "Dense baselines deliver strong semantic matching when tuned properly.",
            body_style,
        )
    )

    doc.build(story)
    return dest_path
