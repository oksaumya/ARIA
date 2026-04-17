from typing import List
from src.agent.state import ResearchReport


def render_markdown(report: ResearchReport) -> str:
    """Convert a ResearchReport to formatted markdown."""
    lines = [
        f"# {report['title']}",
        "",
        "## Abstract",
        report.get("abstract", ""),
        "",
        "## Key Findings",
    ]
    for finding in report.get("key_findings", []):
        lines.append(f"- {finding}")
    lines += [
        "",
        "## Conclusion",
        report.get("conclusion", ""),
        "",
        "## Sources",
    ]
    for s in report.get("sources", []):
        title = s.get("title", "Unknown")
        url = s.get("url", "#")
        lines.append(f"- [{title}]({url})")

    return "\n".join(lines)


def export_pdf(report_markdown: str, qa_history: List[dict] = None) -> bytes:
    """Generate a PDF from report markdown and optional Q&A history."""
    import re
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Effective width using default 10mm margins on each side
    W = pdf.w - 2 * pdf.l_margin

    def _safe(text: str) -> str:
        # Strip markdown links [text](url) → text, then encode to latin-1
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    for line in report_markdown.split("\n"):
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(W, 10, _safe(line[2:]))
            pdf.ln(4)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
            pdf.multi_cell(W, 8, _safe(line[3:]))
            pdf.ln(2)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(W, 6, _safe(f"- {line[2:]}"))
        elif line.strip():
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(W, 6, _safe(line))
        else:
            pdf.ln(3)

    if qa_history:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(W, 10, "Follow-up Q&A")
        pdf.ln(4)

        for i, qa in enumerate(qa_history, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(W, 7, _safe(f"Q{i}: {qa['question']}"))
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(W, 6, _safe(f"A: {qa['answer']}"))
            pdf.ln(4)

    return bytes(pdf.output())
