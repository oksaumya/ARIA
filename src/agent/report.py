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

    # Use consistent margins
    LEFT_MARGIN = 15
    RIGHT_MARGIN = 15
    pdf.set_left_margin(LEFT_MARGIN)
    pdf.set_right_margin(RIGHT_MARGIN)

    pdf.add_page()

    # Effective content width
    W = pdf.w - LEFT_MARGIN - RIGHT_MARGIN

    def _safe(text: str) -> str:
        """Strip all markdown formatting and encode to latin-1."""
        # Strip markdown links [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Strip bold **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        # Strip italic *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
        # Strip inline code `text`
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Encode to latin-1 for PDF compatibility
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _reset_x():
        """Reset cursor to left margin to prevent drift."""
        pdf.set_x(LEFT_MARGIN)

    def _write_cell(width, height, text, align="L"):
        """Write a multi_cell with explicit x-position reset."""
        _reset_x()
        pdf.multi_cell(width, height, text, align=align)
        _reset_x()

    # --- Title / H1 ---
    first_h1 = True

    for line in report_markdown.split("\n"):
        stripped = line.strip()

        if stripped.startswith("# "):
            # H1 — main title
            _reset_x()
            pdf.set_font("Helvetica", "B", 20)
            if first_h1:
                pdf.ln(5)
                first_h1 = False
            _write_cell(W, 10, _safe(stripped[2:]), align="L")
            # Draw a separator line under the title
            y = pdf.get_y() + 2
            pdf.line(LEFT_MARGIN, y, pdf.w - RIGHT_MARGIN, y)
            pdf.ln(6)

        elif stripped.startswith("### "):
            # H3 — sub-sub-heading
            _reset_x()
            pdf.set_font("Helvetica", "B", 11)
            pdf.ln(3)
            _write_cell(W, 7, _safe(stripped[4:]))
            pdf.ln(1)

        elif stripped.startswith("## "):
            # H2 — section heading
            _reset_x()
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(5)
            _write_cell(W, 9, _safe(stripped[3:]))
            pdf.ln(3)

        elif stripped.startswith("- "):
            # Bullet point — use indent for proper alignment
            _reset_x()
            pdf.set_font("Helvetica", "", 11)
            bullet_indent = 5
            bullet_text = _safe(stripped[2:])
            pdf.set_x(LEFT_MARGIN + bullet_indent)
            bullet_w = W - bullet_indent
            # Use a bullet character instead of dash for cleaner look
            pdf.multi_cell(bullet_w, 6, f"\x95  {bullet_text}")
            _reset_x()

        elif stripped:
            # Body text
            _reset_x()
            pdf.set_font("Helvetica", "", 11)
            _write_cell(W, 6, _safe(stripped))

        else:
            # Empty line — small vertical space
            pdf.ln(3)

    # --- Q&A Section ---
    if qa_history:
        pdf.add_page()
        _reset_x()
        pdf.set_font("Helvetica", "B", 16)
        _write_cell(W, 10, "Follow-up Q&A")
        y = pdf.get_y() + 2
        pdf.line(LEFT_MARGIN, y, pdf.w - RIGHT_MARGIN, y)
        pdf.ln(6)

        for i, qa in enumerate(qa_history, 1):
            _reset_x()
            pdf.set_font("Helvetica", "B", 11)
            _write_cell(W, 7, _safe(f"Q{i}: {qa['question']}"))
            pdf.ln(1)

            _reset_x()
            pdf.set_font("Helvetica", "", 11)
            _write_cell(W, 6, _safe(f"A: {qa['answer']}"))
            pdf.ln(5)

    return bytes(pdf.output())
