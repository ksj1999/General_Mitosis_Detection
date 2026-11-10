from pathlib import Path
import re
import textwrap


REPORT_MD = Path("jun_model/demo1/reports/midogpp_scanner_generalization_report.md")
REPORT_PDF = Path("jun_model/demo1/reports/midogpp_scanner_generalization_report.pdf")


def escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def clean_markdown_line(line):
    line = line.rstrip()
    line = re.sub(r"^\s*#+\s*", "", line)
    line = line.replace("**", "")
    line = line.replace("`", "")
    return line


def md_to_lines(markdown):
    lines = []
    for raw in markdown.splitlines():
        if not raw.strip():
            lines.append("")
            continue

        if raw.startswith("|"):
            cells = [cell.strip() for cell in raw.strip("|").split("|")]
            if all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            lines.append(" | ".join(cells))
            continue

        cleaned = clean_markdown_line(raw)
        if cleaned.startswith("- "):
            cleaned = "* " + cleaned[2:]
        lines.append(cleaned)
    return lines


def wrap_lines(lines, width=92):
    wrapped = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        indent = "  " if line.startswith("* ") else ""
        pieces = textwrap.wrap(line, width=width, subsequent_indent=indent)
        wrapped.extend(pieces or [""])
    return wrapped


def paginate(lines, lines_per_page=46):
    pages = []
    current = []
    for line in lines:
        if len(current) >= lines_per_page:
            pages.append(current)
            current = []
        current.append(line)
    if current:
        pages.append(current)
    return pages


def page_stream(lines, page_number, page_count):
    commands = ["BT", "/F1 10 Tf", "50 760 Td", "14 TL"]
    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.extend(["/F1 8 Tf", f"(Page {page_number} of {page_count}) Tj", "ET"])
    return "\n".join(commands).encode("ascii", errors="replace")


def build_pdf(pages):
    objects = []

    def add_object(data):
        objects.append(data)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    content_ids = []
    for index, page_lines in enumerate(pages, start=1):
        stream = page_stream(page_lines, index, len(pages))
        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add_object(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            + b"/Contents "
            + str(content_id).encode("ascii")
            + b" 0 R >>"
        )
        content_ids.append(content_id)
        page_ids.append(page_id)

    kids = b" ".join(str(page_id).encode("ascii") + b" 0 R" for page_id in page_ids)
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode("ascii") + b" >>"

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, data in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(data)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        b"trailer\n"
        + b"<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\n"
        + b"startxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    return bytes(pdf)


def main():
    markdown = REPORT_MD.read_text(encoding="utf-8")
    lines = wrap_lines(md_to_lines(markdown))
    pages = paginate(lines)
    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PDF.write_bytes(build_pdf(pages))
    print(f"wrote {REPORT_PDF}")


if __name__ == "__main__":
    main()
