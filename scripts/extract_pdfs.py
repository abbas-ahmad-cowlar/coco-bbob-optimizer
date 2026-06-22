"""Extract reference-PDF text, images, OCR, tables, and readable Markdown.

Outputs are written under docs/extracted-pdfs/<pdf-stem>/.
This script is intentionally local/project-specific so reference PDFs can
be processed with the same repeatable workflow.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pymupdf4llm
import pytesseract
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_ROOT = DOCS_DIR / "extracted-pdfs"
DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass
class ImageRecord:
    page: int
    index: int
    xref: int
    width: int | None
    height: int | None
    colorspace: str | None
    bpc: int | None
    file: str
    ocr_file: str | None


@dataclass
class PdfRecord:
    source_pdf: str
    title: str
    slug: str
    pages: int
    output_dir: str
    native_text_file: str
    ocr_text_file: str
    layout_markdown_file: str | None
    tables_file: str
    combined_markdown_file: str
    rendered_pages_dir: str
    images_dir: str
    extracted_images: int


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-") or "pdf"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def rel(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def configure_tesseract() -> bool:
    explicit = os.environ.get("TESSERACT_CMD")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(DEFAULT_TESSERACT)
    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return True
    return False


def ocr_image_file(image_path: Path) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            return pytesseract.image_to_string(img)
    except Exception as exc:  # keep extraction moving
        return f"[OCR failed for {image_path.name}: {type(exc).__name__}: {exc}]"


def markdown_table(table: list[list[str | None]]) -> str:
    if not table:
        return ""
    max_cols = max(len(row) for row in table)
    rows = []
    for row in table:
        clean = [("" if cell is None else str(cell).replace("\n", " ").strip()) for cell in row]
        clean += [""] * (max_cols - len(clean))
        rows.append(clean)
    header = rows[0]
    body = rows[1:]
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(out)


def extract_tables(pdf_path: Path, out_dir: Path) -> Path:
    tables_path = out_dir / "tables.md"
    chunks = [f"# Extracted Tables\n\nSource: `{pdf_path.name}`\n"]
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                if not tables:
                    continue
                chunks.append(f"\n## Page {page_num}\n")
                for idx, table in enumerate(tables, start=1):
                    chunks.append(f"\n### Table {idx}\n\n{markdown_table(table)}\n")
    except Exception as exc:
        chunks.append(f"\n[Table extraction failed: {type(exc).__name__}: {exc}]\n")
    write_text(tables_path, "\n".join(chunks))
    return tables_path


def extract_one(pdf_path: Path, overwrite: bool = True, dpi: int = 200) -> PdfRecord:
    slug = slugify(pdf_path.stem)
    out_dir = OUT_ROOT / slug
    if overwrite and out_dir.exists():
        # Non-destructive enough for generated output: clear only this script's output folder.
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "rendered-pages"
    images_dir = out_dir / "images"
    ocr_pages_dir = out_dir / "page-ocr"
    image_ocr_dir = out_dir / "image-ocr"
    for folder in (pages_dir, images_dir, ocr_pages_dir, image_ocr_dir):
        folder.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    metadata = dict(doc.metadata or {})
    metadata["page_count"] = len(doc)
    write_text(out_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

    native_chunks = [f"# Native Extracted Text\n\nSource: `{pdf_path.name}`\n"]
    ocr_chunks = [f"# OCR Text From Rendered Pages\n\nSource: `{pdf_path.name}`\n"]
    combined_chunks = [
        f"# {pdf_path.stem}\n",
        f"Source PDF: `{pdf_path.name}`  ",
        f"Pages: {len(doc)}\n",
        "This Markdown combines native text extraction, OCR from rendered pages, extracted image links, image OCR, and table references.\n",
    ]
    image_records: list[ImageRecord] = []
    page_images: dict[int, list[ImageRecord]] = {}

    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    for page_index, page in enumerate(doc, start=1):
        native_text = page.get_text("text") or ""
        native_chunks.append(f"\n\n## Page {page_index}\n\n{native_text.strip()}\n")

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_image = pages_dir / f"page-{page_index:03d}.png"
        pix.save(page_image)
        page_ocr = ocr_image_file(page_image)
        page_ocr_path = ocr_pages_dir / f"page-{page_index:03d}.txt"
        write_text(page_ocr_path, page_ocr)
        ocr_chunks.append(f"\n\n## Page {page_index}\n\n{page_ocr.strip()}\n")

        seen_xrefs: set[int] = set()
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                extracted = doc.extract_image(xref)
                ext = extracted.get("ext", "png")
                image_bytes = extracted["image"]
                image_name = f"page-{page_index:03d}-image-{image_index:02d}-xref-{xref}.{ext}"
                image_path = images_dir / image_name
                image_path.write_bytes(image_bytes)

                # OCR the embedded image as-is.
                image_ocr_text = ocr_image_file(image_path)
                image_ocr_path = image_ocr_dir / f"{image_path.stem}.txt"
                write_text(image_ocr_path, image_ocr_text)

                width = extracted.get("width")
                height = extracted.get("height")
                record = ImageRecord(
                    page=page_index,
                    index=image_index,
                    xref=xref,
                    width=width,
                    height=height,
                    colorspace=str(extracted.get("colorspace")) if extracted.get("colorspace") else None,
                    bpc=extracted.get("bpc"),
                    file=rel(image_path, out_dir),
                    ocr_file=rel(image_ocr_path, out_dir),
                )
                image_records.append(record)
                page_images.setdefault(page_index, []).append(record)
            except Exception as exc:
                fail_path = image_ocr_dir / f"page-{page_index:03d}-image-{image_index:02d}-xref-{xref}-error.txt"
                write_text(fail_path, f"Image extraction failed: {type(exc).__name__}: {exc}")

        combined_chunks.append(f"\n---\n\n## Page {page_index}\n")
        combined_chunks.append(f"\nRendered page image: [page-{page_index:03d}.png]({rel(page_image, out_dir)})\n")
        combined_chunks.append("\n### Native Text\n\n")
        combined_chunks.append(native_text.strip() or "[No native text extracted]")
        combined_chunks.append("\n\n### OCR Text\n\n")
        combined_chunks.append(page_ocr.strip() or "[No OCR text extracted]")
        if page_images.get(page_index):
            combined_chunks.append("\n\n### Extracted Images\n")
            for record in page_images[page_index]:
                combined_chunks.append(
                    f"\n- Image {record.index}: [{record.file}]({record.file})"
                    f" ({record.width}x{record.height}, xref {record.xref})"
                    f"; OCR: [{record.ocr_file}]({record.ocr_file})"
                )

    native_text_path = out_dir / "native-text.md"
    ocr_text_path = out_dir / "ocr-text.md"
    write_text(native_text_path, "\n".join(native_chunks))
    write_text(ocr_text_path, "\n".join(ocr_chunks))
    write_text(out_dir / "images.json", json.dumps([asdict(r) for r in image_records], indent=2, ensure_ascii=False))

    layout_markdown_path: Path | None = out_dir / "layout-aware-markdown.md"
    try:
        layout_md = pymupdf4llm.to_markdown(str(pdf_path))
        write_text(layout_markdown_path, layout_md)
    except Exception as exc:
        write_text(out_dir / "layout-aware-markdown-error.txt", f"{type(exc).__name__}: {exc}")
        layout_markdown_path = None

    tables_path = extract_tables(pdf_path, out_dir)
    combined_chunks.append("\n\n---\n\n## Extracted Tables\n\n")
    combined_chunks.append(f"See [tables.md]({rel(tables_path, out_dir)}).\n")
    combined_path = out_dir / "document.md"
    write_text(combined_path, "\n".join(combined_chunks))

    return PdfRecord(
        source_pdf=str(pdf_path),
        title=metadata.get("title") or pdf_path.stem,
        slug=slug,
        pages=len(doc),
        output_dir=str(out_dir),
        native_text_file=str(native_text_path),
        ocr_text_file=str(ocr_text_path),
        layout_markdown_file=str(layout_markdown_path) if layout_markdown_path else None,
        tables_file=str(tables_path),
        combined_markdown_file=str(combined_path),
        rendered_pages_dir=str(pages_dir),
        images_dir=str(images_dir),
        extracted_images=len(image_records),
    )


def main() -> int:
    tesseract_available = configure_tesseract()
    if not tesseract_available:
        print("WARNING: Tesseract executable not found. OCR will fail.", file=sys.stderr)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf in pdfs:
        print(f"Extracting {pdf.name}...")
        record = extract_one(pdf)
        records.append(record)
        print(f"  -> {record.output_dir} ({record.pages} pages, {record.extracted_images} images)")

    index_lines = [
        "# Extracted Reference PDFs\n",
        "This index was generated by `scripts/extract_pdfs.py`.\n",
    ]
    for record in records:
        out_dir = Path(record.output_dir)
        index_lines.extend(
            [
                f"\n## {Path(record.source_pdf).name}\n",
                f"- Pages: {record.pages}",
                f"- Extracted embedded images: {record.extracted_images}",
                f"- Combined readable Markdown: [{rel(Path(record.combined_markdown_file), DOCS_DIR)}]({rel(Path(record.combined_markdown_file), DOCS_DIR)})",
                f"- Native text: [{rel(Path(record.native_text_file), DOCS_DIR)}]({rel(Path(record.native_text_file), DOCS_DIR)})",
                f"- OCR text: [{rel(Path(record.ocr_text_file), DOCS_DIR)}]({rel(Path(record.ocr_text_file), DOCS_DIR)})",
                f"- Layout-aware Markdown: {('[%s](%s)' % (rel(Path(record.layout_markdown_file), DOCS_DIR), rel(Path(record.layout_markdown_file), DOCS_DIR))) if record.layout_markdown_file else 'not available'}",
                f"- Tables: [{rel(Path(record.tables_file), DOCS_DIR)}]({rel(Path(record.tables_file), DOCS_DIR)})",
                f"- Rendered page images: `{rel(Path(record.rendered_pages_dir), DOCS_DIR)}/`",
                f"- Extracted images: `{rel(Path(record.images_dir), DOCS_DIR)}/`",
            ]
        )

    index_path = DOCS_DIR / "pdf-extraction-index.md"
    write_text(index_path, "\n".join(index_lines))
    write_text(OUT_ROOT / "extraction-manifest.json", json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
    print(f"\nIndex: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
