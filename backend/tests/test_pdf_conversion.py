from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from llm_manager.pdf_conversion import render_pdf_pages


def create_pdf(path: Path, pages: int) -> None:
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            document.new_page(595, 842)
        document.save(path)
    finally:
        document.close()


def test_render_pdf_pages_creates_ordered_jpegs(tmp_path: Path) -> None:
    source = tmp_path / "material.pdf"
    create_pdf(source, 2)

    pages = render_pdf_pages(source, tmp_path / "pages", dpi=72)

    assert [path.name for path in pages] == [
        "material_第001页.jpg",
        "material_第002页.jpg",
    ]
    with Image.open(pages[0]) as image:
        assert image.size == (595, 842)


def test_render_pdf_pages_rejects_more_than_100_pages(tmp_path: Path) -> None:
    source = tmp_path / "large.pdf"
    create_pdf(source, 101)

    with pytest.raises(ValueError, match="超过 100 页限制"):
        render_pdf_pages(source, tmp_path / "pages")
