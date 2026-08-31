"""Render PDF pages to images for the shared recognition pipelines."""

from __future__ import annotations

from pathlib import Path


MAX_PDF_PAGES = 100
PDF_RENDER_DPI = 200


def render_pdf_pages(
    source_path: Path,
    output_dir: Path,
    *,
    max_pages: int = MAX_PDF_PAGES,
    dpi: int = PDF_RENDER_DPI,
) -> list[Path]:
    """Render every PDF page to JPEG, rejecting empty or oversized PDFs."""

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - deployment configuration.
        raise RuntimeError("PDF 转图片依赖未安装：pypdfium2") from exc

    try:
        document = pdfium.PdfDocument(str(source_path))
    except Exception as exc:
        raise ValueError(f"PDF 无法解析：{source_path.name}：{exc}") from exc
    try:
        page_count = len(document)
        if page_count == 0:
            raise ValueError(f"PDF 无有效页面：{source_path.name}")
        if page_count > max_pages:
            raise ValueError(
                f"PDF 页数超过 {max_pages} 页限制：{source_path.name}（{page_count} 页）"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []
        scale = dpi / 72
        for page_index in range(page_count):
            page = document[page_index]
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB")
                output_path = output_dir / (
                    f"{source_path.stem}_第{page_index + 1:03d}页.jpg"
                )
                image.save(output_path, format="JPEG", quality=95, subsampling=0)
                rendered.append(output_path)
            finally:
                bitmap.close()
                page.close()
        return rendered
    finally:
        document.close()


def recognition_images(source_path: Path, output_dir: Path) -> list[Path]:
    """Return one image input or the rendered pages of one PDF input."""

    if source_path.suffix.lower() == ".pdf":
        return render_pdf_pages(source_path, output_dir)
    return [source_path]
