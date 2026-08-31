from pathlib import Path


WEB_DEMO = Path(__file__).resolve().parents[2] / "web_demo"


def test_application_page_exposes_three_document_columns() -> None:
    html = (WEB_DEMO / "application.html").read_text(encoding="utf-8")

    assert html.count("data-document-column=") == 3
    assert "<h1>申请单三证抽取</h1>" not in html
    assert "<title>申请单识别</title>" in html
    assert "DG12 / DG13" in html
    assert "DG14" in html
    assert "Z002" in html


def test_application_page_calls_application_recognition_api() -> None:
    script = (WEB_DEMO / "application.js").read_text(encoding="utf-8")

    assert "/api/v1/applications/" in script
    assert "data.documents.id_cards" in script
    assert "data.documents.driver_licenses" in script
    assert "data.documents.vehicle_licenses" in script
    assert "application-files.html" in script
    html = (WEB_DEMO / "application.html").read_text(encoding="utf-8")
    assert "查看原始图片" in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html


def test_application_source_gallery_supports_all_groups_and_image_controls() -> None:
    html = (WEB_DEMO / "application-files.html").read_text(encoding="utf-8")
    script = (WEB_DEMO / "application-files.js").read_text(encoding="utf-8")

    assert "source-groups" in html
    assert 'data-viewer-action="zoom-in"' in html
    assert 'data-viewer-action="rotate-left"' in html
    assert 'data-viewer-action="rotate-right"' in html
    assert "/files`" in script
    assert "data.groups.forEach" in script
    assert 'event.key === "ArrowLeft"' in script


def test_application_source_gallery_thumbnails_do_not_crop_images() -> None:
    styles = (WEB_DEMO / "application-files.css").read_text(encoding="utf-8")

    assert "aspect-ratio: 4/3" in styles
    assert "object-fit: scale-down" in styles
    assert "object-position: center" in styles
    assert "?thumbnail=true" in (
        WEB_DEMO / "application-files.js"
    ).read_text(encoding="utf-8")
