from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from llm_manager.config import DocumentCropConfig
from llm_manager.image_preprocess import (
    correct_document_orientation,
    crop_document,
)


def crop_config(**overrides: object) -> DocumentCropConfig:
    values: dict[str, object] = {
        "enabled": True,
        "perspective_correction": True,
        "detection_max_side": 1000,
        "min_area_ratio": 0.10,
        "max_area_ratio": 0.98,
        "min_rectangularity": 0.65,
        "max_outside_edge_ratio": 0.15,
        "min_aspect_ratio": 1.15,
        "max_aspect_ratio": 3.8,
        "padding_ratio": 0.04,
        "jpeg_quality": 95,
    }
    values.update(overrides)
    return DocumentCropConfig(**values)  # type: ignore[arg-type]


def test_document_crop_detects_large_rectangular_document(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "crop.jpg"
    image = Image.new("RGB", (1200, 900), (190, 190, 190))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 240, 1080, 720), fill="white", outline="black", width=16)
    for y in range(310, 650, 55):
        draw.line((210, y, 920, y), fill=(70, 70, 70), width=5)
    image.save(source, quality=95)

    result = crop_document(source, output, crop_config())

    assert result.applied is True
    assert result.path == output.resolve()
    assert result.confidence >= 0.65
    assert result.original_size == (1200, 900)
    assert result.output_size is not None
    assert result.output_size[0] > result.output_size[1]
    assert output.is_file()


def test_document_crop_falls_back_for_image_without_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "blank.jpg"
    output = tmp_path / "crop.jpg"
    Image.new("RGB", (1000, 800), "white").save(source)

    result = crop_document(source, output, crop_config())

    assert result.applied is False
    assert result.path == source.resolve()
    assert result.reason == "no_trusted_document_boundary"
    assert not output.exists()


def test_document_crop_falls_back_when_another_document_is_outside_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multiple.jpg"
    output = tmp_path / "crop.jpg"
    image = Image.new("RGB", (1400, 900), (185, 185, 185))
    draw = ImageDraw.Draw(image)
    for left in (80, 760):
        draw.rectangle(
            (left, 230, left + 560, 610),
            fill="white",
            outline="black",
            width=16,
        )
        for y in range(300, 560, 55):
            draw.line((left + 70, y, left + 480, y), fill="black", width=5)
    image.save(source, quality=95)

    result = crop_document(source, output, crop_config())

    assert result.applied is False
    assert result.path == source.resolve()
    assert result.reason == "multiple_document_boundaries"
    assert not output.exists()


def test_document_crop_disabled_returns_original(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (800, 600), "white").save(source)

    result = crop_document(
        source,
        tmp_path / "crop.jpg",
        crop_config(enabled=False),
    )

    assert result.applied is False
    assert result.path == source.resolve()
    assert result.reason == "disabled"


def test_document_orientation_rotates_before_multimodal_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "sideways.jpg"
    output = tmp_path / "oriented.jpg"
    Image.new("RGB", (800, 1200), "white").save(source)
    monkeypatch.setattr(
        "llm_manager.image_preprocess.detect_document_orientation",
        lambda _path: {
            "angle": 90,
            "confidence": 0.98,
            "source": "test_orientation_model",
            "error": None,
        },
    )

    result = correct_document_orientation(source, output)

    assert result.applied is True
    assert result.angle == 90
    assert result.path == output.resolve()
    assert result.original_size == (800, 1200)
    assert result.output_size == (1200, 800)
    with Image.open(output) as corrected:
        assert corrected.size == (1200, 800)


def test_document_orientation_keeps_upright_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "upright.jpg"
    output = tmp_path / "oriented.jpg"
    Image.new("RGB", (1200, 800), "white").save(source)
    monkeypatch.setattr(
        "llm_manager.image_preprocess.detect_document_orientation",
        lambda _path: {
            "angle": 0,
            "confidence": 0.99,
            "source": "test_orientation_model",
            "error": None,
        },
    )

    result = correct_document_orientation(source, output)

    assert result.applied is False
    assert result.path == source.resolve()
    assert result.reason == "already_upright"
    assert not output.exists()


def test_document_orientation_bakes_exif_rotation_into_inference_pixels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "exif_sideways.jpg"
    output = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (800, 1200), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    monkeypatch.setattr(
        "llm_manager.image_preprocess.detect_document_orientation",
        lambda _path: {
            "angle": 0,
            "confidence": 1.0,
            "source": "exif_aware_test",
            "error": None,
        },
    )

    result = correct_document_orientation(source, output)

    assert result.applied is True
    assert result.reason == "exif_orientation_normalized"
    assert result.original_size == (800, 1200)
    assert result.output_size == (1200, 800)
    with Image.open(output) as corrected:
        assert corrected.size == (1200, 800)
        assert corrected.getexif().get(274) is None


def test_document_orientation_failure_falls_back_to_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1200, 800), "white").save(source)
    monkeypatch.setattr(
        "llm_manager.image_preprocess.detect_document_orientation",
        lambda _path: {
            "angle": 0,
            "confidence": None,
            "source": "fallback",
            "error": "model unavailable",
        },
    )

    result = correct_document_orientation(source, tmp_path / "oriented.jpg")

    assert result.applied is False
    assert result.path == source.resolve()
    assert result.reason == "detection_failed:model unavailable"
