"""Lightweight local image preparation before remote OCR."""

from __future__ import annotations

from pathlib import Path

from .orientation import detect_document_orientation

try:
    from PIL import Image, ImageEnhance, ImageOps, ImageStat
except ModuleNotFoundError:  # Optional at runtime; OCR can fall back to originals.
    Image = ImageEnhance = ImageOps = ImageStat = None  # type: ignore[assignment]


_EXIF_ORIENTATION_NAMES = {
    1: "normal",
    2: "mirror_horizontal",
    3: "rotate_180",
    4: "mirror_vertical",
    5: "mirror_horizontal_rotate_270",
    6: "rotate_90",
    7: "mirror_horizontal_rotate_90",
    8: "rotate_270",
}


def normalize_image_orientation(image: object) -> tuple[object, dict[str, object]]:
    """Normalize EXIF orientation once at the image-processing boundary.

    OCR coordinates, enhancement and every later local crop must operate on
    the same pixel orientation.  This function intentionally handles only
    metadata-backed orientation; visual 180-degree detection requires an OCR
    or document-orientation model and is reported separately as ``undetected``.
    """

    if Image is None or ImageOps is None:
        return image, {"source": "unavailable", "rotation": "none"}

    exif_orientation: int | None = None
    try:
        exif_orientation = int(getattr(image, "getexif")().get(274) or 1)
    except (AttributeError, TypeError, ValueError):
        exif_orientation = None
    normalized = ImageOps.exif_transpose(image)
    rotation = _EXIF_ORIENTATION_NAMES.get(exif_orientation or 1, "unknown")
    return normalized, {
        "source": "exif" if exif_orientation not in (None, 1) else "pixel",
        "exif_orientation": exif_orientation or 1,
        "rotation": rotation,
        "visual_detection": "undetected",
    }


def apply_orientation_correction(image: object, angle: int) -> object:
    """Apply Paddle's reported counterclockwise correction angle."""

    angle %= 360
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_90)  # type: ignore[union-attr]
    if angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)  # type: ignore[union-attr]
    if angle == 270:
        return image.transpose(Image.Transpose.ROTATE_270)  # type: ignore[union-attr]
    return image


def prepare_opposite_orientation_retry(
    source_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Rotate a prepared image 180 degrees for a low-confidence OCR retry."""

    if Image is None:
        raise RuntimeError("Pillow 未安装，无法进行方向重试")

    with Image.open(source_path) as image:
        normalized, _ = normalize_image_orientation(image)
        rotated = normalized.convert("RGB").transpose(
            Image.Transpose.ROTATE_180
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(output_path, format="JPEG", quality=95, optimize=True)
        return {
            "rotation": 180,
            "width": rotated.width,
            "height": rotated.height,
            "source": str(source_path),
        }


def _align_image_to_ocr_coordinates(
    image: object,
    coordinate_extent: tuple[float, float] | None,
) -> tuple[object, str]:
    """Rotate portrait pixels when OCR coordinates use a landscape canvas."""

    if not coordinate_extent:
        return image, "none"
    extent_x, extent_y = coordinate_extent
    if (
        extent_x > image.width * 1.02  # type: ignore[attr-defined]
        and extent_x <= image.height * 1.05  # type: ignore[attr-defined]
        and extent_y <= image.width * 1.05  # type: ignore[attr-defined]
    ):
        return image.transpose(Image.Transpose.ROTATE_90), "counterclockwise_90"  # type: ignore[union-attr]
    return image, "none"


def prepare_image(source_path: Path, output_path: Path) -> dict[str, object]:
    """Normalize EXIF orientation and gently improve contrast/sharpness.

    The original file is never modified.  If Pillow cannot decode a file,
    callers can safely fall back to the original path.
    """

    if Image is None:
        raise RuntimeError("Pillow 未安装，跳过图片预处理")

    with Image.open(source_path) as image:
        original_size = image.size
        normalized, orientation = normalize_image_orientation(image)
        # EXIF has already described the physical correction.  Do not run a
        # second classifier on the un-normalized pixels or risk double rotate.
        orientation_detection = (
            {"enabled": False, "angle": 0, "confidence": 1.0, "source": "exif", "error": None}
            if orientation.get("source") == "exif"
            else detect_document_orientation(source_path)
        )
        detected_angle = int(orientation_detection.get("angle") or 0) % 360
        normalized = apply_orientation_correction(normalized, detected_angle)
        orientation = {
            **orientation,
            "detector": orientation_detection,
            "applied_correction": detected_angle,
        }
        normalized = normalized.convert("RGB")
        statistics = ImageStat.Stat(normalized.convert("L"))
        if min(normalized.size) < 1000:
            scale = 1200 / min(normalized.size)
            normalized = normalized.resize(
                (
                    round(normalized.width * scale),
                    round(normalized.height * scale),
                ),
                Image.Resampling.LANCZOS,
            )
        if max(normalized.size) > 3200:
            scale = 3200 / max(normalized.size)
            normalized = normalized.resize(
                (
                    round(normalized.width * scale),
                    round(normalized.height * scale),
                ),
                Image.Resampling.LANCZOS,
            )
        enhanced = ImageOps.autocontrast(normalized, cutoff=1)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.12)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path, format="JPEG", quality=95, optimize=True)
        return {
            "width": original_size[0],
            "height": original_size[1],
            "prepared_width": enhanced.width,
            "prepared_height": enhanced.height,
            "format": image.format,
            "brightness": round(statistics.mean[0], 2),
            "contrast": round(statistics.stddev[0], 2),
            "orientation": orientation,
        }


def prepare_plate_crop(
    source_path: Path,
    output_path: Path,
    bbox: list[float] | tuple[float, float, float, float],
    *,
    coordinate_extent: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Crop and enhance one plate region using OCR-image coordinates."""

    if Image is None:
        raise RuntimeError("Pillow 未安装，无法进行车牌局部复识别")
    if len(bbox) != 4:
        raise ValueError("车牌坐标必须包含 left、top、right、bottom")

    with Image.open(source_path) as image:
        normalized, orientation = normalize_image_orientation(image)
        normalized = normalized.convert("RGB")
        normalized, rotation = _align_image_to_ocr_coordinates(
            normalized,
            coordinate_extent,
        )
        left, top, right, bottom = map(float, bbox)
        if right <= left or bottom <= top:
            raise ValueError("车牌坐标范围无效")

        width = right - left
        height = bottom - top
        pad_x = max(12.0, width * 0.16)
        pad_y = max(10.0, height * 0.55)
        crop_box = (
            max(0, round(left - pad_x)),
            max(0, round(top - pad_y)),
            min(normalized.width, round(right + pad_x)),
            min(normalized.height, round(bottom + pad_y)),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError("车牌裁剪区域超出图片范围")

        cropped = normalized.crop(crop_box)
        # First preserve color edges while enlarging, then enhance in
        # grayscale and enlarge once more. This order is more effective for
        # Z/2, S/5 and similar shape confusions than one large resize.
        base_scale = min(4.0, max(2.0, 320.0 / cropped.height))
        enlarged = cropped.resize(
            (
                max(1, round(cropped.width * base_scale)),
                max(1, round(cropped.height * base_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        color_enhanced = ImageOps.autocontrast(enlarged, cutoff=1)
        color_enhanced = ImageEnhance.Contrast(color_enhanced).enhance(1.35)
        color_enhanced = ImageEnhance.Sharpness(color_enhanced).enhance(1.5)
        enhanced = ImageOps.grayscale(color_enhanced)
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.2)
        enhanced = enhanced.resize(
            (enhanced.width * 2, enhanced.height * 2),
            Image.Resampling.LANCZOS,
        ).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path, format="PNG", optimize=True)
        return {
            "source_width": normalized.width,
            "source_height": normalized.height,
            "bbox": [left, top, right, bottom],
            "crop_box": list(crop_box),
            "crop_width": cropped.width,
            "crop_height": cropped.height,
            "output_width": enhanced.width,
            "output_height": enhanced.height,
            "scale": round(base_scale * 2, 4),
            "coordinate_rotation": rotation,
            "orientation": orientation,
            "mode": "grayscale",
            "contrast_factor": 1.35,
            "sharpness_factor": 2.2,
        }


def prepare_vehicle_field_crop(
    source_path: Path,
    output_path: Path,
    bbox: list[float] | tuple[float, float, float, float],
    *,
    profile: str,
    coordinate_extent: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Crop one vehicle-license field with a field-appropriate profile."""

    if Image is None:
        raise RuntimeError("Pillow 未安装，无法进行字段局部复识别")
    if profile not in {"identifier", "numeric", "text"}:
        raise ValueError(f"不支持的字段增强配置：{profile}")
    if len(bbox) != 4:
        raise ValueError("字段坐标必须包含 left、top、right、bottom")

    with Image.open(source_path) as image:
        normalized, orientation = normalize_image_orientation(image)
        normalized = normalized.convert("RGB")
        normalized, rotation = _align_image_to_ocr_coordinates(
            normalized,
            coordinate_extent,
        )
        left, top, right, bottom = map(float, bbox)
        if right <= left or bottom <= top:
            raise ValueError("字段坐标范围无效")

        width = right - left
        height = bottom - top
        if profile == "text":
            # The address region locator already keeps the wrapped line.
            # Additional padding pulls owner/model rows into the crop.
            pad_x = 0.0
            pad_y = 0.0
        else:
            pad_x = max(10.0, width * 0.06)
            pad_y = max(8.0, height * 0.28)
        crop_box = (
            max(0, round(left - pad_x)),
            max(0, round(top - pad_y)),
            min(normalized.width, round(right + pad_x)),
            min(normalized.height, round(bottom + pad_y)),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError("字段裁剪区域超出图片范围")

        cropped = normalized.crop(crop_box)
        target_height = {
            "identifier": 320.0,
            "numeric": 480.0,
        }.get(profile)
        scale = (
            1.0
            if profile == "text"
            else min(6.0, max(1.5, target_height / cropped.height))
        )
        enlarged = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

        if profile == "text":
            # Vehicle-license security backgrounds have strong chromatic
            # patterns. The blue channel retained address glyph strokes most
            # consistently in the address-error benchmark.
            enhanced = enlarged.getchannel("B").convert("RGB")
            mode = "blue_channel"
            contrast_factor = 1.0
            sharpness_factor = 1.0
        elif profile == "identifier":
            # Preserve color and use restrained sharpening for long codes.
            enhanced = ImageOps.autocontrast(enlarged, cutoff=1)
            contrast_factor = 1.15
            sharpness_factor = 1.3
            enhanced = ImageEnhance.Contrast(enhanced).enhance(
                contrast_factor
            )
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(
                sharpness_factor
            )
            mode = "color"
        else:
            # Dates, masses and dimensions benefit from stronger grayscale
            # separation because their glyph shapes are less color-dependent.
            enhanced = ImageOps.grayscale(enlarged)
            enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.25)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.8).convert(
                "RGB"
            )
            mode = "grayscale"
            contrast_factor = 1.25
            sharpness_factor = 1.8

        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path, format="PNG", optimize=True)
        return {
            "source_width": normalized.width,
            "source_height": normalized.height,
            "bbox": [left, top, right, bottom],
            "crop_box": list(crop_box),
            "crop_width": cropped.width,
            "crop_height": cropped.height,
            "output_width": enhanced.width,
            "output_height": enhanced.height,
            "scale": round(scale, 4),
            "coordinate_rotation": rotation,
            "orientation": orientation,
            "profile": profile,
            "mode": mode,
            "contrast_factor": contrast_factor,
            "sharpness_factor": sharpness_factor,
        }


def prepare_driver_class_contrast(
    source_path: Path,
    output_path: Path,
    *,
    channel: str = "B",
    threshold: int = 150,
) -> dict[str, object]:
    """Create a high-contrast class crop for faint C1/C1D glyphs."""

    if Image is None:
        raise RuntimeError("Pillow 未安装，无法进行准驾车型增强复识别")
    if channel not in {"R", "G", "B"}:
        raise ValueError(f"不支持的准驾车型颜色通道：{channel}")
    if not 0 <= threshold <= 255:
        raise ValueError("准驾车型二值化阈值必须在 0 到 255 之间")
    with Image.open(source_path) as image:
        normalized, orientation = normalize_image_orientation(image)
        rgb = normalized.convert("RGB")
        selected = ImageOps.autocontrast(rgb.getchannel(channel), cutoff=1)
        enhanced = selected.point(
            lambda value: 255 if value > threshold else 0
        ).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path, format="PNG", optimize=True)
        return {
            "source_width": rgb.width,
            "source_height": rgb.height,
            "output_width": enhanced.width,
            "output_height": enhanced.height,
            "orientation": orientation,
            "profile": (
                "driver_class_high_contrast"
                if channel == "B" and threshold == 150
                else "driver_class_channel_contrast"
            ),
            "mode": f"{channel.lower()}_channel_binary",
            "threshold": threshold,
        }
