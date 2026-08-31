"""Local document-boundary detection for multimodal model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DocumentCropConfig

from ocr_manager.orientation import detect_document_orientation
from ocr_manager.preprocess import apply_orientation_correction

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # Fail open when optional image dependencies are absent.
    cv2 = np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:  # Fail open when Pillow is unavailable.
    Image = ImageOps = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DocumentCropResult:
    """Outcome of one safe crop attempt."""

    path: Path
    applied: bool
    confidence: float
    reason: str
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    corners: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class DocumentOrientationResult:
    """Outcome of document orientation detection and safe correction."""

    path: Path
    applied: bool
    angle: int
    confidence: float | None
    source: str
    reason: str
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None


def correct_document_orientation(
    source_path: Path,
    output_path: Path,
    detection: dict[str, Any] | None = None,
) -> DocumentOrientationResult:
    """Rotate a document to its natural reading direction before inference.

    Detection and image processing failures deliberately fall back to the
    original path so optional orientation correction never blocks extraction.
    """

    source_path = source_path.resolve()
    if Image is None:
        return DocumentOrientationResult(
            source_path, False, 0, None, "fallback", "pillow_unavailable"
        )
    try:
        with Image.open(source_path) as source_image:
            exif_orientation = int(source_image.getexif().get(274) or 1)
    except Exception:
        exif_orientation = 1
    # EXIF is authoritative for metadata-backed camera rotation. Running a
    # visual classifier against the raw matrix as well can apply the same
    # correction twice.
    detection = (
        {
            "angle": 0,
            "confidence": 1.0,
            "source": "exif",
            "error": None,
        }
        if exif_orientation != 1
        else detection or detect_document_orientation(source_path)
    )
    angle = int(detection.get("angle") or 0) % 360
    confidence_value = detection.get("confidence")
    confidence = (
        float(confidence_value)
        if isinstance(confidence_value, (int, float))
        else None
    )
    source = str(detection.get("source") or "fallback")
    error = detection.get("error")
    if error:
        return DocumentOrientationResult(
            source_path,
            False,
            0,
            confidence,
            source,
            f"detection_failed:{error}",
        )
    if angle not in {90, 180, 270}:
        if angle != 0:
            return DocumentOrientationResult(
                source_path,
                False,
                angle,
                confidence,
                source,
                "unsupported_angle",
            )
    try:
        with Image.open(source_path) as image:
            original_size = image.size
            # OCR libraries and browsers commonly honor EXIF while Pillow's
            # raw pixel matrix does not. Bake the metadata transform into the
            # saved inference image so OCR boxes and later crops share one
            # coordinate system.
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            exif_applied = exif_orientation != 1
            if angle == 0 and not exif_applied:
                return DocumentOrientationResult(
                    source_path,
                    False,
                    0,
                    confidence,
                    source,
                    "already_upright",
                    original_size=original_size,
                    output_size=original_size,
                )
            corrected = apply_orientation_correction(normalized, angle)
            output_path = output_path.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            corrected.save(output_path, format="JPEG", quality=98, subsampling=0)
            return DocumentOrientationResult(
                output_path,
                True,
                angle,
                confidence,
                source,
                (
                    "exif_and_orientation_corrected"
                    if exif_applied and angle
                    else "exif_orientation_normalized"
                    if exif_applied
                    else "orientation_corrected"
                ),
                original_size=original_size,
                output_size=corrected.size,
            )
    except Exception as exc:
        return DocumentOrientationResult(
            source_path,
            False,
            angle,
            confidence,
            source,
            f"fallback:{type(exc).__name__}:{exc}",
        )


def _order_points(points: Any) -> Any:
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _expand_points(points: Any, ratio: float, width: int, height: int) -> Any:
    if ratio <= 0:
        return points
    center = points.mean(axis=0)
    expanded = center + (points - center) * (1.0 + ratio * 2.0)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype("float32")


def _warp(image: Any, points: Any) -> Any:
    top_left, top_right, bottom_right, bottom_left = points
    target_width = round(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    target_height = round(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if target_width < 320 or target_height < 180:
        raise ValueError("检测到的证件区域尺寸过小")
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_document(
    source_path: Path,
    output_path: Path,
    config: DocumentCropConfig,
) -> DocumentCropResult:
    """Detect, validate and rectify a document, falling back to the source."""

    source_path = source_path.resolve()
    if not config.enabled:
        return DocumentCropResult(source_path, False, 0.0, "disabled")
    if cv2 is None or np is None:
        return DocumentCropResult(
            source_path,
            False,
            0.0,
            "opencv_unavailable",
        )
    try:
        encoded = np.fromfile(str(source_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片解码失败")
        height, width = image.shape[:2]
        original_size = (width, height)
        scale = min(1.0, config.detection_max_side / max(width, height))
        if scale < 1.0:
            detector = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detector = image

        gray = cv2.cvtColor(detector, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        lower = max(20, round(median * 0.55))
        upper = max(lower + 30, min(255, round(median * 1.35)))
        edges = cv2.Canny(blurred, lower, upper)
        kernel_size = max(5, round(max(detector.shape[:2]) / 180))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        connected = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        detector_area = float(detector.shape[0] * detector.shape[1])
        candidates: list[tuple[float, float, float, Any]] = []
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            rectangle = cv2.minAreaRect(contour)
            rect_width, rect_height = rectangle[1]
            rectangle_area = float(rect_width * rect_height)
            if rectangle_area <= 0:
                continue
            area_ratio = rectangle_area / detector_area
            aspect_ratio = max(rect_width, rect_height) / max(
                1.0,
                min(rect_width, rect_height),
            )
            rectangularity = min(1.0, contour_area / rectangle_area)
            if not (
                config.min_area_ratio <= area_ratio <= config.max_area_ratio
                and config.min_aspect_ratio
                <= aspect_ratio
                <= config.max_aspect_ratio
                and rectangularity >= config.min_rectangularity
            ):
                continue
            score = (
                min(1.0, area_ratio / 0.35) * 0.45
                + rectangularity * 0.45
                + 0.10
            )
            candidates.append(
                (score, area_ratio, rectangularity, rectangle)
            )
        if not candidates:
            return DocumentCropResult(
                source_path,
                False,
                0.0,
                "no_trusted_document_boundary",
                original_size=original_size,
            )

        if len(candidates) > 1:
            confidence = max(item[0] for item in candidates)
            return DocumentCropResult(
                source_path,
                False,
                round(min(1.0, confidence), 4),
                "multiple_document_boundaries",
                original_size=original_size,
            )

        score, _area_ratio, _rectangularity, rectangle = max(
            candidates,
            key=lambda item: item[0],
        )
        detector_box = cv2.boxPoints(rectangle).astype("int32")
        detector_mask = np.zeros_like(edges)
        cv2.fillConvexPoly(detector_mask, detector_box, 255)
        inside_edge_count = int(np.count_nonzero(edges[detector_mask > 0]))
        outside_edge_count = int(np.count_nonzero(edges[detector_mask == 0]))
        outside_edge_ratio = outside_edge_count / max(
            1,
            inside_edge_count + outside_edge_count,
        )
        if outside_edge_ratio > config.max_outside_edge_ratio:
            return DocumentCropResult(
                source_path,
                False,
                round(min(1.0, score), 4),
                "content_detected_outside_boundary",
                original_size=original_size,
            )
        points = cv2.boxPoints(rectangle).astype("float32") / scale
        points = _order_points(points)
        points = _expand_points(
            points,
            config.padding_ratio,
            width,
            height,
        )
        if config.perspective_correction:
            cropped = _warp(image, points)
        else:
            left = max(0, int(np.floor(points[:, 0].min())))
            top = max(0, int(np.floor(points[:, 1].min())))
            right = min(width, int(np.ceil(points[:, 0].max())))
            bottom = min(height, int(np.ceil(points[:, 1].max())))
            if right - left < 320 or bottom - top < 180:
                raise ValueError("检测到的证件区域尺寸过小")
            cropped = image[top:bottom, left:right]

        output_height, output_width = cropped.shape[:2]
        if output_width * output_height < width * height * config.min_area_ratio:
            raise ValueError("裁剪结果面积过小")
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success, output = cv2.imencode(
            ".jpg",
            cropped,
            [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
        )
        if not success:
            raise ValueError("裁剪图编码失败")
        output.tofile(str(output_path))
        return DocumentCropResult(
            output_path,
            True,
            round(min(1.0, score), 4),
            "document_boundary_detected",
            original_size=original_size,
            output_size=(output_width, output_height),
            corners=tuple(
                (round(float(x), 2), round(float(y), 2)) for x, y in points
            ),
        )
    except Exception as exc:  # A preprocessing failure must not block inference.
        return DocumentCropResult(
            source_path,
            False,
            0.0,
            f"fallback:{type(exc).__name__}:{exc}",
        )
