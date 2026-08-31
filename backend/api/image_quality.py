"""Conservative, PP-OCR-aware quality gate for mobile document photos."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageQualityIssue:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ImageQualityReport:
    issues: tuple[ImageQualityIssue, ...]
    metrics: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return not self.issues


def _decode_image(content: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as image:
            # PP-OCR normalizes EXIF orientation before inference, so quality
            # checks must inspect the same visual coordinate space. Otherwise
            # OCR boxes can be clipped against swapped image dimensions.
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("无法读取图片，请重新拍摄或选择 JPG/PNG 图片") from exc


def _glare_component_metrics(
    mask: np.ndarray,
    denominator: int,
) -> tuple[float, float, int]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return 0.0, 0.0, 0
    largest_ratio = 0.0
    elongated_area = 0
    elongated_count = 0
    for left, top, width, height, area in stats[1:]:
        del left, top
        ratio = float(area) / float(max(1, denominator))
        largest_ratio = max(largest_ratio, ratio)
        elongation = max(width, height) / max(1.0, min(width, height))
        if elongation >= 3.5 and ratio >= 0.0008:
            elongated_area += int(area)
            elongated_count += 1
    return (
        largest_ratio,
        float(elongated_area) / float(max(1, denominator)),
        elongated_count,
    )


def _document_boundary(gray: np.ndarray) -> dict[str, Any] | None:
    """Find one card-like boundary without treating absence as automatic failure."""

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = max(20, round(median * 0.55))
    upper = max(lower + 30, min(255, round(median * 1.35)))
    edges = cv2.Canny(blurred, lower, upper)
    side = max(gray.shape)
    kernel_size = max(5, round(side / 160))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(gray.shape[0] * gray.shape[1])
    candidates: list[tuple[float, float, float, Any, Any]] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        rectangle_area = float(rect_width * rect_height)
        if contour_area <= 0 or rectangle_area <= 0:
            continue
        area_ratio = rectangle_area / image_area
        aspect_ratio = max(rect_width, rect_height) / max(
            1.0, min(rect_width, rect_height)
        )
        rectangularity = min(1.0, contour_area / rectangle_area)
        if not (
            0.04 <= area_ratio <= 1.05
            and 1.20 <= aspect_ratio <= 2.15
            and rectangularity >= 0.55
        ):
            continue
        score = area_ratio * 0.65 + rectangularity * 0.35
        points = cv2.boxPoints(rectangle)
        candidates.append(
            (score, area_ratio, rectangularity, rectangle, points)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    distinct: list[tuple[float, float, float, Any, Any]] = []
    for candidate in candidates:
        center = candidate[3][0]
        if any(
            cv2.pointPolygonTest(existing[4], center, False) >= 0
            for existing in distinct
        ):
            continue
        distinct.append(candidate)
    _score, area_ratio, rectangularity, _rectangle, points = distinct[0]
    height, width = gray.shape
    margin = max(3.0, min(width, height) * 0.006)
    touches_frame = bool(
        np.any(points[:, 0] <= margin)
        or np.any(points[:, 1] <= margin)
        or np.any(points[:, 0] >= width - 1 - margin)
        or np.any(points[:, 1] >= height - 1 - margin)
    )
    return {
        "area_ratio": round(area_ratio, 4),
        "rectangularity": round(rectangularity, 4),
        "touches_frame": touches_frame,
        "candidate_count": len(distinct),
        "_points": points,
    }


def _colored_document_regions(
    image: np.ndarray,
    document_type: str,
) -> list[np.ndarray]:
    """Return separate green/blue-green licence page rectangles."""

    if document_type not in {"driver_license", "vehicle_license"}:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (28, 18, 45), (108, 255, 255))
    side = max(mask.shape)
    kernel_size = max(9, round(side / 75))
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(mask.size)
    regions: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        rectangle_area = float(rect_width * rect_height)
        if contour_area <= 0 or rectangle_area <= 0:
            continue
        area_ratio = rectangle_area / image_area
        aspect_ratio = max(rect_width, rect_height) / max(
            1.0, min(rect_width, rect_height)
        )
        rectangularity = min(1.0, contour_area / rectangle_area)
        if (
            area_ratio >= 0.055
            and 1.15 <= aspect_ratio <= 2.55
            and rectangularity >= 0.32
        ):
            regions.append(
                (rectangle_area, cv2.boxPoints(rectangle).astype(np.float32))
            )
    regions.sort(key=lambda item: item[0], reverse=True)
    return [points for _area, points in regions]


def _colored_document_candidate_count(
    image: np.ndarray,
    document_type: str,
) -> int:
    return len(_colored_document_regions(image, document_type))


def _looks_like_driver_license_booklet(regions: list[np.ndarray]) -> bool:
    """Recognize aligned main/secondary pages in either phone orientation."""

    if len(regions) != 2:
        return False
    first, second = [
        cv2.boundingRect(points.astype(np.int32)) for points in regions
    ]
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    height_ratio = min(first_height, second_height) / max(
        first_height,
        second_height,
    )
    width_ratio = min(first_width, second_width) / max(
        first_width,
        second_width,
    )
    vertical_overlap = max(
        0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    horizontal_overlap = max(
        0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    vertical_overlap_ratio = vertical_overlap / max(
        1,
        min(first_height, second_height),
    )
    horizontal_overlap_ratio = horizontal_overlap / max(
        1,
        min(first_width, second_width),
    )
    horizontal_gap = max(
        0,
        max(first_x, second_x)
        - min(first_x + first_width, second_x + second_width),
    )
    vertical_gap = max(
        0,
        max(first_y, second_y)
        - min(first_y + first_height, second_y + second_height),
    )
    side_by_side = (
        vertical_overlap_ratio >= 0.78
        and horizontal_gap <= max(first_width, second_width) * 0.16
    )
    rotated_stack = (
        horizontal_overlap_ratio >= 0.60
        and vertical_gap <= max(first_height, second_height) * 0.20
    )
    return (
        height_ratio >= 0.72
        and width_ratio >= 0.62
        and (side_by_side or rotated_stack)
    )


def _order_rectangle_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _document_roi(
    image: np.ndarray,
    boundary: dict[str, Any] | None,
    document_type: str,
) -> tuple[np.ndarray, str]:
    points: np.ndarray | None = None
    source = "full_image"
    if boundary is not None:
        points = np.asarray(boundary["_points"], dtype=np.float32)
        source = "document_boundary"
    else:
        colored_regions = _colored_document_regions(image, document_type)
        if colored_regions:
            points = colored_regions[0]
            source = "colored_document"
    if points is None:
        roi = image
    else:
        top_left, top_right, bottom_right, bottom_left = (
            _order_rectangle_points(points)
        )
        width = round(
            max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
        )
        height = round(
            max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )
        )
        if width >= 240 and height >= 160:
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(
                np.array(
                    [top_left, top_right, bottom_right, bottom_left],
                    dtype=np.float32,
                ),
                destination,
            )
            roi = cv2.warpPerspective(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            roi = image
            source = "full_image"
    height, width = roi.shape[:2]
    trim_x = round(width * 0.04)
    trim_y = round(height * 0.04)
    if width - trim_x * 2 >= 200 and height - trim_y * 2 >= 120:
        roi = roi[trim_y : height - trim_y, trim_x : width - trim_x]
    scale = min(1.0, 960.0 / max(roi.shape[:2]))
    if scale < 1.0:
        roi = cv2.resize(
            roi,
            (round(roi.shape[1] * scale), round(roi.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return roi, source


def _blur_metrics(
    image: np.ndarray,
) -> dict[str, Any]:
    scale = min(1.0, 960.0 / max(image.shape[:2]))
    roi = (
        cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else image
    )
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    noise_kernel = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
        dtype=np.float32,
    )
    noise_response = cv2.filter2D(gray, cv2.CV_32F, noise_kernel)
    noise_sigma = (
        float(np.abs(noise_response[1:-1, 1:-1]).sum())
        * np.sqrt(np.pi / 2.0)
        / max(1.0, 6.0 * (gray.shape[0] - 2) * (gray.shape[1] - 2))
    )
    denoised = (
        cv2.GaussianBlur(gray, (5, 5), 1.2)
        if noise_sigma >= 1.0
        else cv2.GaussianBlur(gray, (3, 3), 0)
    )
    laplacian = cv2.Laplacian(denoised, cv2.CV_32F)
    gradient_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    gradient_square = gradient_x * gradient_x + gradient_y * gradient_y
    coarse = cv2.resize(
        gray,
        (max(1, gray.shape[1] // 2), max(1, gray.shape[0] // 2)),
        interpolation=cv2.INTER_AREA,
    )
    coarse = cv2.GaussianBlur(coarse, (3, 3), 0)
    coarse_gradient_x = cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3)
    coarse_gradient_y = cv2.Sobel(coarse, cv2.CV_32F, 0, 1, ksize=3)

    tile_laplacians: list[float] = []
    tile_tenengrads: list[float] = []
    rows, columns = 4, 6
    height, width = denoised.shape
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left = round(column * width / columns)
            right = round((column + 1) * width / columns)
            tile = denoised[top:bottom, left:right]
            if tile.size < 64 or float(tile.std()) < 10:
                continue
            tile_laplacians.append(
                float(cv2.Laplacian(tile, cv2.CV_32F).var())
            )
            tile_gradient_x = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
            tile_gradient_y = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
            tile_tenengrads.append(
                float((tile_gradient_x**2 + tile_gradient_y**2).mean())
            )

    sharp_tiles = sum(
        laplacian_value >= 35 and tenengrad_value >= 900
        for laplacian_value, tenengrad_value in zip(
            tile_laplacians,
            tile_tenengrads,
        )
    )
    informative_tiles = len(tile_laplacians)
    mean_gradient_x = float(np.mean(np.abs(gradient_x)))
    mean_gradient_y = float(np.mean(np.abs(gradient_y)))
    motion_anisotropy = max(mean_gradient_x, mean_gradient_y) / max(
        1e-6,
        min(mean_gradient_x, mean_gradient_y),
    )
    return {
        "roi_source": "full_image",
        "roi_width": int(width),
        "roi_height": int(height),
        "laplacian_variance": round(float(laplacian.var()), 2),
        "tenengrad": round(float(gradient_square.mean()), 2),
        "noise_sigma": round(float(noise_sigma), 4),
        "coarse_laplacian_variance": round(
            float(cv2.Laplacian(coarse, cv2.CV_32F).var()),
            2,
        ),
        "coarse_tenengrad": round(
            float(
                (
                    coarse_gradient_x * coarse_gradient_x
                    + coarse_gradient_y * coarse_gradient_y
                ).mean()
            ),
            2,
        ),
        "median_tile_laplacian": round(
            float(np.median(tile_laplacians)) if tile_laplacians else 0.0,
            2,
        ),
        "median_tile_tenengrad": round(
            float(np.median(tile_tenengrads)) if tile_tenengrads else 0.0,
            2,
        ),
        "informative_tiles": informative_tiles,
        "sharp_tile_ratio": round(
            sharp_tiles / max(1, informative_tiles),
            4,
        ),
        "motion_anisotropy": round(motion_anisotropy, 4),
    }


def _is_too_blurry(metrics: dict[str, Any]) -> bool:
    laplacian = float(metrics["laplacian_variance"])
    tenengrad = float(metrics["tenengrad"])
    median_laplacian = float(metrics["median_tile_laplacian"])
    sharp_tile_ratio = float(metrics["sharp_tile_ratio"])
    coarse_laplacian = float(metrics["coarse_laplacian_variance"])

    # Deliberately permissive: only reject when the entire frame has almost
    # no usable detail at either fine or coarse scale. Moderate defocus,
    # motion streaks and noisy phone photos continue to OCR.
    return (
        laplacian < 2.0
        and tenengrad < 800
        and median_laplacian < 2.0
        and coarse_laplacian < 8.0
        and sharp_tile_ratio < 0.02
    )


def _is_severe_glare(
    glare_ratio: float,
    largest_glare_ratio: float,
    elongated_glare_ratio: float,
) -> bool:
    """Reject only glare that covers an obviously large document area."""

    large_area_glare = (
        glare_ratio >= 0.12
        and (
            largest_glare_ratio >= 0.08
            or elongated_glare_ratio >= 0.06
        )
    )
    extreme_glare = glare_ratio >= 0.20 or largest_glare_ratio >= 0.16
    return large_area_glare or extreme_glare


def _normalized_ocr_tokens(
    ppocr_result: dict[str, Any] | None,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Return valid PP tokens clipped to the prepared image coordinates."""

    if not isinstance(ppocr_result, dict):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in ppocr_result.get("tokens") or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue
        left, top, right, bottom = (float(value) for value in bbox)
        left = max(0.0, min(float(image_width), left))
        right = max(0.0, min(float(image_width), right))
        top = max(0.0, min(float(image_height), top))
        bottom = max(0.0, min(float(image_height), bottom))
        if right - left < 2 or bottom - top < 2:
            continue
        raw_score = raw.get("score")
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float)) and 0 <= raw_score <= 1
            else None
        )
        text = str(raw.get("text") or "").strip()
        normalized.append(
            {
                "text": text,
                "text_length": max(1, len(text)),
                "score": score,
                "bbox": (left, top, right, bottom),
            }
        )
    return normalized


def _compact_ocr_text(value: object) -> str:
    return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", str(value or "").upper())


def _id_card_side(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> str | None:
    raw_side = _compact_ocr_text(
        ppocr_result.get("quality_side") or ppocr_result.get("document_side")
    )
    if raw_side in {"DG12", "FRONT", "人像面", "正面"}:
        return "front"
    if raw_side in {"DG13", "BACK", "国徽面", "反面"}:
        return "back"

    texts = [_compact_ocr_text(token.get("text")) for token in tokens]
    front_score = sum(
        any(marker in text for marker in ("姓名", "公民身份号码", "身份证号码"))
        or bool(re.fullmatch(r"\d{17}[0-9X]", text))
        for text in texts
    )
    back_score = sum(
        any(marker in text for marker in ("居民身份证", "中华人民共和国", "签发机关", "有效期限"))
        for text in texts
    )
    if front_score > back_score and front_score:
        return "front"
    if back_score > front_score and back_score:
        return "back"
    return None


def _id_card_text_scope(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Limit ID-card quality checks to the semantic text band.

    This deliberately does not project a fixed template. OCR anchors only
    establish the upper/lower text range, so perspective and card scale do not
    make every downstream region drift together.
    """

    side = _id_card_side(ppocr_result, tokens)
    if side is None:
        return tokens, {
            "side": "unknown",
            "strategy": "all_ocr_tokens",
            "reliable": False,
            "anchors_missing": False,
            "original_token_count": len(tokens),
            "selected_token_count": len(tokens),
        }

    def text(token: dict[str, Any]) -> str:
        return _compact_ocr_text(token.get("text"))

    def is_id_number(value: str) -> bool:
        return bool(re.fullmatch(r"\d{17}[0-9X]", value))

    if side == "front":
        top_anchors = [token for token in tokens if "姓名" in text(token)]
        bottom_anchors = [
            token
            for token in tokens
            if any(
                marker in text(token)
                for marker in ("公民身份号码", "公民身份证号码", "身份证号码", "身份号码", "证件号码")
            )
            or is_id_number(text(token))
        ]
        related_markers = ("姓名", "性别", "民族", "出生", "住址", "身份", "号码")
        related = [
            token
            for token in tokens
            if any(marker in text(token) for marker in related_markers)
            or is_id_number(text(token))
        ]
    else:
        top_anchors = [
            token
            for token in tokens
            if "中华人民共和国" in text(token) or "居民身份证" in text(token)
        ]
        bottom_anchors = [
            token
            for token in tokens
            if "有效期限" in text(token)
            or bool(re.search(r"\d{4}\d{2}\d{2}.*\d{4}\d{2}\d{2}", text(token)))
        ]
        related_markers = ("中华人民共和国", "居民身份证", "签发机关", "有效期限", "公安局")
        related = [
            token
            for token in tokens
            if any(marker in text(token) for marker in related_markers)
            or bool(re.search(r"\d{4}\d{2}\d{2}", text(token)))
        ]

    anchors_missing = not top_anchors and not bottom_anchors
    if anchors_missing:
        return [], {
            "side": side,
            "strategy": "anchors_missing",
            "reliable": False,
            "anchors_missing": True,
            "top_anchor_count": 0,
            "bottom_anchor_count": 0,
            "original_token_count": len(tokens),
            "selected_token_count": 0,
        }

    boundary_tokens = [*top_anchors, *bottom_anchors]
    strategy = "semantic_band"
    reliable = bool(top_anchors and bottom_anchors)
    if not reliable:
        strategy = "partial_semantic_band"
        boundary_tokens = related or boundary_tokens

    heights = [
        float(token["bbox"][3]) - float(token["bbox"][1])
        for token in boundary_tokens
    ]
    median_height = float(np.median(heights)) if heights else 1.0
    band_top = min(float(token["bbox"][1]) for token in boundary_tokens) - 1.5 * median_height
    band_bottom = max(float(token["bbox"][3]) for token in boundary_tokens) + 1.5 * median_height
    selected = [
        token
        for token in tokens
        if band_top
        <= (float(token["bbox"][1]) + float(token["bbox"][3])) / 2.0
        <= band_bottom
    ]
    return selected, {
        "side": side,
        "strategy": strategy,
        "reliable": reliable,
        "anchors_missing": False,
        "top_anchor_count": len(top_anchors),
        "bottom_anchor_count": len(bottom_anchors),
        "band_top": round(max(0.0, band_top), 2),
        "band_bottom": round(band_bottom, 2),
        "original_token_count": len(tokens),
        "selected_token_count": len(selected),
    }


def _driver_license_side(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> str | None:
    """Distinguish a driving-license main page, secondary page, or both."""

    raw_side = _compact_ocr_text(
        ppocr_result.get("quality_side") or ppocr_result.get("document_side")
    )
    if raw_side in {"MAIN", "PRIMARY", "FRONT", "正页", "主页"}:
        return "main"
    if raw_side in {"SECONDARY", "BACK", "副页"}:
        return "secondary"
    if raw_side in {"BOTH", "双页", "正副页"}:
        return "both"

    texts = [_compact_ocr_text(token.get("text")) for token in tokens]
    main_score = sum(
        (
            "机动车驾驶证" in text
            and "副页" not in text
        )
        or any(
            marker in text
            for marker in (
                "初次领证日期",
                "准驾车型",
                "有效期限",
                "出生日期",
                "DATEOFBIRTH",
                "DATEOFFIRSTISSUE",
                "VALIDPERIOD",
            )
        )
        for text in texts
    )
    secondary_score = sum(
        any(
            marker in text
            for marker in (
                "驾驶证副页",
                "档案编号",
                "记录",
                "有效起始日期",
            )
        )
        for text in texts
    )
    if main_score and secondary_score:
        return "both"
    if main_score:
        return "main"
    if secondary_score:
        return "secondary"
    return None


def _driver_license_text_scope(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select OCR bands belonging to the detected driving-license page(s).

    A customer may photograph only the main page, only the secondary page, or
    both pages in the same wallet.  Each detected page therefore contributes
    its own semantic band; absence of the other page is not a quality defect.
    """

    side = _driver_license_side(ppocr_result, tokens)
    if side is None:
        return tokens, {
            "side": "unknown",
            "strategy": "all_ocr_tokens",
            "reliable": False,
            "anchors_missing": False,
            "original_token_count": len(tokens),
            "selected_token_count": len(tokens),
        }

    def text(token: dict[str, Any]) -> str:
        return _compact_ocr_text(token.get("text"))

    def main_anchor(token: dict[str, Any]) -> bool:
        value = text(token)
        return (
            ("机动车驾驶证" in value and "副页" not in value)
            or any(
                marker in value
                for marker in (
                    "初次领证日期",
                    "准驾车型",
                    "有效期限",
                    "出生日期",
                    "DATEOFBIRTH",
                    "DATEOFFIRSTISSUE",
                    "VALIDPERIOD",
                )
            )
        )

    def secondary_anchor(token: dict[str, Any]) -> bool:
        value = text(token)
        return any(
            marker in value
            for marker in ("驾驶证副页", "档案编号", "记录", "有效起始日期")
        )

    requested_groups: list[tuple[str, list[dict[str, Any]]]] = []
    if side in {"main", "both"}:
        requested_groups.append(("main", [t for t in tokens if main_anchor(t)]))
    if side in {"secondary", "both"}:
        requested_groups.append(
            ("secondary", [t for t in tokens if secondary_anchor(t)])
        )

    selected_ids: set[int] = set()
    bands: list[dict[str, Any]] = []
    for page, anchors in requested_groups:
        if not anchors:
            continue
        anchor_heights = [
            min(
                float(token["bbox"][2]) - float(token["bbox"][0]),
                float(token["bbox"][3]) - float(token["bbox"][1]),
            )
            for token in anchors
        ]
        margin = 2.0 * float(np.median(anchor_heights)) if anchor_heights else 0.0
        band_top = min(float(token["bbox"][1]) for token in anchors) - margin
        band_bottom = max(float(token["bbox"][3]) for token in anchors) + margin
        band_tokens = [
            token
            for token in tokens
            if band_top
            <= (float(token["bbox"][1]) + float(token["bbox"][3])) / 2.0
            <= band_bottom
        ]
        selected_ids.update(id(token) for token in band_tokens)
        bands.append(
            {
                "page": page,
                "anchor_count": len(anchors),
                "band_top": round(max(0.0, band_top), 2),
                "band_bottom": round(band_bottom, 2),
                "selected_token_count": len(band_tokens),
            }
        )

    selected = [token for token in tokens if id(token) in selected_ids]
    anchors_missing = not bands
    return selected, {
        "side": side,
        "strategy": "semantic_page_bands" if bands else "anchors_missing",
        "reliable": bool(bands),
        "anchors_missing": anchors_missing,
        "bands": bands,
        "original_token_count": len(tokens),
        "selected_token_count": len(selected),
    }


def _driver_license_field_tokens(
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep OCR boxes that can reasonably represent a licence field.

    The semantic page band has already removed most content above/below the
    licence.  Labels, typed values and reliable Chinese values inside that
    band are useful field evidence; isolated low-confidence background OCR is
    deliberately excluded from the per-field hard glare rule.
    """

    field_markers = (
        "证号",
        "姓名",
        "性别",
        "国籍",
        "住址",
        "出生日期",
        "初次领证日期",
        "准驾车型",
        "有效期限",
        "档案编号",
        "记录",
        "有效起始日期",
        "NAME",
        "SEX",
        "NATIONALITY",
        "ADDRESS",
        "DATEOFBIRTH",
        "DATEOFFIRSTISSUE",
        "CLASS",
        "VALIDPERIOD",
    )

    def useful(token: dict[str, Any]) -> bool:
        value = _compact_ocr_text(token.get("text"))
        if not value:
            return False
        if any(marker in value for marker in field_markers):
            return True
        if re.fullmatch(r"\d{10,18}", value):
            return True
        if re.search(r"\d{4}\d{2}\d{2}", value):
            return True
        if re.fullmatch(r"(?:[ABC]\d[A-Z]?|[DEFMNP])", value) or value in {
            "男",
            "女",
            "中国",
            "中国CHN",
        }:
            return True
        score = token.get("score")
        # Names, addresses and issuing authorities cannot be recognized by a
        # fixed regex. Inside a reliable page band, readable Chinese text is
        # still valuable field evidence.
        return (
            isinstance(score, (int, float))
            and float(score) >= 0.80
            and bool(re.search(r"[\u4e00-\u9fff]", value))
        )

    return [token for token in tokens if useful(token)]


_DRIVER_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "certificate_number": ("证号",),
    "name": ("姓名", "NAME"),
    "sex": ("性别", "SEX"),
    "nationality": ("国籍", "NATIONALITY"),
    "address": ("住址", "ADDRESS"),
    "birth_date": ("出生日期", "DATEOFBIRTH"),
    "first_issue_date": ("初次领证日期", "DATEOFFIRSTISSUE"),
    "vehicle_class": ("准驾车型", "CLASS"),
    "valid_period": ("有效期限", "VALIDPERIOD"),
    "archive_number": ("档案编号",),
}

_DRIVER_FIELD_NAMES = {
    "certificate_number": "证号",
    "name": "姓名",
    "sex": "性别",
    "nationality": "国籍",
    "address": "住址",
    "birth_date": "出生日期",
    "first_issue_date": "初次领证日期",
    "vehicle_class": "准驾车型",
    "valid_period": "有效期限",
    "archive_number": "档案编号",
}

# OCR text is used only to locate these key fields. Their recognized values and
# confidence scores never participate in the glare decision.
_DRIVER_GLARE_ACTIONABLE_FIELDS = {
    "certificate_number",
    "address",
    "valid_period",
}

_DRIVER_FIELD_GLARE_THRESHOLD = 0.08
_DRIVER_FIELD_CONTIGUOUS_GLARE_THRESHOLD = 0.03
_DRIVER_FIELD_TEXTURE_LOSS_THRESHOLD = 0.18
_DRIVER_FIELD_CONTIGUOUS_TEXTURE_LOSS_THRESHOLD = 0.08
_DRIVER_FIELD_TEXTURE_LOSS_MIN_SPECULAR_EVIDENCE = 0.001
_DRIVER_FIELD_TEXTURE_LOSS_MIN_LINKED_SPECULAR_EVIDENCE = 0.0008
_DRIVER_FIELD_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA = 15.0
_DRIVER_FIELD_HORIZONTAL_PADDING_RATIO = 0.20
_DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS = 1.50
_DRIVER_COMPLETE_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS = 0.50
_DRIVER_FIELD_VERTICAL_PADDING_RATIO = 0.10
_DRIVER_SPECULAR_MIN_BRIGHTNESS_DELTA = 15.0
_DRIVER_SPECULAR_MIN_COMPONENT_FILL_RATIO = 0.35
_DRIVER_READABLE_VALUE_CONFIDENCE = 0.98
_DRIVER_MIN_STANDALONE_TEXT_HEIGHT_RATIO = 0.010
_DRIVER_ELONGATED_GLARE_MIN_COMPONENT_COUNT = 4
_DRIVER_ELONGATED_GLARE_MIN_TOTAL_AREA_RATIO = 0.001
_DRIVER_ELONGATED_GLARE_MIN_ASPECT_RATIO = 5.0
_DRIVER_ELONGATED_GLARE_MIN_LENGTH_RATIO = 0.025
_DRIVER_ELONGATED_GLARE_MAX_THICKNESS_RATIO = 0.025
_DRIVER_ELONGATED_GLARE_MIN_TEXT_OVERLAP = 0.05
_DRIVER_FALLBACK_GLARE_MIN_RATIO = 0.015
_DRIVER_FALLBACK_GLARE_MAX_RATIO = 0.023
_DRIVER_FALLBACK_MULTI_STREAK_COUNT = 2
_DRIVER_FALLBACK_MULTI_STREAK_RATIO = 0.002
_DRIVER_FALLBACK_FIELD_CANDIDATE_COUNT = 2
_DRIVER_NEAR_SPECULAR_GLARE_THRESHOLD = 0.03
_DRIVER_NEAR_SPECULAR_CONTIGUOUS_GLARE_THRESHOLD = 0.015
_DRIVER_LARGE_GLARE_MIN_RATIO = 0.015
_DRIVER_LARGE_GLARE_MAX_RATIO = 0.05
_DRIVER_LARGE_GLARE_MIN_STRICT_SPECULAR_RATIO = 0.0015
_DRIVER_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO = 0.00045
_DRIVER_LARGE_GLARE_MIN_GLOBAL_COMPONENT_RATIO = 0.01
_DRIVER_LARGE_GLARE_MIN_TEXT_OVERLAP = 0.03
_DRIVER_LARGE_GLARE_MIN_TEXT_COUNT = 2
_DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP = 0.08
_DRIVER_LARGE_GLARE_MIN_AFFECTED_TEXT_COUNT = 4
_DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_RATIO = 0.004
_DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO = 0.0015
_DRIVER_STRICT_STREAK_MIN_GLOBAL_RATIO = 0.008
_DRIVER_STRICT_STREAK_MAX_LOW_GLOBAL_RATIO = 0.015
_DRIVER_STRICT_STREAK_MAX_MULTI_GLOBAL_RATIO = 0.03
_DRIVER_STRICT_STREAK_MIN_RATIO = 0.005
_DRIVER_STRICT_STREAK_MIN_COMPONENT_RATIO = 0.0015
_DRIVER_STRICT_STREAK_MIN_TEXT_COUNT = 2
_DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_COUNT = 2
_DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_AREA_RATIO = 0.002
_DRIVER_STRICT_STREAK_MIN_MULTI_TEXT_COUNT = 5
_DRIVER_STRICT_STREAK_MIN_MULTI_COUNT = 3
_DRIVER_STRICT_STREAK_MIN_MULTI_AREA_RATIO = 0.005
_DRIVER_STRONG_STRICT_GLARE_MIN_RATIO = 0.015
_DRIVER_STRONG_STRICT_GLARE_MIN_COMPONENT_RATIO = 0.003
_DRIVER_STRONG_STRICT_GLARE_MIN_TEXT_COUNT = 5
_DRIVER_STRONG_STRICT_GLARE_MIN_TEXT_OVERLAP = 0.20
_DRIVER_DEGRADED_PAGE_GLARE_MIN_STRICT_RATIO = 0.004
_DRIVER_DEGRADED_PAGE_GLARE_MIN_COMPONENT_RATIO = 0.001
_DRIVER_DEGRADED_SECONDARY_MIN_TEXT_COUNT = 5
_DRIVER_DEGRADED_MAIN_MAX_LABEL_BACKED_FIELDS = 2
_DRIVER_DEGRADED_MAIN_MAX_SPECIFIC_FIELDS = 1


def _driver_field_label(token: dict[str, Any]) -> tuple[str, str] | None:
    value = _compact_ocr_text(token.get("text"))
    # Longer English aliases must win over their substrings (for example,
    # NATIONALITY contains NATIONAL but not the NAME field semantically).
    candidates = sorted(
        (
            (field, alias)
            for field, aliases in _DRIVER_FIELD_ALIASES.items()
            for alias in aliases
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for field, alias in candidates:
        if alias in value:
            return field, alias
    return None


def _driver_field_candidate(field: str, value: str) -> bool:
    compact = _compact_ocr_text(value)
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    if field == "certificate_number":
        return bool(re.fullmatch(r"\d{14,17}[0-9X]", compact))
    if field == "name":
        return 2 <= chinese_count <= 6 and not any(
            marker in compact
            for marker in ("姓名", "记录", "证号", "档案", "公安", "警察")
        )
    if field == "sex":
        return compact in {"男", "女"}
    if field == "nationality":
        return compact in {"中", "中国", "CHN", "中CHN", "中国CHN"}
    if field == "address":
        return chinese_count >= 3 and not any(
            marker in compact
            for marker in (
                "中华人民共和国",
                "机动车驾驶证",
                "驾驶证副页",
                "公安局",
                "警察支队",
                "出生日期",
                "初次领证",
                "有效期限",
                "有效起始",
                "年月日",
                "记录",
            )
        )
    if field in {"birth_date", "first_issue_date"}:
        raw = str(value).strip()
        return bool(
            re.fullmatch(r"\d{8}", compact)
            or re.fullmatch(r"\d{4}\D+\d{1,2}\D+\d{1,2}", raw)
        )
    if field == "vehicle_class":
        return bool(re.fullmatch(r"(?:[ABC]\d[A-Z]?|[DEFMNP])", compact))
    if field == "valid_period":
        # Glare commonly erases the second date or a few digits. Keep a
        # date-like partial value for region localisation; correctness of the
        # recognized date does not participate in the glare decision.
        return compact == "长期" or len(re.findall(r"\d", str(value))) >= 6
    if field == "archive_number":
        return bool(re.fullmatch(r"\d{8,16}", compact))
    return False


def _driver_tokens_on_same_line(
    label: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, float]:
    """Return line compatibility and a rotation-independent proximity score."""

    left, top, right, bottom = (float(value) for value in label["bbox"])
    c_left, c_top, c_right, c_bottom = (
        float(value) for value in candidate["bbox"]
    )
    width = right - left
    height = bottom - top
    if width >= height:
        perpendicular_gap = max(0.0, max(top, c_top) - min(bottom, c_bottom))
        short_side = max(1.0, min(height, c_bottom - c_top))
        axial_gap = max(0.0, max(left, c_left) - min(right, c_right))
    else:
        perpendicular_gap = max(0.0, max(left, c_left) - min(right, c_right))
        short_side = max(1.0, min(width, c_right - c_left))
        axial_gap = max(0.0, max(top, c_top) - min(bottom, c_bottom))
    # Crossing into an adjacent printed row is much riskier than a moderate
    # gap along the current row. Weight that perpendicular drift so a nearby
    # issuing-authority stamp cannot beat the actual address value.
    proximity = 3.0 * perpendicular_gap + axial_gap
    return perpendicular_gap <= 1.5 * short_side, proximity


def _driver_license_field_assignments(
    tokens: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Associate labels and values using text rules plus line proximity."""

    assignments = {
        field: {"labels": [], "values": []}
        for field in _DRIVER_FIELD_ALIASES
    }
    labelled_ids: set[int] = set()

    def append_value(
        assignment: dict[str, Any],
        token: dict[str, Any],
        field_value: str,
        **metadata: Any,
    ) -> None:
        source_id = id(token)
        if any(existing.get("_source_id") == source_id for existing in assignment["values"]):
            return
        assignment["values"].append(
            {
                **token,
                "field_value": field_value,
                "_source_id": source_id,
                **metadata,
            }
        )

    for token in tokens:
        matched = _driver_field_label(token)
        if matched is None:
            continue
        field, alias = matched
        assignments[field]["labels"].append(token)
        labelled_ids.add(id(token))
        compact = _compact_ocr_text(token.get("text"))
        remainder = compact.split(alias, 1)[1]
        if remainder and _driver_field_candidate(field, remainder):
            append_value(assignments[field], token, remainder, inline=True)

    unlabelled = [token for token in tokens if id(token) not in labelled_ids]
    for field, assignment in assignments.items():
        for label in assignment["labels"]:
            nearby: list[tuple[float, dict[str, Any]]] = []
            for token in unlabelled:
                if not _driver_field_candidate(field, str(token.get("text") or "")):
                    continue
                compatible, distance = _driver_tokens_on_same_line(label, token)
                if compatible:
                    nearby.append((distance, token))
            if nearby:
                token = min(nearby, key=lambda item: item[0])[1]
                append_value(
                    assignment,
                    token,
                    _compact_ocr_text(token.get("text")),
                )

    # Numbers and dates are sufficiently typed to remain useful even when
    # glare erases their printed label or PP splits it into a different box.
    for token in unlabelled:
        compact = _compact_ocr_text(token.get("text"))
        inferred_field: str | None = None
        if re.fullmatch(r"\d{10,17}[0-9X]", compact):
            inferred_field = "certificate_number" if len(compact) >= 15 else "archive_number"
        elif re.fullmatch(r"(?:[ABC]\d[A-Z]?|[DEFMNP])", compact):
            inferred_field = "vehicle_class"
        if inferred_field is not None:
            assignment = assignments[inferred_field]
            append_value(assignment, token, compact, inferred=True)

    # The same identity number can also be printed in a small white strip
    # beneath the portrait. Once a value has been associated with a printed
    # ``证号`` label, discard unlabeled duplicates so that the white strip is
    # not mistaken for glare over the actual certificate-number field.
    certificate_values = assignments["certificate_number"]["values"]
    if any(not token.get("inferred") for token in certificate_values):
        assignments["certificate_number"]["values"] = [
            token for token in certificate_values if not token.get("inferred")
        ]

    # Severe glare can erase the printed address label while PP-OCR still
    # recovers a partial value. In that case keep the longest address-like
    # token inside the already scoped main-page OCR band. Issuing-authority
    # and title fragments are excluded by _driver_field_candidate().
    address_assignment = assignments["address"]
    if not address_assignment["values"]:
        address_candidates = [
            token
            for token in unlabelled
            if _driver_field_candidate("address", str(token.get("text") or ""))
        ]
        if address_candidates:
            token = max(
                address_candidates,
                key=lambda item: len(_compact_ocr_text(item.get("text"))),
            )
            append_value(
                address_assignment,
                token,
                _compact_ocr_text(token.get("text")),
                inferred=True,
            )
    return assignments


def _driver_partial_address_projection(
    token: dict[str, Any],
) -> dict[str, Any] | None:
    """Project a separate ROI beyond a short, glare-truncated address box."""

    value = _compact_ocr_text(
        token.get("field_value") or token.get("text")
    )
    if len(value) > 4:
        return None
    left, top, right, bottom = (
        float(value) for value in token["bbox"]
    )
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    projected = {**token, "field_value": str(token.get("field_value") or "")}
    if width >= height:
        projected["bbox"] = [
            right,
            top,
            right + 8.0 * height,
            bottom,
        ]
    else:
        projected["bbox"] = [
            left,
            bottom,
            right,
            bottom + 8.0 * width,
        ]
    projected["projected_from_partial_address"] = True
    return projected


def _driver_partial_valid_period_projection(
    token: dict[str, Any],
) -> dict[str, Any] | None:
    """Project toward the incomplete side of a truncated validity value."""

    raw_value = str(token.get("field_value") or token.get("text") or "")
    compact = _compact_ocr_text(raw_value)
    complete_dates = re.findall(r"\d{4}\D?\d{1,2}\D?\d{1,2}", raw_value)
    if compact == "长期" or len(complete_dates) >= 2:
        return None
    left, top, right, bottom = (
        float(value) for value in token["bbox"]
    )
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    projected = {**token, "field_value": raw_value}
    projection_direction = "after"
    period_parts = re.split(r"[至到]", compact, maxsplit=1)
    if len(period_parts) == 2:
        start_digits = len(re.findall(r"\d", period_parts[0]))
        end_digits = len(re.findall(r"\d", period_parts[1]))
        if start_digits < 8 <= end_digits:
            projection_direction = "before"
    if width >= height:
        if projection_direction == "before":
            projected["bbox"] = [
                left - 8.0 * height,
                top,
                left,
                bottom,
            ]
        else:
            projected["bbox"] = [
                right,
                top,
                right + 8.0 * height,
                bottom,
            ]
    else:
        if projection_direction == "before":
            projected["bbox"] = [
                left,
                top - 8.0 * width,
                right,
                top,
            ]
        else:
            projected["bbox"] = [
                left,
                bottom,
                right,
                bottom + 8.0 * width,
            ]
    projected["projected_from_partial_valid_period"] = True
    projected["projection_direction"] = projection_direction
    return projected


def _driver_minimum_horizontal_padding_heights(
    field: str,
    token: dict[str, Any],
) -> float:
    """Use wide padding only for missing or visibly truncated field values."""

    if token.get("projected_from_partial_address") or token.get(
        "projected_from_partial_valid_period"
    ):
        return _DRIVER_COMPLETE_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
    raw_value = str(token.get("field_value") or "")
    if not raw_value:
        return _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
    if field == "address" and len(_compact_ocr_text(raw_value)) <= 4:
        return _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
    if (
        field == "valid_period"
        and _driver_partial_valid_period_projection(token) is not None
    ):
        return _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
    return _DRIVER_COMPLETE_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS


def _driver_has_reliably_readable_value(
    field: str,
    value_tokens: list[dict[str, Any]],
) -> bool:
    """Suppress only the washout channel when OCR recovered a complete value."""

    for token in value_tokens:
        score = token.get("score")
        if not isinstance(score, (int, float)) or float(score) < (
            _DRIVER_READABLE_VALUE_CONFIDENCE
        ):
            continue
        value = _compact_ocr_text(
            token.get("field_value") or token.get("text")
        )
        if field == "address" and len(value) >= 8:
            return True
        if field == "valid_period" and len(re.findall(r"\d", value)) >= 16:
            return True
        if field == "certificate_number" and re.fullmatch(
            r"\d{14,17}[0-9X]",
            value,
        ):
            return True
    return False


def _driver_weak_physical_evidence(
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recognize a badly degraded physical licence from independent fields.

    A distant or strongly reflective licence can lose every printed label while
    PP-OCR still recovers the characteristic vehicle class, nationality marker,
    and several date-shaped values. Requiring all three signals avoids treating
    an arbitrary unrecognized document as a driving licence.
    """

    raw_values = [str(token.get("text") or "").strip() for token in tokens]
    values = [_compact_ocr_text(value) for value in raw_values]
    vehicle_classes = sorted(
        {
            value.upper()
            for value in values
            if re.fullmatch(r"(?:[ABC]\d[A-Z]?|[DEFMNP])", value.upper())
        }
    )
    nationality_markers = sorted(
        {
            value
            for value in values
            if value in {"中国", "中国CH", "中国CHN", "CHN"}
        }
    )
    date_like_values = [
        raw_value
        for raw_value in raw_values
        if len(re.findall(r"\d", raw_value)) >= 6
        and bool(re.search(r"[-./年月日]", raw_value))
    ]
    detected = bool(
        len(vehicle_classes) == 1
        and (
            (nationality_markers and len(date_like_values) >= 2)
            or len(date_like_values) >= 3
        )
    )
    return {
        "detected": detected,
        "vehicle_classes": vehicle_classes,
        "nationality_markers": nationality_markers,
        "date_like_values": date_like_values,
        "minimum_date_like_count": 2,
        "strategy": "vehicle_class_with_nationality_or_three_dates",
    }


def _driver_physical_glare_fallback_metrics(
    *,
    glare_ratio: float,
    elongated_glare_count: int,
    elongated_glare_ratio: float,
    ocr_metrics: dict[str, Any],
    field_glare: dict[str, Any] | None,
) -> dict[str, Any]:
    """Detect strong physical-card glare that confident OCR can conceal."""

    max_text_overlap = float(ocr_metrics.get("max_glare_overlap") or 0.0)
    affected_token_count = int(
        ocr_metrics.get("glare_affected_token_count") or 0
    )
    field_candidate_count = int(
        (field_glare or {}).get("candidate_count") or 0
    )
    field_max_overlap = float((field_glare or {}).get("max_overlap") or 0.0)
    multiple_streaks = (
        glare_ratio >= _DRIVER_FALLBACK_GLARE_MIN_RATIO
        and glare_ratio <= _DRIVER_FALLBACK_GLARE_MAX_RATIO
        and elongated_glare_count >= _DRIVER_FALLBACK_MULTI_STREAK_COUNT
        and elongated_glare_ratio >= _DRIVER_FALLBACK_MULTI_STREAK_RATIO
        and field_candidate_count >= _DRIVER_FALLBACK_FIELD_CANDIDATE_COUNT
        and field_max_overlap >= _DRIVER_FIELD_GLARE_THRESHOLD
    )
    return {
        "detected": multiple_streaks,
        "decision_basis": (
            "multiple_streaks_crossing_text"
            if multiple_streaks
            else None
        ),
        "strong_text_glare": False,
        "multiple_streaks": multiple_streaks,
        "glare_ratio": round(glare_ratio, 4),
        "minimum_glare_ratio": _DRIVER_FALLBACK_GLARE_MIN_RATIO,
        "maximum_glare_ratio": _DRIVER_FALLBACK_GLARE_MAX_RATIO,
        "max_text_overlap": round(max_text_overlap, 4),
        "affected_token_count": affected_token_count,
        "elongated_glare_count": elongated_glare_count,
        "minimum_elongated_glare_count": _DRIVER_FALLBACK_MULTI_STREAK_COUNT,
        "elongated_glare_ratio": round(elongated_glare_ratio, 4),
        "minimum_elongated_glare_ratio": _DRIVER_FALLBACK_MULTI_STREAK_RATIO,
        "field_candidate_count": field_candidate_count,
        "minimum_field_candidate_count": (
            _DRIVER_FALLBACK_FIELD_CANDIDATE_COUNT
        ),
        "field_max_overlap": round(field_max_overlap, 4),
        "minimum_field_max_overlap": _DRIVER_FIELD_GLARE_THRESHOLD,
    }


def _driver_large_area_glare_fallback_metrics(
    strict_specular_mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
    *,
    glare_ratio: float,
    largest_glare_ratio: float,
    ocr_metrics: dict[str, Any],
    driver_scope: dict[str, Any] | None,
    weak_physical_evidence: dict[str, Any] | None,
    elongated_glare_count: int = 0,
    elongated_glare_ratio: float = 0.0,
    semantic_glare_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect one obvious broad reflection after field rules miss it."""

    image_pixels = max(1, int(strict_specular_mask.size))
    strict_specular_ratio = float(
        np.count_nonzero(strict_specular_mask)
    ) / float(image_pixels)
    strict_largest_ratio, _elongated_ratio, _elongated_count = (
        _glare_component_metrics(strict_specular_mask, image_pixels)
    )
    text_overlaps = [
        overlap
        for token in tokens
        if (
            overlap := _token_mask_overlap(
                strict_specular_mask,
                token,
                coordinate_scale,
                horizontal_padding_ratio=0.20,
                minimum_horizontal_padding_heights=0.5,
                vertical_padding_ratio=0.10,
                padding_follows_text_axis=True,
            )
        )
        is not None
    ]
    max_text_overlap = max(text_overlaps, default=0.0)
    affected_text_count = sum(
        overlap >= _DRIVER_LARGE_GLARE_MIN_TEXT_OVERLAP
        for overlap in text_overlaps
    )
    scope_side = str((driver_scope or {}).get("side") or "unknown")
    global_shape_supported = bool(
        _DRIVER_LARGE_GLARE_MIN_RATIO
        <= glare_ratio
        <= _DRIVER_LARGE_GLARE_MAX_RATIO
        and largest_glare_ratio
        >= _DRIVER_LARGE_GLARE_MIN_GLOBAL_COMPONENT_RATIO
    )
    recognized_document_glare = bool(
        scope_side != "unknown"
        and global_shape_supported
        and strict_specular_ratio
        >= _DRIVER_LARGE_GLARE_MIN_STRICT_SPECULAR_RATIO
        and strict_largest_ratio
        >= _DRIVER_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO
        and affected_text_count >= _DRIVER_LARGE_GLARE_MIN_TEXT_COUNT
        and max_text_overlap
        >= _DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP
        and int(ocr_metrics.get("glare_affected_token_count") or 0)
        >= _DRIVER_LARGE_GLARE_MIN_AFFECTED_TEXT_COUNT
    )
    strict_text_supported = bool(
        strict_specular_ratio >= _DRIVER_STRICT_STREAK_MIN_RATIO
        and strict_largest_ratio
        >= _DRIVER_STRICT_STREAK_MIN_COMPONENT_RATIO
        and affected_text_count >= _DRIVER_STRICT_STREAK_MIN_TEXT_COUNT
        and max_text_overlap
        >= _DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP
    )
    low_global_strict_streak_glare = bool(
        scope_side != "unknown"
        and _DRIVER_STRICT_STREAK_MIN_GLOBAL_RATIO
        <= glare_ratio
        < _DRIVER_STRICT_STREAK_MAX_LOW_GLOBAL_RATIO
        and strict_text_supported
        and elongated_glare_count
        >= _DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_COUNT
        and elongated_glare_ratio
        >= _DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_AREA_RATIO
    )
    multiple_strict_streak_glare = bool(
        scope_side != "unknown"
        and _DRIVER_STRICT_STREAK_MAX_LOW_GLOBAL_RATIO
        <= glare_ratio
        <= _DRIVER_STRICT_STREAK_MAX_MULTI_GLOBAL_RATIO
        and elongated_glare_count
        >= _DRIVER_STRICT_STREAK_MIN_MULTI_COUNT
        and elongated_glare_ratio
        >= _DRIVER_STRICT_STREAK_MIN_MULTI_AREA_RATIO
        and strict_specular_ratio >= _DRIVER_STRICT_STREAK_MIN_RATIO
        and strict_largest_ratio
        >= _DRIVER_STRICT_STREAK_MIN_COMPONENT_RATIO
        and affected_text_count
        >= _DRIVER_STRICT_STREAK_MIN_MULTI_TEXT_COUNT
        and max_text_overlap
        >= _DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP
    )
    overwhelming_strict_text_glare = bool(
        scope_side != "unknown"
        and glare_ratio <= _DRIVER_LARGE_GLARE_MAX_RATIO
        and strict_specular_ratio
        >= _DRIVER_STRONG_STRICT_GLARE_MIN_RATIO
        and strict_largest_ratio
        >= _DRIVER_STRONG_STRICT_GLARE_MIN_COMPONENT_RATIO
        and affected_text_count
        >= _DRIVER_STRONG_STRICT_GLARE_MIN_TEXT_COUNT
        and max_text_overlap
        >= _DRIVER_STRONG_STRICT_GLARE_MIN_TEXT_OVERLAP
        and int(ocr_metrics.get("glare_affected_token_count") or 0) >= 1
    )
    physical_page_region_supported = bool(
        global_shape_supported
        and strict_specular_ratio
        >= _DRIVER_DEGRADED_PAGE_GLARE_MIN_STRICT_RATIO
        and strict_largest_ratio
        >= _DRIVER_DEGRADED_PAGE_GLARE_MIN_COMPONENT_RATIO
    )
    field_rows = list((semantic_glare_metrics or {}).get("fields") or [])
    label_backed_value_fields = {
        str(row.get("field") or "")
        for row in field_rows
        if int(row.get("label_count") or 0) > 0
        and bool(row.get("texts"))
    }
    driver_specific_label_fields = label_backed_value_fields & {
        "archive_number",
        "first_issue_date",
        "valid_period",
        "vehicle_class",
    }
    secondary_page_title_detected = any(
        "驾驶证副页" in _compact_ocr_text(token.get("text"))
        for token in tokens
    )
    degraded_secondary_page_glare = bool(
        scope_side == "secondary"
        and physical_page_region_supported
        and affected_text_count
        >= _DRIVER_DEGRADED_SECONDARY_MIN_TEXT_COUNT
        and max_text_overlap
        >= _DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP
    )
    degraded_main_page_glare = bool(
        scope_side == "both"
        and secondary_page_title_detected
        and physical_page_region_supported
        and len(label_backed_value_fields)
        <= _DRIVER_DEGRADED_MAIN_MAX_LABEL_BACKED_FIELDS
        and len(driver_specific_label_fields)
        <= _DRIVER_DEGRADED_MAIN_MAX_SPECIFIC_FIELDS
    )
    vehicle_classes = list(
        (weak_physical_evidence or {}).get("vehicle_classes") or []
    )
    degraded_document_glare = bool(
        scope_side == "unknown"
        and global_shape_supported
        and int(ocr_metrics.get("token_count") or 0) >= 6
        and len(vehicle_classes) == 1
        and strict_specular_ratio
        >= _DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_RATIO
        and strict_largest_ratio
        >= _DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO
    )
    detected = bool(
        recognized_document_glare
        or degraded_document_glare
        or low_global_strict_streak_glare
        or multiple_strict_streak_glare
        or overwhelming_strict_text_glare
        or degraded_secondary_page_glare
        or degraded_main_page_glare
    )
    decision_basis = (
        "large_specular_region_crossing_text"
        if recognized_document_glare
        else (
            "large_specular_region_with_degraded_driver_evidence"
            if degraded_document_glare
            else (
                "low_global_strict_streaks_crossing_text"
                if low_global_strict_streak_glare
                else (
                    "multiple_strict_streaks_crossing_text"
                    if multiple_strict_streak_glare
                    else (
                        "overwhelming_strict_glare_crossing_text"
                        if overwhelming_strict_text_glare
                        else (
                            "degraded_secondary_page_under_broad_glare"
                            if degraded_secondary_page_glare
                            else (
                                "degraded_main_page_beside_secondary_page"
                                if degraded_main_page_glare
                                else None
                            )
                        )
                    )
                )
            )
        )
    )
    return {
        "detected": detected,
        "decision_basis": decision_basis,
        "recognized_document_glare": recognized_document_glare,
        "degraded_document_glare": degraded_document_glare,
        "low_global_strict_streak_glare": low_global_strict_streak_glare,
        "multiple_strict_streak_glare": multiple_strict_streak_glare,
        "overwhelming_strict_text_glare": overwhelming_strict_text_glare,
        "degraded_secondary_page_glare": degraded_secondary_page_glare,
        "degraded_main_page_glare": degraded_main_page_glare,
        "scope_side": scope_side,
        "glare_ratio": round(glare_ratio, 4),
        "minimum_glare_ratio": _DRIVER_LARGE_GLARE_MIN_RATIO,
        "maximum_glare_ratio": _DRIVER_LARGE_GLARE_MAX_RATIO,
        "largest_glare_ratio": round(largest_glare_ratio, 4),
        "minimum_largest_glare_ratio": (
            _DRIVER_LARGE_GLARE_MIN_GLOBAL_COMPONENT_RATIO
        ),
        "strict_specular_ratio": round(strict_specular_ratio, 4),
        "minimum_strict_specular_ratio": (
            _DRIVER_LARGE_GLARE_MIN_STRICT_SPECULAR_RATIO
        ),
        "strict_largest_component_ratio": round(strict_largest_ratio, 4),
        "minimum_strict_component_ratio": (
            _DRIVER_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO
        ),
        "strict_text_overlap_count": affected_text_count,
        "minimum_strict_text_overlap_count": (
            _DRIVER_LARGE_GLARE_MIN_TEXT_COUNT
        ),
        "max_strict_text_overlap": round(max_text_overlap, 4),
        "minimum_max_strict_text_overlap": (
            _DRIVER_LARGE_GLARE_MIN_MAX_TEXT_OVERLAP
        ),
        "broad_glare_affected_token_count": int(
            ocr_metrics.get("glare_affected_token_count") or 0
        ),
        "minimum_broad_glare_affected_token_count": (
            _DRIVER_LARGE_GLARE_MIN_AFFECTED_TEXT_COUNT
        ),
        "elongated_glare_count": elongated_glare_count,
        "elongated_glare_ratio": round(elongated_glare_ratio, 4),
        "minimum_low_global_elongated_glare_count": (
            _DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_COUNT
        ),
        "minimum_low_global_elongated_glare_ratio": (
            _DRIVER_STRICT_STREAK_MIN_LOW_GLOBAL_AREA_RATIO
        ),
        "physical_page_region_supported": physical_page_region_supported,
        "secondary_page_title_detected": secondary_page_title_detected,
        "label_backed_value_field_count": len(label_backed_value_fields),
        "driver_specific_label_field_count": len(
            driver_specific_label_fields
        ),
        "maximum_degraded_main_label_backed_fields": (
            _DRIVER_DEGRADED_MAIN_MAX_LABEL_BACKED_FIELDS
        ),
        "maximum_degraded_main_specific_fields": (
            _DRIVER_DEGRADED_MAIN_MAX_SPECIFIC_FIELDS
        ),
        "vehicle_classes": vehicle_classes,
        "unknown_minimum_strict_specular_ratio": (
            _DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_RATIO
        ),
        "unknown_minimum_strict_component_ratio": (
            _DRIVER_UNKNOWN_LARGE_GLARE_MIN_STRICT_COMPONENT_RATIO
        ),
    }


def _electronic_driver_license_keywords(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify electronic-licence evidence without trusting background text.

    A physical licence reverse page legitimately contains words such as
    ``代号`` and ``规定``. Those words may also be recognized from a monitor
    behind the licence. Once physical-page anchors are present, a single
    generic keyword is therefore insufficient to bypass all quality checks.
    """

    keywords = ("电子", "扫码", "生成", "状态", "代号", "规定", "主页", "刷新")
    evidence = _compact_ocr_text(ppocr_result.get("text"))
    if tokens:
        evidence += "".join(_compact_ocr_text(token.get("text")) for token in tokens)
    candidates = [keyword for keyword in keywords if keyword in evidence]
    physical_side = _driver_license_side(ppocr_result, tokens)

    if physical_side is None:
        matched = candidates
        strategy = "keyword_without_physical_page_anchor"
    else:
        ui_keywords = {"扫码", "生成", "状态", "主页", "刷新"}
        ui_matches = [keyword for keyword in candidates if keyword in ui_keywords]
        # ``电子`` is itself explicit document-mode evidence. Unlike generic
        # reverse-page words such as ``代号`` or ``规定``, it does not appear on
        # a normal physical driving licence and should win even when OCR also
        # sees physical-page field anchors rendered on the screen.
        explicit_electronic_marker = "电子" in evidence
        detected = explicit_electronic_marker or len(ui_matches) >= 2
        matched = candidates if detected else []
        strategy = "physical_page_accepts_electronic_marker_or_two_ui_keywords"

    return {
        "detected": bool(matched),
        "matched_keywords": matched,
        "candidate_keywords": candidates,
        "physical_page_side": physical_side,
        "strategy": strategy,
    }


def _vehicle_license_side(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> str | None:
    """Distinguish a vehicle-licence main page, secondary page, or both."""

    raw_side = _compact_ocr_text(
        ppocr_result.get("quality_side") or ppocr_result.get("document_side")
    )
    if raw_side in {"MAIN", "PRIMARY", "FRONT", "正本", "主页"}:
        return "main"
    if raw_side in {"SECONDARY", "BACK", "副页"}:
        return "secondary"
    if raw_side in {"BOTH", "双页", "正副页", "正本与副页"}:
        return "both"

    texts = [_compact_ocr_text(token.get("text")) for token in tokens]
    main_score = sum(
        ("机动车行驶证" in text and "副页" not in text)
        or any(
            marker in text
            for marker in (
                "车辆类型",
                "所有人",
                "使用性质",
                "品牌型号",
                "车辆识别代号",
                "发动机号码",
                "注册日期",
                "发证日期",
                "PLATENO",
                "VEHICLETYPE",
                "OWNER",
                "USECHARACTER",
                "VEHICLEIDENTIFICATIONNUMBER",
                "ENGINENO",
                "REGISTERDATE",
                "ISSUEDATE",
            )
        )
        for text in texts
    )
    secondary_score = sum(
        any(
            marker in text
            for marker in (
                "行驶证副页",
                "档案编号",
                "档案号码",
                "核定载人数",
                "总质量",
                "核定载质量",
                "整备质量",
                "准牵引总质量",
                "外廓尺寸",
                "检验记录",
                "强制报废期止",
                "检验有效期",
            )
        )
        for text in texts
    )
    if main_score and secondary_score:
        return "both"
    if main_score:
        return "main"
    if secondary_score:
        return "secondary"
    return None


def _vehicle_license_text_scope(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select independent semantic bands for vehicle-licence pages."""

    side = _vehicle_license_side(ppocr_result, tokens)
    if side is None:
        return tokens, {
            "side": "unknown",
            "strategy": "all_ocr_tokens",
            "reliable": False,
            "anchors_missing": False,
            "original_token_count": len(tokens),
            "selected_token_count": len(tokens),
        }

    def text(token: dict[str, Any]) -> str:
        return _compact_ocr_text(token.get("text"))

    def main_anchor(token: dict[str, Any]) -> bool:
        value = text(token)
        return (
            ("机动车行驶证" in value and "副页" not in value)
            or any(
                marker in value
                for marker in (
                    "车辆类型",
                    "所有人",
                    "使用性质",
                    "品牌型号",
                    "车辆识别代号",
                    "发动机号码",
                    "注册日期",
                    "发证日期",
                    "PLATENO",
                    "VEHICLETYPE",
                    "OWNER",
                    "USECHARACTER",
                    "VEHICLEIDENTIFICATIONNUMBER",
                    "ENGINENO",
                    "REGISTERDATE",
                    "ISSUEDATE",
                )
            )
        )

    def secondary_anchor(token: dict[str, Any]) -> bool:
        value = text(token)
        return any(
            marker in value
            for marker in (
                "行驶证副页",
                "档案编号",
                "档案号码",
                "核定载人数",
                "总质量",
                "核定载质量",
                "整备质量",
                "准牵引总质量",
                "外廓尺寸",
                "检验记录",
                "强制报废期止",
                "检验有效期",
            )
        )

    requested_groups: list[tuple[str, list[dict[str, Any]]]] = []
    if side in {"main", "both"}:
        requested_groups.append(("main", [t for t in tokens if main_anchor(t)]))
    if side in {"secondary", "both"}:
        requested_groups.append(
            ("secondary", [t for t in tokens if secondary_anchor(t)])
        )

    selected_ids: set[int] = set()
    bands: list[dict[str, Any]] = []
    for page, anchors in requested_groups:
        if not anchors:
            continue
        anchor_heights = [
            min(
                float(token["bbox"][2]) - float(token["bbox"][0]),
                float(token["bbox"][3]) - float(token["bbox"][1]),
            )
            for token in anchors
        ]
        margin = 2.0 * float(np.median(anchor_heights)) if anchor_heights else 0.0
        band_top = min(float(token["bbox"][1]) for token in anchors) - margin
        band_bottom = max(float(token["bbox"][3]) for token in anchors) + margin
        band_tokens = [
            token
            for token in tokens
            if band_top
            <= (float(token["bbox"][1]) + float(token["bbox"][3])) / 2.0
            <= band_bottom
        ]
        selected_ids.update(id(token) for token in band_tokens)
        bands.append(
            {
                "page": page,
                "anchor_count": len(anchors),
                "band_top": round(max(0.0, band_top), 2),
                "band_bottom": round(band_bottom, 2),
                "selected_token_count": len(band_tokens),
            }
        )

    selected = [token for token in tokens if id(token) in selected_ids]
    return selected, {
        "side": side,
        "strategy": "semantic_page_bands" if bands else "anchors_missing",
        "reliable": bool(bands),
        "anchors_missing": not bands,
        "bands": bands,
        "original_token_count": len(tokens),
        "selected_token_count": len(selected),
    }


def _vehicle_license_field_tokens(
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep useful vehicle fields and typed values inside semantic bands."""

    field_markers = (
        "号牌号码",
        "车辆类型",
        "所有人",
        "住址",
        "使用性质",
        "品牌型号",
        "车辆识别代号",
        "发动机号码",
        "注册日期",
        "发证日期",
        "档案编号",
        "档案号码",
        "总质量",
        "核定载人数",
        "核定载质量",
        "强制报废期止",
        "检验有效期",
        "准牵引总质量",
        "整备质量",
        "外廓尺寸",
        "检验记录",
        "PLATENO",
        "VEHICLETYPE",
        "OWNER",
        "ADDRESS",
        "USECHARACTER",
        "MODEL",
        "VEHICLEIDENTIFICATIONNUMBER",
        "VIN",
        "ENGINENO",
        "REGISTERDATE",
        "ISSUEDATE",
    )

    def useful(token: dict[str, Any]) -> bool:
        value = _compact_ocr_text(token.get("text"))
        if not value:
            return False
        if any(marker in value for marker in field_markers):
            return True
        if re.fullmatch(r"[A-HJ-NPR-Z0-9*]{12,18}", value):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{4,7}", value):
            return True
        if re.search(r"\d{4}\d{2}\d{2}", value):
            return True
        if re.search(r"\d+(?:KG|人)$", value):
            return True
        if re.search(r"\d{3,5}[X×]\d{3,5}[X×]\d{3,5}", value):
            return True
        score = token.get("score")
        return (
            isinstance(score, (int, float))
            and float(score) >= 0.80
            and bool(re.search(r"[\u4e00-\u9fff]", value))
        )

    return [token for token in tokens if useful(token)]


_VEHICLE_FIELD_ALIASES = {
    "plate_number": (
        "号牌号码",
        "PLATENUMBER",
        "PLATENO",
        "PLANO",
    ),
    "owner": ("所有人", "OWNER"),
    "address": ("住址", "ADDRESS"),
    "vin": (
        "车辆识别代号",
        "车辆识别代码",
        "VEHICLEIDENTIFICATIONNUMBER",
        "VIN",
    ),
    "engine_number": ("发动机号码", "发动机号", "ENGINENO"),
}

_VEHICLE_FIELD_NAMES = {
    "plate_number": "号牌号码",
    "owner": "所有人",
    "address": "住址",
    "vin": "车辆识别代号",
    "engine_number": "发动机号码",
}

_VEHICLE_ALL_FIELD_LABELS = (
    "号牌号码",
    "车辆类型",
    "所有人",
    "住址",
    "使用性质",
    "品牌型号",
    "车辆识别代号",
    "车辆识别代码",
    "发动机号码",
    "发动机号",
    "注册日期",
    "发证日期",
    "档案编号",
    "档案号码",
    "核定载人数",
    "总质量",
    "核定载质量",
    "整备质量",
    "准牵引总质量",
    "外廓尺寸",
    "检验记录",
    "强制报废期止",
    "检验有效期",
    "备注",
    "PLATENUMBER",
    "PLATENO",
    "PLANO",
    "VEHICLETYPE",
    "OWNER",
    "ADDRESS",
    "USECHARACTER",
    "MODEL",
    "VEHICLEIDENTIFICATIONNUMBER",
    "VIN",
    "ENGINENO",
    "REGISTERDATE",
    "ISSUEDATE",
    "FILENO",
    "APPROVEDPASSENGERS",
    "CAPACITY",
    "TOTALMASS",
    "CURBWEIGHT",
    "RATEDLOAD",
    "OVERALLDIMENSION",
    "TRACTIONMASS",
    "REMARKS",
    "INSPECTIONRECORD",
    "VALIDUNTIL",
)

# These are the business-critical fields on the vehicle-licence main page.
# OCR only locates their regions; value formats and confidence scores do not
# participate in the glare decision.
_VEHICLE_GLARE_ACTIONABLE_FIELDS = set(_VEHICLE_FIELD_ALIASES)
_VEHICLE_FIELD_CONTIGUOUS_GLARE_THRESHOLD = 0.08
_VEHICLE_TEXTURE_LOSS_MIN_SPECULAR_EVIDENCE = 0.01
_VEHICLE_TEXTURE_LOSS_MIN_CONTIGUOUS_SPECULAR_EVIDENCE = 0.003
_VEHICLE_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA = 15.0
_VEHICLE_SPECULAR_MIN_BRIGHTNESS_DELTA = 15.0
_VEHICLE_SPECULAR_MIN_COMPONENT_FILL_RATIO = 0.35

_VEHICLE_PLATE_PROVINCE_PREFIXES = (
    "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    "使领学警港澳"
)


def _vehicle_plate_candidate_priority(value: str) -> int | None:
    """Rank plausible plate values while rejecting printed label variants."""

    compact = _compact_ocr_text(value)
    if re.fullmatch(
        rf"[{_VEHICLE_PLATE_PROVINCE_PREFIXES}]"
        r"[A-Z][A-HJ-NP-Z0-9]{5,6}",
        compact,
    ):
        return 0
    # Keep an OCR-damaged partial plate as a fallback only. Requiring a valid
    # province prefix prevents English labels such as PLANO from becoming the
    # selected value merely because they are geometrically close to the label.
    if re.fullmatch(
        rf"[{_VEHICLE_PLATE_PROVINCE_PREFIXES}]"
        r"[A-Z0-9*]{2,7}",
        compact,
    ):
        return 1
    return None


def _vehicle_is_only_field_label(value: str) -> bool:
    """Return true when OCR text consists solely of printed field labels."""

    remainder = _compact_ocr_text(value)
    if not remainder:
        return False
    for label in sorted(_VEHICLE_ALL_FIELD_LABELS, key=len, reverse=True):
        remainder = remainder.replace(label, "")
    return not remainder


def _vehicle_field_label(token: dict[str, Any]) -> tuple[str, str] | None:
    value = _compact_ocr_text(token.get("text"))
    candidates = sorted(
        (
            (field, alias)
            for field, aliases in _VEHICLE_FIELD_ALIASES.items()
            for alias in aliases
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for field, alias in candidates:
        if alias in value:
            return field, alias
    return None


def _vehicle_license_field_assignments(
    tokens: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Locate vehicle fields without validating their recognized values."""

    assignments = {
        field: {"labels": [], "values": []}
        for field in _VEHICLE_FIELD_ALIASES
    }
    labelled_ids: set[int] = set()

    def append_value(
        assignment: dict[str, Any],
        token: dict[str, Any],
        field_value: str,
        **metadata: Any,
    ) -> None:
        source_id = id(token)
        if any(
            existing.get("_source_id") == source_id
            for existing in assignment["values"]
        ):
            return
        assignment["values"].append(
            {
                **token,
                "field_value": field_value,
                "_source_id": source_id,
                **metadata,
            }
        )

    for token in tokens:
        matched = _vehicle_field_label(token)
        if matched is None:
            continue
        field, alias = matched
        assignments[field]["labels"].append(token)
        labelled_ids.add(id(token))
        compact = _compact_ocr_text(token.get("text"))
        remainder = compact.split(alias, 1)[1]
        # Printed vehicle licences commonly put the English translation after
        # the Chinese label (for example ``所有人 Owner``). Remove all aliases
        # of the same field before deciding that this box also contains a value.
        for field_alias in sorted(
            _VEHICLE_FIELD_ALIASES[field],
            key=len,
            reverse=True,
        ):
            remainder = remainder.replace(field_alias, "")
        if remainder and not _vehicle_is_only_field_label(remainder):
            append_value(
                assignments[field],
                token,
                remainder,
                inline=True,
            )

    # When PP-OCR splits a label and its (possibly partial) value into separate
    # boxes, associate the nearest box on the same text line. Intentionally do
    # not apply plate/VIN/name/address syntax rules here: glare is precisely one
    # reason that a recognized value can be incomplete or malformed.
    unlabelled = [token for token in tokens if id(token) not in labelled_ids]
    for field, assignment in assignments.items():
        for label in assignment["labels"]:
            nearby: list[tuple[int, float, dict[str, Any]]] = []
            for token in unlabelled:
                compact = _compact_ocr_text(token.get("text"))
                if not compact:
                    continue
                if _vehicle_is_only_field_label(compact):
                    continue
                # OCR may return the English translation as a separate, partly
                # clipped token (``Owne`` for ``Owner``). It is still printed
                # label text, not the owner's recognized value.
                if re.fullmatch(r"[A-Z]+", compact) and any(
                    len(compact) >= 2 and alias.startswith(compact)
                    for aliases in _VEHICLE_FIELD_ALIASES.values()
                    for alias in aliases
                    if re.fullmatch(r"[A-Z]+", alias)
                ):
                    continue
                # A security strip often splits a Chinese field label into
                # fragments such as ``住`` + ``址`` or ``所有`` + ``人``.
                # Those fragments remain label text and must never become the
                # corresponding field value.
                if any(
                    compact != alias and compact in alias
                    for aliases in _VEHICLE_FIELD_ALIASES.values()
                    for alias in aliases
                    if re.search(r"[\u4e00-\u9fff]", alias)
                ):
                    continue
                if field == "address" and len(compact) < 2:
                    continue
                priority = 0
                if field == "plate_number":
                    plate_priority = _vehicle_plate_candidate_priority(compact)
                    if plate_priority is None:
                        continue
                    priority = plate_priority
                compatible, distance = _driver_tokens_on_same_line(label, token)
                if compatible:
                    nearby.append((priority, distance, token))
            if nearby:
                token = min(nearby, key=lambda item: (item[0], item[1]))[2]
                append_value(
                    assignment,
                    token,
                    _compact_ocr_text(token.get("text")),
                )

    # Main and secondary pages repeat the plate number, while one of their
    # printed labels may be split or missed by OCR. Add every independently
    # recognized, format-valid plate value so both real value boxes remain
    # measurable and label boxes never need to stand in for them.
    plate_assignment = assignments["plate_number"]
    for token in unlabelled:
        compact = _compact_ocr_text(token.get("text"))
        if _vehicle_plate_candidate_priority(compact) == 0:
            append_value(
                plate_assignment,
                token,
                compact,
                inferred=True,
            )
    return assignments


def _electronic_vehicle_license_keywords(
    ppocr_result: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect an electronic vehicle licence from explicit screen evidence."""

    evidence = _compact_ocr_text(ppocr_result.get("text"))
    if tokens:
        evidence += "".join(
            _compact_ocr_text(token.get("text")) for token in tokens
        )
    explicit_title = "电子行驶证" in evidence
    supporting_keywords = [
        keyword
        for keyword in ("状态", "二维码", "生成日期", "下载使用")
        if keyword in evidence
    ]
    return {
        "detected": explicit_title,
        "explicit_title": explicit_title,
        "matched_keywords": supporting_keywords,
        "strategy": "explicit_electronic_vehicle_license_title",
    }


def _mask_overlap_metrics(
    mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
) -> dict[str, Any]:
    """Measure a binary defect mask inside selected OCR field boxes."""

    rows: list[dict[str, Any]] = []
    for token in tokens:
        overlap = _token_mask_overlap(mask, token, coordinate_scale)
        if overlap is None:
            continue
        score = token.get("score")
        rows.append(
            {
                "text": str(token.get("text") or ""),
                "overlap": overlap,
                "score": (
                    float(score) if isinstance(score, (int, float)) else None
                ),
            }
        )

    threshold = 0.08
    confidence_threshold = 0.80
    # Keep very high overlap as diagnostic evidence, but do not reject a
    # readable field solely because the pale licence substrate or security
    # pattern was classified as bright. A field becomes actionable only when
    # the same box also has missing/clearly degraded OCR confidence.
    severe_overlap_threshold = 0.45
    candidates = [row for row in rows if float(row["overlap"]) >= threshold]
    severe = [
        row
        for row in candidates
        if float(row["overlap"]) >= severe_overlap_threshold
    ]
    triggered = [
        row
        for row in candidates
        if row["score"] is None or float(row["score"]) < confidence_threshold
    ]
    readable_glare = [
        row
        for row in candidates
        if row["score"] is not None
        and float(row["score"]) >= confidence_threshold
    ]

    def summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "text": str(row["text"]),
                "overlap": round(float(row["overlap"]), 4),
                "score": (
                    round(float(row["score"]), 4)
                    if row["score"] is not None
                    else None
                ),
            }
            for row in sorted(
                items,
                key=lambda row: float(row["overlap"]),
                reverse=True,
            )[:5]
        ]

    return {
        "field_token_count": len(rows),
        "threshold": threshold,
        "confidence_threshold": confidence_threshold,
        "severe_overlap_threshold": severe_overlap_threshold,
        "severe_overlap_is_diagnostic_only": True,
        "max_overlap": round(
            max((float(row["overlap"]) for row in rows), default=0.0),
            4,
        ),
        "candidate_count": len(candidates),
        "candidate_fields": summaries(candidates),
        "severe_count": len(severe),
        "severe_fields": summaries(severe),
        "triggered_count": len(triggered),
        "triggered_fields": summaries(triggered),
        "readable_glare_count": len(readable_glare),
        "readable_glare_fields": summaries(readable_glare),
    }


def _token_mask_overlap(
    mask: np.ndarray,
    token: dict[str, Any],
    coordinate_scale: float,
    *,
    horizontal_padding_ratio: float = 0.10,
    minimum_horizontal_padding_heights: float = 0.0,
    vertical_padding_ratio: float = 0.15,
    padding_follows_text_axis: bool = False,
) -> float | None:
    metrics = _token_mask_overlap_details(
        mask,
        token,
        coordinate_scale,
        horizontal_padding_ratio=horizontal_padding_ratio,
        minimum_horizontal_padding_heights=minimum_horizontal_padding_heights,
        vertical_padding_ratio=vertical_padding_ratio,
        padding_follows_text_axis=padding_follows_text_axis,
    )
    return None if metrics is None else float(metrics["overlap"])


def _token_mask_overlap_details(
    mask: np.ndarray,
    token: dict[str, Any],
    coordinate_scale: float,
    *,
    horizontal_padding_ratio: float = 0.10,
    minimum_horizontal_padding_heights: float = 0.0,
    vertical_padding_ratio: float = 0.15,
    padding_follows_text_axis: bool = False,
) -> dict[str, float | int] | None:
    """Measure total and largest-contiguous mask coverage in one OCR ROI."""

    left, top, right, bottom = token["bbox"]
    box_width = right - left
    box_height = bottom - top
    if padding_follows_text_axis and box_height > box_width:
        padding_x = max(2.0, box_width * vertical_padding_ratio)
        padding_y = max(
            2.0,
            box_height * horizontal_padding_ratio,
            box_width * minimum_horizontal_padding_heights,
        )
    else:
        padding_x = max(
            2.0,
            box_width * horizontal_padding_ratio,
            box_height * minimum_horizontal_padding_heights,
        )
        padding_y = max(2.0, box_height * vertical_padding_ratio)
    x1 = max(0, round((left - padding_x) * coordinate_scale))
    y1 = max(0, round((top - padding_y) * coordinate_scale))
    x2 = min(mask.shape[1], round((right + padding_x) * coordinate_scale))
    y2 = min(mask.shape[0], round((bottom + padding_y) * coordinate_scale))
    roi = mask[y1:y2, x1:x2]
    if roi.size < 16:
        return None
    binary_roi = (roi != 0).astype(np.uint8)
    mask_pixels = int(np.count_nonzero(binary_roi))
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_roi,
        connectivity=8,
    )
    largest_component_pixels = (
        int(stats[1:, cv2.CC_STAT_AREA].max())
        if component_count > 1
        else 0
    )
    roi_pixels = int(binary_roi.size)
    return {
        "overlap": float(mask_pixels) / float(max(1, roi_pixels)),
        "largest_contiguous_overlap": (
            float(largest_component_pixels) / float(max(1, roi_pixels))
        ),
        "mask_pixels": mask_pixels,
        "largest_component_pixels": largest_component_pixels,
        "roi_pixels": roi_pixels,
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,
    }


def _texture_specular_spatial_evidence(
    specular_mask: np.ndarray,
    texture_loss_mask: np.ndarray,
    grayscale_image: np.ndarray | None,
    overlap_details: dict[str, Any],
    *,
    minimum_component_ratio: float,
    minimum_linked_specular_ratio: float,
    minimum_brightness_delta: float,
) -> dict[str, Any]:
    """Require a washout component, highlight core and brightness anomaly to align."""

    x1 = int(overlap_details["roi_x1"])
    y1 = int(overlap_details["roi_y1"])
    x2 = int(overlap_details["roi_x2"])
    y2 = int(overlap_details["roi_y2"])
    specular_roi = (specular_mask[y1:y2, x1:x2] != 0).astype(np.uint8)
    texture_roi = (texture_loss_mask[y1:y2, x1:x2] != 0).astype(np.uint8)
    if specular_roi.size < 16 or specular_roi.shape != texture_roi.shape:
        return {
            "spatially_linked": False,
            "max_linked_specular_ratio": 0.0,
            "max_component_brightness_delta": 0.0,
        }

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        texture_roi,
        connectivity=8,
    )
    roi_pixels = max(1, int(texture_roi.size))
    # A small dilation allows the white highlight core to touch the boundary
    # of the low-gradient washout component without accepting evidence from a
    # different part of the OCR box.
    radius = max(2, round(min(texture_roi.shape) * 0.04))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    grayscale_roi = (
        grayscale_image[y1:y2, x1:x2]
        if grayscale_image is not None
        and grayscale_image.shape[:2] == specular_mask.shape[:2]
        else None
    )
    roi_reference_brightness = (
        float(np.median(grayscale_roi))
        if grayscale_roi is not None and grayscale_roi.size
        else None
    )

    max_linked_specular_ratio = 0.0
    max_component_brightness_delta = 0.0
    spatially_linked = False
    for component_index in range(1, component_count):
        component_pixels = int(stats[component_index, cv2.CC_STAT_AREA])
        component_ratio = float(component_pixels) / float(roi_pixels)
        if component_ratio < minimum_component_ratio:
            continue
        component = (labels == component_index).astype(np.uint8)
        nearby = cv2.dilate(component, kernel, iterations=1)
        linked_specular_pixels = int(
            np.count_nonzero((nearby != 0) & (specular_roi != 0))
        )
        linked_specular_ratio = float(linked_specular_pixels) / float(roi_pixels)
        max_linked_specular_ratio = max(
            max_linked_specular_ratio,
            linked_specular_ratio,
        )
        brightness_delta = 0.0
        if grayscale_roi is not None and roi_reference_brightness is not None:
            component_values = grayscale_roi[component != 0]
            if component_values.size:
                brightness_delta = (
                    float(np.median(component_values))
                    - roi_reference_brightness
                )
        max_component_brightness_delta = max(
            max_component_brightness_delta,
            brightness_delta,
        )
        brightness_supported = (
            grayscale_roi is None
            or brightness_delta >= minimum_brightness_delta
        )
        if (
            linked_specular_ratio >= minimum_linked_specular_ratio
            and brightness_supported
        ):
            spatially_linked = True

    return {
        "spatially_linked": spatially_linked,
        "max_linked_specular_ratio": max_linked_specular_ratio,
        "max_component_brightness_delta": max_component_brightness_delta,
    }


def _driver_elongated_text_glare_metrics(
    mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
) -> dict[str, Any]:
    """Detect several thin highlight streaks crossing recognized text."""

    text_region_mask = np.zeros_like(mask, dtype=np.uint8)
    for token in tokens:
        overlap = _token_mask_overlap_details(
            mask,
            token,
            coordinate_scale,
            horizontal_padding_ratio=0.10,
            minimum_horizontal_padding_heights=0.50,
            vertical_padding_ratio=0.15,
            padding_follows_text_axis=True,
        )
        if overlap is None:
            continue
        cv2.rectangle(
            text_region_mask,
            (int(overlap["roi_x1"]), int(overlap["roi_y1"])),
            (int(overlap["roi_x2"]), int(overlap["roi_y2"])),
            1,
            -1,
        )

    binary_mask = (mask != 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
    image_pixels = max(1, int(binary_mask.size))
    image_side = max(binary_mask.shape)
    components: list[dict[str, Any]] = []
    for component_index in range(1, component_count):
        width = max(1, int(stats[component_index, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[component_index, cv2.CC_STAT_HEIGHT]))
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        long_side = max(width, height)
        thickness = min(width, height)
        aspect_ratio = float(long_side) / float(max(1, thickness))
        if (
            area < 20
            or aspect_ratio < _DRIVER_ELONGATED_GLARE_MIN_ASPECT_RATIO
            or long_side
            < image_side * _DRIVER_ELONGATED_GLARE_MIN_LENGTH_RATIO
            or thickness
            > image_side * _DRIVER_ELONGATED_GLARE_MAX_THICKNESS_RATIO
        ):
            continue
        component = labels == component_index
        text_overlap = float(
            np.count_nonzero(component & (text_region_mask != 0))
        ) / float(max(1, area))
        if text_overlap < _DRIVER_ELONGATED_GLARE_MIN_TEXT_OVERLAP:
            continue
        components.append(
            {
                "area": area,
                "width": width,
                "height": height,
                "aspect_ratio": round(aspect_ratio, 2),
                "text_overlap": round(text_overlap, 4),
            }
        )

    total_area_ratio = float(
        sum(int(component["area"]) for component in components)
    ) / float(image_pixels)
    detected = (
        len(components) >= _DRIVER_ELONGATED_GLARE_MIN_COMPONENT_COUNT
        and total_area_ratio
        >= _DRIVER_ELONGATED_GLARE_MIN_TOTAL_AREA_RATIO
    )
    return {
        "detected": detected,
        "component_count": len(components),
        "total_area_ratio": round(total_area_ratio, 4),
        "minimum_component_count": (
            _DRIVER_ELONGATED_GLARE_MIN_COMPONENT_COUNT
        ),
        "minimum_total_area_ratio": (
            _DRIVER_ELONGATED_GLARE_MIN_TOTAL_AREA_RATIO
        ),
        "minimum_aspect_ratio": _DRIVER_ELONGATED_GLARE_MIN_ASPECT_RATIO,
        "minimum_length_ratio": _DRIVER_ELONGATED_GLARE_MIN_LENGTH_RATIO,
        "maximum_thickness_ratio": (
            _DRIVER_ELONGATED_GLARE_MAX_THICKNESS_RATIO
        ),
        "minimum_text_overlap": _DRIVER_ELONGATED_GLARE_MIN_TEXT_OVERLAP,
        "components": sorted(
            components,
            key=lambda component: int(component["area"]),
            reverse=True,
        )[:20],
    }


def _specular_component_evidence(
    specular_mask: np.ndarray,
    grayscale_image: np.ndarray | None,
    overlap_details: dict[str, Any],
    *,
    minimum_component_ratio: float,
    minimum_fill_ratio: float,
    minimum_brightness_delta: float,
) -> dict[str, Any]:
    """Distinguish a solid local highlight from pale security-pattern networks."""

    x1 = int(overlap_details["roi_x1"])
    y1 = int(overlap_details["roi_y1"])
    x2 = int(overlap_details["roi_x2"])
    y2 = int(overlap_details["roi_y2"])
    specular_roi = (specular_mask[y1:y2, x1:x2] != 0).astype(np.uint8)
    if specular_roi.size < 16:
        return {
            "supported": False,
            "max_component_fill_ratio": 0.0,
            "max_component_brightness_delta": 0.0,
        }

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        specular_roi,
        connectivity=8,
    )
    roi_pixels = max(1, int(specular_roi.size))
    grayscale_roi = (
        grayscale_image[y1:y2, x1:x2]
        if grayscale_image is not None
        and grayscale_image.shape[:2] == specular_mask.shape[:2]
        else None
    )
    roi_reference_brightness = (
        float(np.median(grayscale_roi))
        if grayscale_roi is not None and grayscale_roi.size
        else None
    )
    supported = False
    max_component_fill_ratio = 0.0
    max_component_brightness_delta = 0.0
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        component_ratio = float(area) / float(roi_pixels)
        if component_ratio < minimum_component_ratio:
            continue
        width = max(1, int(stats[component_index, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[component_index, cv2.CC_STAT_HEIGHT]))
        fill_ratio = float(area) / float(width * height)
        max_component_fill_ratio = max(max_component_fill_ratio, fill_ratio)
        brightness_delta = 0.0
        if grayscale_roi is not None and roi_reference_brightness is not None:
            component_values = grayscale_roi[labels == component_index]
            if component_values.size:
                brightness_delta = (
                    float(np.median(component_values))
                    - roi_reference_brightness
                )
        max_component_brightness_delta = max(
            max_component_brightness_delta,
            brightness_delta,
        )
        brightness_supported = (
            grayscale_roi is None
            or brightness_delta >= minimum_brightness_delta
        )
        if (
            fill_ratio >= minimum_fill_ratio
            and brightness_supported
        ):
            supported = True

    return {
        "supported": supported,
        "max_component_fill_ratio": max_component_fill_ratio,
        "max_component_brightness_delta": max_component_brightness_delta,
    }


def _driver_license_semantic_glare_metrics(
    mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
    texture_loss_mask: np.ndarray | None = None,
    grayscale_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure glare in OCR-located driver-license key-field regions.

    OCR text is used to associate boxes with fields. A complete, high-confidence
    value can suppress only the texture-washout branch; direct specular glare
    remains actionable.
    """

    threshold = _DRIVER_FIELD_GLARE_THRESHOLD
    contiguous_threshold = _DRIVER_FIELD_CONTIGUOUS_GLARE_THRESHOLD
    texture_loss_threshold = _DRIVER_FIELD_TEXTURE_LOSS_THRESHOLD
    contiguous_texture_loss_threshold = (
        _DRIVER_FIELD_CONTIGUOUS_TEXTURE_LOSS_THRESHOLD
    )
    texture_loss_min_specular_evidence = (
        _DRIVER_FIELD_TEXTURE_LOSS_MIN_SPECULAR_EVIDENCE
    )
    assignments = _driver_license_field_assignments(tokens)
    fields: list[dict[str, Any]] = []
    triggered: list[dict[str, Any]] = []
    for field, assignment in assignments.items():
        labels = assignment["labels"]
        value_tokens = assignment["values"]
        if not labels and not value_tokens:
            continue
        values = [
            str(token.get("field_value") or token.get("text") or "")
            for token in value_tokens
        ]
        reliably_readable_value = _driver_has_reliably_readable_value(
            field,
            value_tokens,
        )
        overlap_tokens = list(value_tokens or labels)
        if field == "address" and value_tokens:
            overlap_tokens.extend(
                projected
                for token in value_tokens
                if (projected := _driver_partial_address_projection(token))
                is not None
            )
        if field == "valid_period" and value_tokens:
            overlap_tokens.extend(
                projected
                for token in value_tokens
                if (
                    projected := _driver_partial_valid_period_projection(token)
                )
                is not None
            )
        box_metrics: list[dict[str, Any]] = []
        for token in overlap_tokens:
            minimum_padding_heights = (
                _driver_minimum_horizontal_padding_heights(field, token)
            )
            overlap = _token_mask_overlap_details(
                mask,
                token,
                coordinate_scale,
                horizontal_padding_ratio=(
                    _DRIVER_FIELD_HORIZONTAL_PADDING_RATIO
                ),
                minimum_horizontal_padding_heights=(
                    minimum_padding_heights
                ),
                vertical_padding_ratio=_DRIVER_FIELD_VERTICAL_PADDING_RATIO,
                padding_follows_text_axis=True,
            )
            if overlap is None:
                continue
            specular_component = _specular_component_evidence(
                mask,
                grayscale_image,
                overlap,
                minimum_component_ratio=contiguous_threshold,
                minimum_fill_ratio=(
                    _DRIVER_SPECULAR_MIN_COMPONENT_FILL_RATIO
                ),
                minimum_brightness_delta=(
                    _DRIVER_SPECULAR_MIN_BRIGHTNESS_DELTA
                ),
            )
            texture_overlap = (
                _token_mask_overlap_details(
                    texture_loss_mask,
                    token,
                    coordinate_scale,
                    horizontal_padding_ratio=(
                        _DRIVER_FIELD_HORIZONTAL_PADDING_RATIO
                    ),
                    minimum_horizontal_padding_heights=(
                        minimum_padding_heights
                    ),
                    vertical_padding_ratio=_DRIVER_FIELD_VERTICAL_PADDING_RATIO,
                    padding_follows_text_axis=True,
                )
                if texture_loss_mask is not None
                else None
            )
            spatial_evidence = (
                _texture_specular_spatial_evidence(
                    mask,
                    texture_loss_mask,
                    grayscale_image,
                    overlap,
                    minimum_component_ratio=(
                        contiguous_texture_loss_threshold
                    ),
                    minimum_linked_specular_ratio=(
                        _DRIVER_FIELD_TEXTURE_LOSS_MIN_LINKED_SPECULAR_EVIDENCE
                    ),
                    minimum_brightness_delta=(
                        _DRIVER_FIELD_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
                    ),
                )
                if texture_loss_mask is not None
                else {
                    "spatially_linked": False,
                    "max_linked_specular_ratio": 0.0,
                    "max_component_brightness_delta": 0.0,
                }
            )
            box_metrics.append(
                {
                    "text": str(
                        token.get("field_value") or token.get("text") or ""
                    ),
                    **overlap,
                    "texture_loss_overlap": (
                        float(texture_overlap["overlap"])
                        if texture_overlap is not None
                        else 0.0
                    ),
                    "largest_contiguous_texture_loss_overlap": (
                        float(texture_overlap["largest_contiguous_overlap"])
                        if texture_overlap is not None
                        else 0.0
                    ),
                    "texture_specular_spatially_linked": bool(
                        spatial_evidence["spatially_linked"]
                    ),
                    "linked_specular_overlap": float(
                        spatial_evidence["max_linked_specular_ratio"]
                    ),
                    "texture_component_brightness_delta": float(
                        spatial_evidence["max_component_brightness_delta"]
                    ),
                    "specular_component_supported": bool(
                        specular_component["supported"]
                    ),
                    "specular_component_fill_ratio": float(
                        specular_component["max_component_fill_ratio"]
                    ),
                    "specular_component_brightness_delta": float(
                        specular_component["max_component_brightness_delta"]
                    ),
                    "minimum_horizontal_padding_heights": float(
                        minimum_padding_heights
                    ),
                }
            )
        max_overlap = max(
            (float(item["overlap"]) for item in box_metrics),
            default=0.0,
        )
        max_contiguous_overlap = max(
            (
                float(item["largest_contiguous_overlap"])
                for item in box_metrics
            ),
            default=0.0,
        )
        max_texture_loss_overlap = max(
            (float(item["texture_loss_overlap"]) for item in box_metrics),
            default=0.0,
        )
        max_contiguous_texture_loss_overlap = max(
            (
                float(item["largest_contiguous_texture_loss_overlap"])
                for item in box_metrics
            ),
            default=0.0,
        )
        specular_glare_detected = any(
            float(item["overlap"]) >= threshold
            and float(item["largest_contiguous_overlap"])
            >= contiguous_threshold
            and bool(item["specular_component_supported"])
            for item in box_metrics
        )
        texture_washout_candidate = any(
            float(item["overlap"]) >= texture_loss_min_specular_evidence
            and float(item["texture_loss_overlap"]) >= texture_loss_threshold
            and float(item["largest_contiguous_texture_loss_overlap"])
            >= contiguous_texture_loss_threshold
            and bool(item["texture_specular_spatially_linked"])
            for item in box_metrics
        )
        # Real plastic-cover glare may sit immediately beside a washed-out
        # field instead of forming one connected mask after downscaling. Keep
        # this relaxed channel field-local and require independent evidence
        # from highlight coverage, texture loss, and local brightness delta.
        near_specular_texture_washout = any(
            float(item["overlap"])
            >= _DRIVER_NEAR_SPECULAR_GLARE_THRESHOLD
            and float(item["largest_contiguous_overlap"])
            >= _DRIVER_NEAR_SPECULAR_CONTIGUOUS_GLARE_THRESHOLD
            and float(item["texture_loss_overlap"])
            >= texture_loss_threshold
            and float(item["largest_contiguous_texture_loss_overlap"])
            >= contiguous_texture_loss_threshold
            and float(item["texture_component_brightness_delta"])
            >= _DRIVER_FIELD_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
            for item in box_metrics
        )
        texture_washout_detected = (
            (texture_washout_candidate or near_specular_texture_washout)
            and not reliably_readable_value
        )
        row = {
            "field": field,
            "field_name": _DRIVER_FIELD_NAMES[field],
            "texts": values,
            "scores": [
                round(float(token["score"]), 4)
                if isinstance(token.get("score"), (int, float))
                else None
                for token in value_tokens
            ],
            "label_count": len(labels),
            "inferred_value_count": sum(
                bool(token.get("inferred")) for token in value_tokens
            ),
            "box_count": len(overlap_tokens),
            "box_glare_metrics": [
                {
                    "text": str(item["text"]),
                    "glare_overlap": round(float(item["overlap"]), 4),
                    "largest_contiguous_glare_overlap": round(
                        float(item["largest_contiguous_overlap"]),
                        4,
                    ),
                    "texture_loss_overlap": round(
                        float(item["texture_loss_overlap"]),
                        4,
                    ),
                    "largest_contiguous_texture_loss_overlap": round(
                        float(item["largest_contiguous_texture_loss_overlap"]),
                        4,
                    ),
                    "texture_specular_spatially_linked": bool(
                        item["texture_specular_spatially_linked"]
                    ),
                    "linked_specular_overlap": round(
                        float(item["linked_specular_overlap"]),
                        4,
                    ),
                    "texture_component_brightness_delta": round(
                        float(item["texture_component_brightness_delta"]),
                        2,
                    ),
                    "specular_component_supported": bool(
                        item["specular_component_supported"]
                    ),
                    "specular_component_fill_ratio": round(
                        float(item["specular_component_fill_ratio"]),
                        4,
                    ),
                    "specular_component_brightness_delta": round(
                        float(item["specular_component_brightness_delta"]),
                        2,
                    ),
                    "minimum_horizontal_padding_heights": round(
                        float(item["minimum_horizontal_padding_heights"]),
                        2,
                    ),
                }
                for item in box_metrics
            ],
            "max_glare_overlap": round(max_overlap, 4),
            "max_contiguous_glare_overlap": round(
                max_contiguous_overlap,
                4,
            ),
            "max_texture_loss_overlap": round(max_texture_loss_overlap, 4),
            "max_contiguous_texture_loss_overlap": round(
                max_contiguous_texture_loss_overlap,
                4,
            ),
            "glare_threshold": threshold,
            "contiguous_glare_threshold": contiguous_threshold,
            "texture_loss_threshold": texture_loss_threshold,
            "contiguous_texture_loss_threshold": (
                contiguous_texture_loss_threshold
            ),
            "texture_loss_min_specular_evidence": (
                texture_loss_min_specular_evidence
            ),
            "texture_loss_min_linked_specular_evidence": (
                _DRIVER_FIELD_TEXTURE_LOSS_MIN_LINKED_SPECULAR_EVIDENCE
            ),
            "texture_loss_min_brightness_delta": (
                _DRIVER_FIELD_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
            ),
            "specular_min_brightness_delta": (
                _DRIVER_SPECULAR_MIN_BRIGHTNESS_DELTA
            ),
            "specular_min_component_fill_ratio": (
                _DRIVER_SPECULAR_MIN_COMPONENT_FILL_RATIO
            ),
            "reliably_readable_value": reliably_readable_value,
            "readable_value_confidence_threshold": (
                _DRIVER_READABLE_VALUE_CONFIDENCE
            ),
            "texture_washout_candidate": texture_washout_candidate,
            "near_specular_texture_washout": (
                near_specular_texture_washout
            ),
            "texture_washout_suppressed_by_readable_value": (
                texture_washout_candidate and reliably_readable_value
            ),
            "actionable": field in _DRIVER_GLARE_ACTIONABLE_FIELDS,
        }
        fields.append(row)
        if row["actionable"] and (
            specular_glare_detected or texture_washout_detected
        ):
            triggered.append(row)
    return {
        "threshold": threshold,
        "decision_basis": "specular_highlight_or_texture_washout",
        "contiguous_glare_threshold": contiguous_threshold,
        "texture_loss_threshold": texture_loss_threshold,
        "contiguous_texture_loss_threshold": contiguous_texture_loss_threshold,
        "texture_loss_min_specular_evidence": (
            texture_loss_min_specular_evidence
        ),
        "texture_loss_min_linked_specular_evidence": (
            _DRIVER_FIELD_TEXTURE_LOSS_MIN_LINKED_SPECULAR_EVIDENCE
        ),
        "texture_loss_min_brightness_delta": (
            _DRIVER_FIELD_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
        ),
        "specular_min_brightness_delta": (
            _DRIVER_SPECULAR_MIN_BRIGHTNESS_DELTA
        ),
        "specular_min_component_fill_ratio": (
            _DRIVER_SPECULAR_MIN_COMPONENT_FILL_RATIO
        ),
        "horizontal_padding_ratio": _DRIVER_FIELD_HORIZONTAL_PADDING_RATIO,
        "minimum_horizontal_padding_heights": (
            _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
        ),
        "complete_field_minimum_horizontal_padding_heights": (
            _DRIVER_COMPLETE_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
        ),
        "vertical_padding_ratio": _DRIVER_FIELD_VERTICAL_PADDING_RATIO,
        "padding_follows_text_axis": True,
        "field_count": len(fields),
        "fields": fields,
        "triggered_count": len(triggered),
        "triggered_fields": sorted(
            triggered,
            key=lambda row: float(row["max_glare_overlap"]),
            reverse=True,
        ),
    }


def _vehicle_license_semantic_glare_metrics(
    mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
    texture_loss_mask: np.ndarray | None = None,
    grayscale_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure glare in OCR-located vehicle-licence key-field regions.

    This deliberately mirrors the driver-licence decision: clipped white
    highlights and low-texture washout are two independent glare channels.
    OCR confidence and field-value validation are excluded from the decision.
    """

    threshold = _DRIVER_FIELD_GLARE_THRESHOLD
    # Vehicle licences often have large near-white security-pattern regions.
    # Requiring an 8% connected highlight prevents those diffuse printed areas
    # from behaving like a narrow, truly clipped reflection over a field.
    contiguous_threshold = _VEHICLE_FIELD_CONTIGUOUS_GLARE_THRESHOLD
    texture_loss_threshold = _DRIVER_FIELD_TEXTURE_LOSS_THRESHOLD
    contiguous_texture_loss_threshold = (
        _DRIVER_FIELD_CONTIGUOUS_TEXTURE_LOSS_THRESHOLD
    )
    texture_loss_min_specular_evidence = (
        _VEHICLE_TEXTURE_LOSS_MIN_SPECULAR_EVIDENCE
    )
    texture_loss_min_contiguous_specular_evidence = (
        _VEHICLE_TEXTURE_LOSS_MIN_CONTIGUOUS_SPECULAR_EVIDENCE
    )
    assignments = _vehicle_license_field_assignments(tokens)
    fields: list[dict[str, Any]] = []
    triggered: list[dict[str, Any]] = []
    for field, assignment in assignments.items():
        labels = assignment["labels"]
        value_tokens = assignment["values"]
        if not labels and not value_tokens:
            continue
        values = [
            str(token.get("field_value") or token.get("text") or "")
            for token in value_tokens
        ]
        overlap_tokens = value_tokens or labels
        box_metrics: list[dict[str, Any]] = []
        for token in overlap_tokens:
            overlap = _token_mask_overlap_details(
                mask,
                token,
                coordinate_scale,
                horizontal_padding_ratio=_DRIVER_FIELD_HORIZONTAL_PADDING_RATIO,
                minimum_horizontal_padding_heights=(
                    _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
                ),
                vertical_padding_ratio=_DRIVER_FIELD_VERTICAL_PADDING_RATIO,
                padding_follows_text_axis=True,
            )
            if overlap is None:
                continue
            specular_component = _specular_component_evidence(
                mask,
                grayscale_image,
                overlap,
                minimum_component_ratio=contiguous_threshold,
                minimum_fill_ratio=(
                    _VEHICLE_SPECULAR_MIN_COMPONENT_FILL_RATIO
                ),
                minimum_brightness_delta=(
                    _VEHICLE_SPECULAR_MIN_BRIGHTNESS_DELTA
                ),
            )
            texture_overlap = (
                _token_mask_overlap_details(
                    texture_loss_mask,
                    token,
                    coordinate_scale,
                    horizontal_padding_ratio=(
                        _DRIVER_FIELD_HORIZONTAL_PADDING_RATIO
                    ),
                    minimum_horizontal_padding_heights=(
                        _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
                    ),
                    vertical_padding_ratio=(
                        _DRIVER_FIELD_VERTICAL_PADDING_RATIO
                    ),
                    padding_follows_text_axis=True,
                )
                if texture_loss_mask is not None
                else None
            )
            spatial_evidence = (
                _texture_specular_spatial_evidence(
                    mask,
                    texture_loss_mask,
                    grayscale_image,
                    overlap,
                    minimum_component_ratio=(
                        contiguous_texture_loss_threshold
                    ),
                    minimum_linked_specular_ratio=(
                        _VEHICLE_TEXTURE_LOSS_MIN_CONTIGUOUS_SPECULAR_EVIDENCE
                    ),
                    minimum_brightness_delta=(
                        _VEHICLE_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
                    ),
                )
                if texture_loss_mask is not None
                else {
                    "spatially_linked": False,
                    "max_linked_specular_ratio": 0.0,
                    "max_component_brightness_delta": 0.0,
                }
            )
            box_metrics.append(
                {
                    "text": str(
                        token.get("field_value") or token.get("text") or ""
                    ),
                    **overlap,
                    "texture_loss_overlap": (
                        float(texture_overlap["overlap"])
                        if texture_overlap is not None
                        else 0.0
                    ),
                    "largest_contiguous_texture_loss_overlap": (
                        float(texture_overlap["largest_contiguous_overlap"])
                        if texture_overlap is not None
                        else 0.0
                    ),
                    "texture_specular_spatially_linked": bool(
                        spatial_evidence["spatially_linked"]
                    ),
                    "linked_specular_overlap": float(
                        spatial_evidence["max_linked_specular_ratio"]
                    ),
                    "texture_component_brightness_delta": float(
                        spatial_evidence["max_component_brightness_delta"]
                    ),
                    "specular_component_supported": bool(
                        specular_component["supported"]
                    ),
                    "specular_component_fill_ratio": float(
                        specular_component["max_component_fill_ratio"]
                    ),
                    "specular_component_brightness_delta": float(
                        specular_component["max_component_brightness_delta"]
                    ),
                }
            )

        max_overlap = max(
            (float(item["overlap"]) for item in box_metrics),
            default=0.0,
        )
        max_contiguous_overlap = max(
            (
                float(item["largest_contiguous_overlap"])
                for item in box_metrics
            ),
            default=0.0,
        )
        max_texture_loss_overlap = max(
            (float(item["texture_loss_overlap"]) for item in box_metrics),
            default=0.0,
        )
        max_contiguous_texture_loss_overlap = max(
            (
                float(item["largest_contiguous_texture_loss_overlap"])
                for item in box_metrics
            ),
            default=0.0,
        )
        specular_glare_detected = any(
            float(item["overlap"]) >= threshold
            and float(item["largest_contiguous_overlap"])
            >= contiguous_threshold
            and bool(item["specular_component_supported"])
            for item in box_metrics
        )
        texture_washout_detected = any(
            float(item["overlap"]) >= texture_loss_min_specular_evidence
            and float(item["largest_contiguous_overlap"])
            >= texture_loss_min_contiguous_specular_evidence
            and float(item["texture_loss_overlap"]) >= texture_loss_threshold
            and float(item["largest_contiguous_texture_loss_overlap"])
            >= contiguous_texture_loss_threshold
            and bool(item["texture_specular_spatially_linked"])
            for item in box_metrics
        )
        row = {
            "field": field,
            "field_name": _VEHICLE_FIELD_NAMES[field],
            "texts": values,
            # Kept for diagnostics only; these scores never affect triggered.
            "scores": [
                round(float(token["score"]), 4)
                if isinstance(token.get("score"), (int, float))
                else None
                for token in value_tokens
            ],
            "box_count": len(overlap_tokens),
            "box_glare_metrics": [
                {
                    "text": str(item["text"]),
                    "glare_overlap": round(float(item["overlap"]), 4),
                    "largest_contiguous_glare_overlap": round(
                        float(item["largest_contiguous_overlap"]),
                        4,
                    ),
                    "texture_loss_overlap": round(
                        float(item["texture_loss_overlap"]),
                        4,
                    ),
                    "largest_contiguous_texture_loss_overlap": round(
                        float(item["largest_contiguous_texture_loss_overlap"]),
                        4,
                    ),
                    "texture_specular_spatially_linked": bool(
                        item["texture_specular_spatially_linked"]
                    ),
                    "linked_specular_overlap": round(
                        float(item["linked_specular_overlap"]),
                        4,
                    ),
                    "texture_component_brightness_delta": round(
                        float(item["texture_component_brightness_delta"]),
                        2,
                    ),
                    "specular_component_supported": bool(
                        item["specular_component_supported"]
                    ),
                    "specular_component_fill_ratio": round(
                        float(item["specular_component_fill_ratio"]),
                        4,
                    ),
                    "specular_component_brightness_delta": round(
                        float(item["specular_component_brightness_delta"]),
                        2,
                    ),
                }
                for item in box_metrics
            ],
            "max_glare_overlap": round(max_overlap, 4),
            "max_contiguous_glare_overlap": round(
                max_contiguous_overlap,
                4,
            ),
            "max_texture_loss_overlap": round(max_texture_loss_overlap, 4),
            "max_contiguous_texture_loss_overlap": round(
                max_contiguous_texture_loss_overlap,
                4,
            ),
            "glare_threshold": threshold,
            "contiguous_glare_threshold": contiguous_threshold,
            "texture_loss_threshold": texture_loss_threshold,
            "contiguous_texture_loss_threshold": (
                contiguous_texture_loss_threshold
            ),
            "texture_loss_min_specular_evidence": (
                texture_loss_min_specular_evidence
            ),
            "texture_loss_min_contiguous_specular_evidence": (
                texture_loss_min_contiguous_specular_evidence
            ),
            "texture_loss_min_brightness_delta": (
                _VEHICLE_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
            ),
            "specular_min_brightness_delta": (
                _VEHICLE_SPECULAR_MIN_BRIGHTNESS_DELTA
            ),
            "specular_min_component_fill_ratio": (
                _VEHICLE_SPECULAR_MIN_COMPONENT_FILL_RATIO
            ),
            "actionable": field in _VEHICLE_GLARE_ACTIONABLE_FIELDS,
        }
        fields.append(row)
        if row["actionable"] and (
            specular_glare_detected or texture_washout_detected
        ):
            triggered.append(row)

    return {
        "threshold": threshold,
        "decision_basis": "specular_highlight_or_texture_washout",
        "uses_ocr_confidence": False,
        "uses_value_validation": False,
        "contiguous_glare_threshold": contiguous_threshold,
        "texture_loss_threshold": texture_loss_threshold,
        "contiguous_texture_loss_threshold": contiguous_texture_loss_threshold,
        "texture_loss_min_specular_evidence": (
            texture_loss_min_specular_evidence
        ),
        "texture_loss_min_contiguous_specular_evidence": (
            texture_loss_min_contiguous_specular_evidence
        ),
        "texture_loss_min_brightness_delta": (
            _VEHICLE_TEXTURE_LOSS_MIN_BRIGHTNESS_DELTA
        ),
        "specular_min_brightness_delta": (
            _VEHICLE_SPECULAR_MIN_BRIGHTNESS_DELTA
        ),
        "specular_min_component_fill_ratio": (
            _VEHICLE_SPECULAR_MIN_COMPONENT_FILL_RATIO
        ),
        "horizontal_padding_ratio": _DRIVER_FIELD_HORIZONTAL_PADDING_RATIO,
        "minimum_horizontal_padding_heights": (
            _DRIVER_FIELD_MIN_HORIZONTAL_PADDING_HEIGHTS
        ),
        "vertical_padding_ratio": _DRIVER_FIELD_VERTICAL_PADDING_RATIO,
        "padding_follows_text_axis": True,
        "field_count": len(fields),
        "fields": fields,
        "triggered_count": len(triggered),
        "triggered_fields": sorted(
            triggered,
            key=lambda row: max(
                float(row["max_glare_overlap"]),
                float(row["max_texture_loss_overlap"]),
            ),
            reverse=True,
        ),
    }


def _refine_field_glare_metrics(
    field_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Separate readable diffuse bright backgrounds from local specular glare.

    Security-patterned electronic documents contain dense, near-white lines that
    look like highlights to a pixel mask. When that mask covers most OCR boxes
    while recognition remains uniformly reliable, it is evidence of a bright
    substrate rather than local glare erasing text. Sparse highlights and any
    widespread highlight accompanied by OCR degradation remain actionable.
    """

    refined = dict(field_metrics)
    field_count = int(field_metrics.get("field_token_count") or 0)
    candidate_count = int(
        field_metrics.get("candidate_count")
        if field_metrics.get("candidate_count") is not None
        else field_metrics.get("triggered_count") or 0
    )
    candidate_ratio = candidate_count / max(1, field_count)
    scored_count = int(ocr_metrics.get("scored_token_count") or 0)
    score_mean = float(ocr_metrics.get("score_mean") or 0.0)
    score_p25_value = ocr_metrics.get("score_p25")
    score_p25 = float(score_p25_value) if score_p25_value is not None else 0.0
    low_score_ratio = float(ocr_metrics.get("low_score_ratio") or 0.0)
    low_score_char_ratio = float(ocr_metrics.get("low_score_char_ratio") or 0.0)
    glare_affected_low_score_count = int(
        ocr_metrics.get("glare_affected_low_score_count") or 0
    )

    widespread = (
        field_count >= 6
        and candidate_count >= 4
        and candidate_ratio >= 0.65
    )
    uniformly_readable = (
        scored_count >= 6
        and score_mean >= 0.90
        and score_p25 >= 0.82
        and low_score_ratio <= 0.10
        and low_score_char_ratio <= 0.10
        and glare_affected_low_score_count == 0
    )
    suppressed = widespread and uniformly_readable

    refined.update(
        {
            "candidate_count": candidate_count,
            "candidate_ratio": round(candidate_ratio, 4),
            "suppressed_as_diffuse_bright_background": suppressed,
            "suppression_reason": (
                "widespread_bright_mask_with_uniformly_readable_ocr"
                if suppressed
                else None
            ),
        }
    )
    if suppressed:
        refined["triggered_count"] = 0
        refined["triggered_fields"] = []
    return refined


def _text_region_metrics(
    image: np.ndarray,
    glare_mask: np.ndarray,
    tokens: list[dict[str, Any]],
    coordinate_scale: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Measure OCR confidence and image defects inside detected text boxes."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows: list[dict[str, float | int | None]] = []
    for token in tokens:
        left, top, right, bottom = token["bbox"]
        box_width = right - left
        box_height = bottom - top
        # PP boxes are generally tight. A small margin captures glow/blur that
        # erases the edge of a character without bringing in much background.
        padding_x = max(2.0, box_width * 0.10)
        padding_y = max(2.0, box_height * 0.15)
        x1 = max(0, round((left - padding_x) * coordinate_scale))
        y1 = max(0, round((top - padding_y) * coordinate_scale))
        x2 = min(gray.shape[1], round((right + padding_x) * coordinate_scale))
        y2 = min(gray.shape[0], round((bottom + padding_y) * coordinate_scale))
        roi = gray[y1:y2, x1:x2]
        roi_glare = glare_mask[y1:y2, x1:x2]
        if roi.size < 16:
            continue
        denoised = cv2.GaussianBlur(roi, (3, 3), 0)
        gradient_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
        rows.append(
            {
                "score": token["score"],
                "weight": int(token["text_length"]),
                # The short side is stable even when the prepared image and
                # PP's internally corrected canvas use different rotations.
                "height": round(min(box_width, box_height), 2),
                "glare_ratio": float(np.count_nonzero(roi_glare))
                / float(max(1, roi_glare.size)),
                "laplacian": float(cv2.Laplacian(denoised, cv2.CV_32F).var()),
                "tenengrad": float((gradient_x**2 + gradient_y**2).mean()),
                "contrast": float(roi.std()),
                "mean_brightness": float(roi.mean()),
                "dark_ratio": float(np.count_nonzero(roi <= 38))
                / float(max(1, roi.size)),
                "bright_ratio": float(np.count_nonzero(roi >= 245))
                / float(max(1, roi.size)),
            }
        )

    scored = [row for row in rows if row["score"] is not None]
    score_weights = sum(int(row["weight"]) for row in scored)
    low_score_rows = [row for row in scored if float(row["score"]) < 0.65]
    low_score_weight = sum(int(row["weight"]) for row in low_score_rows)
    low_blurry_rows = [
        row
        for row in low_score_rows
        if float(row["laplacian"]) < 12.0
        and float(row["tenengrad"]) < 1200.0
    ]
    low_dark_rows = [
        row
        for row in low_score_rows
        if float(row["mean_brightness"]) < 48.0
        or float(row["dark_ratio"]) >= 0.55
    ]
    low_overexposed_rows = [
        row
        for row in low_score_rows
        if float(row["bright_ratio"]) >= 0.55
        and float(row["contrast"]) < 24.0
    ]
    glare_affected_rows = [
        row for row in low_score_rows if float(row["glare_ratio"]) >= 0.20
    ]
    all_glare_affected_rows = [
        row for row in rows if float(row["glare_ratio"]) >= 0.18
    ]

    def ratio(items: list[dict[str, float | int | None]]) -> float:
        return len(items) / max(1, len(low_score_rows))

    scores = [float(row["score"]) for row in scored]
    heights = [float(row["height"]) for row in rows]
    all_weight = sum(int(row["weight"]) for row in rows)
    weighted_glare = sum(
        float(row["glare_ratio"]) * int(row["weight"]) for row in rows
    ) / max(1, all_weight)
    if tokens:
        envelope_left = min(float(token["bbox"][0]) for token in tokens)
        envelope_top = min(float(token["bbox"][1]) for token in tokens)
        envelope_right = max(float(token["bbox"][2]) for token in tokens)
        envelope_bottom = max(float(token["bbox"][3]) for token in tokens)
        envelope_width_ratio = (envelope_right - envelope_left) / max(
            1.0, float(image_width)
        )
        envelope_height_ratio = (envelope_bottom - envelope_top) / max(
            1.0, float(image_height)
        )
    else:
        envelope_width_ratio = 0.0
        envelope_height_ratio = 0.0
    median_height = float(np.median(heights)) if heights else None
    return {
        "token_count": len(rows),
        "scored_token_count": len(scored),
        "score_mean": round(
            sum(float(row["score"]) * int(row["weight"]) for row in scored)
            / max(1, score_weights),
            4,
        ),
        "score_p25": round(float(np.percentile(scores, 25)), 4) if scores else None,
        "minimum_score": round(min(scores), 4) if scores else None,
        "low_score_ratio": round(len(low_score_rows) / max(1, len(scored)), 4),
        "low_score_char_ratio": round(low_score_weight / max(1, score_weights), 4),
        "median_text_height": round(median_height, 2) if median_height else None,
        "median_text_height_ratio": round(
            median_height / max(1.0, float(max(image_width, image_height))),
            4,
        )
        if median_height is not None
        else None,
        "text_envelope_width_ratio": round(envelope_width_ratio, 4),
        "text_envelope_height_ratio": round(envelope_height_ratio, 4),
        "text_envelope_area_ratio": round(
            envelope_width_ratio * envelope_height_ratio, 4
        ),
        "max_glare_overlap": round(
            max((float(row["glare_ratio"]) for row in rows), default=0.0),
            4,
        ),
        "weighted_glare_overlap": round(weighted_glare, 4),
        "glare_affected_token_count": len(all_glare_affected_rows),
        "glare_affected_low_score_count": len(glare_affected_rows),
        "low_score_blurry_ratio": round(ratio(low_blurry_rows), 4),
        "low_score_dark_ratio": round(ratio(low_dark_rows), 4),
        "low_score_overexposed_ratio": round(ratio(low_overexposed_rows), 4),
    }


def _ocr_quality_issues(
    metrics: dict[str, Any],
    *,
    severe_global_blur: bool,
    severe_global_glare: bool,
    moderate_glare_requires_ocr_risk: bool = False,
    strong_glare_requires_ocr_risk: bool = False,
) -> list[ImageQualityIssue]:
    """Attribute strong OCR readability failures to the likeliest cause."""

    token_count = int(metrics["token_count"])
    scored_count = int(metrics["scored_token_count"])
    low_score_ratio = float(metrics["low_score_ratio"])
    low_char_ratio = float(metrics["low_score_char_ratio"])
    score_mean = float(metrics["score_mean"])
    score_p25 = metrics["score_p25"]
    ocr_risk = (
        scored_count >= 2
        and (
            low_char_ratio >= 0.30
            or low_score_ratio >= 0.40
            or (score_p25 is not None and float(score_p25) < 0.55)
        )
    )
    severe_ocr_risk = scored_count >= 2 and (
        score_mean < 0.48 or low_score_ratio >= 0.75
    )

    max_glare_overlap = float(metrics["max_glare_overlap"])
    weighted_glare_overlap = float(metrics["weighted_glare_overlap"])
    glare_affected_count = int(metrics["glare_affected_token_count"])
    # PP recognition confidence can remain near 1.0 even when a plastic glare
    # stripe changes individual characters. Strong geometric overlap must
    # therefore be able to reject independently of rec_scores.
    moderate_text_glare = (
        (glare_affected_count >= 2 and max_glare_overlap >= 0.24)
        or weighted_glare_overlap >= 0.12
    )
    strong_text_glare = (
        max_glare_overlap >= 0.40
        and (ocr_risk or not strong_glare_requires_ocr_risk)
    ) or (
        moderate_text_glare
        and (ocr_risk or not moderate_glare_requires_ocr_risk)
    )
    confidence_linked_glare = ocr_risk and (
        int(metrics["glare_affected_low_score_count"]) >= 1
        and max_glare_overlap >= 0.20
    )
    if strong_text_glare or confidence_linked_glare:
        return [
            ImageQualityIssue(
                "GLARE_OCCLUDES_TEXT",
                "证件反光覆盖了部分文字，请取出透明证件套或调整拍摄角度后重新拍摄",
            )
        ]
    median_height_ratio = metrics["median_text_height_ratio"]
    if (
        token_count >= 6
        and median_height_ratio is not None
        and float(median_height_ratio) < 0.012
        and float(metrics["text_envelope_area_ratio"]) < 0.05
    ):
        return [
            ImageQualityIssue(
                "DOCUMENT_TOO_SMALL",
                "证件在画面中占比过小，文字细节不足，请靠近证件后重新拍摄",
            )
        ]
    median_height = metrics["median_text_height"]
    if ocr_risk and median_height is not None and float(median_height) < 10.0:
        return [
            ImageQualityIssue(
                "TEXT_RESOLUTION_TOO_LOW",
                "证件文字过小、拍摄距离可能过远，请靠近证件重新拍摄",
            )
        ]
    if ocr_risk and float(metrics["low_score_dark_ratio"]) >= 0.60:
        return [
            ImageQualityIssue(
                "TEXT_REGION_TOO_DARK",
                "证件文字区域光线不足，请在更明亮、均匀的环境中重新拍摄",
            )
        ]
    if ocr_risk and float(metrics["low_score_overexposed_ratio"]) >= 0.60:
        return [
            ImageQualityIssue(
                "TEXT_REGION_OVEREXPOSED",
                "证件文字区域过曝，请避免强光直射后重新拍摄",
            )
        ]
    if (
        severe_global_blur
        and (token_count == 0 or scored_count == 0 or ocr_risk)
    ) or (
        ocr_risk and float(metrics["low_score_blurry_ratio"]) >= 0.60
    ):
        return [
            ImageQualityIssue(
                "TEXT_REGION_BLURRY",
                "证件文字区域明显模糊，请保持手机稳定并等待对焦后重新拍摄",
            )
        ]
    if token_count == 0 and severe_global_glare:
        return [
            ImageQualityIssue(
                "GLARE_OCCLUDES_TEXT",
                "证件存在大面积反光且未检测到有效文字，请调整拍摄角度后重新拍摄",
            )
        ]
    if token_count == 0 or severe_ocr_risk:
        return [
            ImageQualityIssue(
                "QUALITY_NOT_ACCEPTABLE",
                "图片质量不通过，请重新拍摄后再试",
            )
        ]
    return []


def analyze_image_quality(
    content: bytes,
    document_type: str,
    ppocr_result: dict[str, Any] | None = None,
) -> ImageQualityReport:
    """Apply initial CV rules; OCR-aware rules can be added at this boundary."""

    try:
        image = _decode_image(content)
    except ValueError as exc:
        return ImageQualityReport(
            issues=(ImageQualityIssue("IMAGE_DECODE_FAILED", str(exc)),),
            metrics={},
        )

    original_height, original_width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(original_width, original_height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blur_metrics = _blur_metrics(image)

    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
    local_delta = gray.astype(np.int16) - local_background.astype(np.int16)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    top_hat_size = max(21, round(min(gray.shape) * 0.035))
    if top_hat_size % 2 == 0:
        top_hat_size += 1
    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (top_hat_size, top_hat_size),
        ),
    )
    pure_specular = (value >= 235) & (saturation <= 90)
    relative_specular = (
        (gray >= 220)
        & ((local_delta >= 30) | (top_hat >= 26))
        & (saturation <= 130)
    )
    glare_mask = (pure_specular | relative_specular).astype(np.uint8)
    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )
    # A separate, stricter mask powers the per-field hard rule. Requiring a
    # near-achromatic local highlight prevents the pale green licence stock
    # itself from being counted as glare.
    field_specular_raw_mask = (
        (gray >= 220)
        & ((local_delta >= 30) | (top_hat >= 26))
        & (saturation <= 20)
    ).astype(np.uint8)
    field_specular_mask = cv2.morphologyEx(
        field_specular_raw_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )
    # Key-field decisions use stricter, physically motivated masks. The first
    # captures near-clipped white highlights. The second captures broad glare
    # veils that wash out the printed micro-texture without reaching pure white.
    key_field_specular_mask = (
        (gray >= 245)
        & (local_delta >= 20)
        & (saturation <= 30)
    ).astype(np.uint8)
    large_area_specular_mask = cv2.morphologyEx(
        key_field_specular_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    elongated_glare_mask = (
        (gray >= 190)
        # Thin reflection streaks must stand out at their own spatial scale.
        # ``local_delta`` alone also selects broad pale licence backgrounds and
        # joins nearby streaks into thick components, which then hides the very
        # pattern this detector is intended to find.
        & (top_hat >= 15)
        & (saturation <= 120)
    ).astype(np.uint8)
    elongated_glare_mask = cv2.morphologyEx(
        elongated_glare_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    gray_float = gray.astype(np.float32)
    texture_mean = cv2.blur(gray_float, (9, 9))
    texture_mean_square = cv2.blur(gray_float * gray_float, (9, 9))
    texture_std = np.sqrt(
        np.maximum(
            texture_mean_square - texture_mean * texture_mean,
            0.0,
        )
    )
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    key_field_texture_loss_mask = (
        (gray >= 120)
        & (saturation <= 60)
        & (texture_std < 15)
        & (gradient_magnitude < 30)
    ).astype(np.uint8)
    image_pixels = int(gray.size)
    glare_ratio = float(np.count_nonzero(glare_mask)) / float(max(1, image_pixels))
    (
        largest_glare_ratio,
        elongated_glare_ratio,
        elongated_glare_count,
    ) = _glare_component_metrics(glare_mask, image_pixels)

    severe_global_blur = _is_too_blurry(blur_metrics)
    severe_global_glare = _is_severe_glare(
        glare_ratio,
        largest_glare_ratio,
        elongated_glare_ratio,
    )
    ocr_metrics: dict[str, Any] | None = None
    driver_all_tokens: list[dict[str, Any]] | None = None
    id_card_scope: dict[str, Any] | None = None
    driver_license_scope: dict[str, Any] | None = None
    driver_field_glare: dict[str, Any] | None = None
    driver_semantic_glare: dict[str, Any] | None = None
    driver_electronic: dict[str, Any] | None = None
    driver_quality_disposition: dict[str, Any] | None = None
    driver_document_scale: dict[str, Any] | None = None
    driver_elongated_text_glare: dict[str, Any] | None = None
    driver_physical_glare_fallback: dict[str, Any] | None = None
    driver_large_area_glare_fallback: dict[str, Any] | None = None
    driver_weak_physical: dict[str, Any] | None = None
    vehicle_license_scope: dict[str, Any] | None = None
    vehicle_field_glare: dict[str, Any] | None = None
    vehicle_semantic_glare: dict[str, Any] | None = None
    vehicle_electronic: dict[str, Any] | None = None
    if isinstance(ppocr_result, dict):
        tokens = _normalized_ocr_tokens(
            ppocr_result,
            original_width,
            original_height,
        )
        if document_type == "id_card":
            tokens, id_card_scope = _id_card_text_scope(ppocr_result, tokens)
        elif document_type == "driver_license":
            driver_all_tokens = list(tokens)
            driver_electronic = _electronic_driver_license_keywords(
                ppocr_result,
                tokens,
            )
            tokens, driver_license_scope = _driver_license_text_scope(
                ppocr_result,
                tokens,
            )
            driver_weak_physical = _driver_weak_physical_evidence(tokens)
            driver_elongated_text_glare = (
                _driver_elongated_text_glare_metrics(
                    elongated_glare_mask,
                    tokens,
                    scale,
                )
            )
            driver_field_glare = _mask_overlap_metrics(
                field_specular_mask,
                _driver_license_field_tokens(tokens),
                scale,
            )
            driver_semantic_glare = _driver_license_semantic_glare_metrics(
                key_field_specular_mask,
                tokens,
                scale,
                key_field_texture_loss_mask,
                gray,
            )
        elif document_type == "vehicle_license":
            vehicle_electronic = _electronic_vehicle_license_keywords(
                ppocr_result,
                tokens,
            )
            tokens, vehicle_license_scope = _vehicle_license_text_scope(
                ppocr_result,
                tokens,
            )
            vehicle_field_glare = _mask_overlap_metrics(
                field_specular_mask,
                _vehicle_license_field_tokens(tokens),
                scale,
            )
            vehicle_semantic_glare = _vehicle_license_semantic_glare_metrics(
                key_field_specular_mask,
                tokens,
                scale,
                key_field_texture_loss_mask,
                gray,
            )
        ocr_metrics = _text_region_metrics(
            image,
            glare_mask,
            tokens,
            scale,
            original_width,
            original_height,
        )
        if document_type == "driver_license":
            driver_large_area_glare_fallback = (
                _driver_large_area_glare_fallback_metrics(
                    large_area_specular_mask,
                    driver_all_tokens or tokens,
                    scale,
                    glare_ratio=glare_ratio,
                    largest_glare_ratio=largest_glare_ratio,
                    ocr_metrics=ocr_metrics,
                    driver_scope=driver_license_scope,
                    weak_physical_evidence=driver_weak_physical,
                    elongated_glare_count=elongated_glare_count,
                    elongated_glare_ratio=elongated_glare_ratio,
                    semantic_glare_metrics=driver_semantic_glare,
                )
            )
        if driver_field_glare is not None:
            driver_field_glare = _refine_field_glare_metrics(
                driver_field_glare,
                ocr_metrics,
            )
            driver_physical_glare_fallback = (
                _driver_physical_glare_fallback_metrics(
                    glare_ratio=glare_ratio,
                    elongated_glare_count=elongated_glare_count,
                    elongated_glare_ratio=elongated_glare_ratio,
                    ocr_metrics=ocr_metrics,
                    field_glare=driver_field_glare,
                )
            )
        if vehicle_field_glare is not None:
            vehicle_field_glare = _refine_field_glare_metrics(
                vehicle_field_glare,
                ocr_metrics,
            )
        issues = _ocr_quality_issues(
            ocr_metrics,
            severe_global_blur=severe_global_blur,
            severe_global_glare=severe_global_glare,
            moderate_glare_requires_ocr_risk=document_type
            in {"id_card", "driver_license", "vehicle_license"},
            strong_glare_requires_ocr_risk=document_type
            in {"driver_license", "vehicle_license"},
        )
        if document_type == "driver_license":
            median_text_height_ratio = ocr_metrics.get(
                "median_text_height_ratio"
            )
            driver_document_too_small = (
                int(ocr_metrics.get("token_count") or 0) >= 6
                and median_text_height_ratio is not None
                and float(median_text_height_ratio)
                < _DRIVER_MIN_STANDALONE_TEXT_HEIGHT_RATIO
            )
            driver_document_scale = {
                "document_too_small": driver_document_too_small,
                "decision_basis": "median_text_height_ratio",
                "median_text_height_ratio": median_text_height_ratio,
                "minimum_text_height_ratio": (
                    _DRIVER_MIN_STANDALONE_TEXT_HEIGHT_RATIO
                ),
                "token_count": int(ocr_metrics.get("token_count") or 0),
            }
            if driver_document_too_small:
                # A combined main/secondary-page OCR envelope can look large
                # despite both cards occupying only a shallow strip between
                # substantial margins. Very small text is therefore an
                # independent distance signal for physical driving licences.
                issues = [
                    ImageQualityIssue(
                        "DOCUMENT_TOO_SMALL",
                        "驾驶证在画面中占比过小，文字细节不足，请靠近证件后重新拍摄",
                    )
                ]
        if document_type in {"driver_license", "vehicle_license"}:
            # Both licence types have document-specific field localisation and
            # glare masks. Generic pale-card brightness combined with weak OCR
            # remains diagnostic and must not override the semantic decision.
            issues = [
                issue
                for issue in issues
                if issue.code != "GLARE_OCCLUDES_TEXT"
            ]
        if (
            driver_semantic_glare is not None
            and int(driver_semantic_glare["triggered_count"]) > 0
        ):
            fields = "、".join(
                str(row["field_name"])
                for row in driver_semantic_glare["triggered_fields"][:3]
            )
            issues = [
                ImageQualityIssue(
                    "GLARE_OCCLUDES_TEXT",
                    f"驾驶证{fields}字段区域反光占比过高，请调整拍摄角度后重新拍摄",
                )
            ]
        elif (
            driver_large_area_glare_fallback is not None
            and bool(driver_large_area_glare_fallback.get("detected"))
        ):
            issues = [
                ImageQualityIssue(
                    "GLARE_OCCLUDES_TEXT",
                    "驾驶证卡面存在大面积明显反光，请调整拍摄角度后重新拍摄",
                )
            ]
        elif (
            driver_physical_glare_fallback is not None
            and bool(driver_physical_glare_fallback.get("detected"))
        ):
            fallback_basis = str(
                driver_physical_glare_fallback.get("decision_basis") or ""
            )
            message = (
                "驾驶证存在多条反光线遮挡文字，请取出透明证件套或调整拍摄角度后重新拍摄"
                if fallback_basis == "multiple_streaks_crossing_text"
                else "驾驶证存在强反光遮挡多处文字，请调整拍摄角度后重新拍摄"
            )
            issues = [ImageQualityIssue("GLARE_OCCLUDES_TEXT", message)]
        elif (
            driver_elongated_text_glare is not None
            and bool(driver_elongated_text_glare.get("detected"))
            and _DRIVER_FALLBACK_GLARE_MIN_RATIO <= glare_ratio <= 0.08
        ):
            issues = [
                ImageQualityIssue(
                    "GLARE_OCCLUDES_TEXT",
                    "驾驶证存在多条细长反光线遮挡文字，请取出透明证件套或调整拍摄角度后重新拍摄",
                )
            ]
        if (
            vehicle_semantic_glare is not None
            and int(vehicle_semantic_glare["triggered_count"]) > 0
        ):
            fields = "、".join(
                str(row["field_name"])
                for row in vehicle_semantic_glare["triggered_fields"][:3]
            )
            issues = [
                ImageQualityIssue(
                    "GLARE_OCCLUDES_TEXT",
                    f"行驶证{fields}字段区域反光占比过高，请调整拍摄角度后重新拍摄",
                )
            ]
        if (
            document_type == "driver_license"
            and driver_license_scope is not None
            and driver_semantic_glare is not None
        ):
            label_backed_value_fields = [
                str(row["field"])
                for row in driver_semantic_glare.get("fields") or []
                if int(row.get("label_count") or 0) > 0
                and bool(row.get("texts"))
            ]
            driver_specific_label_fields = sorted(
                set(label_backed_value_fields)
                & {
                    "archive_number",
                    "first_issue_date",
                    "valid_period",
                    "vehicle_class",
                }
            )
            recognizable_physical_license = (
                str(driver_license_scope.get("side") or "") != "unknown"
                or len(driver_specific_label_fields) >= 2
                or bool(
                    driver_weak_physical is not None
                    and driver_weak_physical.get("detected")
                )
                or bool(
                    driver_large_area_glare_fallback is not None
                    and driver_large_area_glare_fallback.get("detected")
                )
            )
            electronic_license = bool(
                driver_electronic is not None
                and driver_electronic.get("detected")
            )
            exclusion_reason = (
                "electronic_driver_license"
                if electronic_license
                else (
                    None
                    if recognizable_physical_license
                    else "driver_license_not_recognized"
                )
            )
            driver_quality_disposition = {
                "quality_applicable": exclusion_reason is None,
                "exclude_from_quality_rejected": exclusion_reason is not None,
                "reason": exclusion_reason,
                "recognizable_physical_license": (
                    recognizable_physical_license
                ),
                "label_backed_value_fields": label_backed_value_fields,
                "driver_specific_label_fields": (
                    driver_specific_label_fields
                ),
                "weak_physical_evidence": driver_weak_physical,
                "large_area_glare_evidence": (
                    driver_large_area_glare_fallback
                ),
            }
            if exclusion_reason is not None:
                # Screen glare is expected on electronic licences. Conversely,
                # when OCR cannot establish that this is a physical driving
                # licence, image defects cannot be attributed to that document
                # type and must not be exported as a quality rejection.
                issues = []
        if (
            document_type == "vehicle_license"
            and vehicle_license_scope is not None
            and str(vehicle_license_scope.get("side") or "") == "unknown"
            and vehicle_semantic_glare is not None
            and int(vehicle_semantic_glare.get("field_count") or 0) == 0
        ):
            # Without a recognizable main/secondary page or any locatable key
            # field, generic bright regions cannot be attributed to document
            # text. Report an invalid/unrecognizable document instead of
            # presenting the low-confidence OCR failure as field glare.
            issues = [
                ImageQualityIssue(
                    "VEHICLE_LICENSE_NOT_RECOGNIZED",
                    "未识别到行驶证有效信息，图片可能不是有效行驶证或证件内容不可辨认，请确认上传正确证件并重新拍摄",
                )
            ]
        if (
            id_card_scope is not None
            and bool(id_card_scope.get("anchors_missing"))
            and issues
            and issues[0].code == "QUALITY_NOT_ACCEPTABLE"
        ):
            side = str(id_card_scope.get("side") or "")
            message = (
                "未识别到姓名和公民身份号码等关键文字，请保持身份证人像面完整清晰后重新拍摄"
                if side == "front"
                else "未识别到证件标题和有效期限等关键文字，请保持身份证国徽面完整清晰后重新拍摄"
            )
            issues = [ImageQualityIssue("ID_CARD_KEY_FIELDS_NOT_DETECTED", message)]
        if (
            document_type == "driver_license"
            and driver_electronic is not None
            and bool(driver_electronic["detected"])
        ):
            # Electronic driving licences are screen-based documents. Reflections,
            # screen glare and UI text are expected, so the physical-card quality
            # rules must not reject them once OCR provides positive evidence.
            issues = []
        if (
            document_type == "vehicle_license"
            and vehicle_electronic is not None
            and bool(vehicle_electronic["detected"])
        ):
            # Electronic vehicle licences are screen documents rather than
            # photographed physical cards, so bright UI regions are expected.
            issues = []
    else:
        # Preserve the standalone/legacy behavior when no PP result is passed.
        issues = []
        if severe_global_blur:
            issues.append(
                ImageQualityIssue(
                    "TOO_BLURRY",
                    "证件文字明显模糊，请保持手机稳定并等待相机对焦后重新拍摄",
                )
            )
        if severe_global_glare:
            issues.append(
                ImageQualityIssue(
                    "SEVERE_GLARE",
                    "证件表面存在明显反光并遮挡文字，请取出透明证件套或调整拍摄角度",
                )
            )
    metrics: dict[str, Any] = {
        "width": original_width,
        "height": original_height,
        "blur": blur_metrics,
        "glare_ratio": round(glare_ratio, 4),
        "largest_glare_ratio": round(largest_glare_ratio, 4),
        "elongated_glare_ratio": round(elongated_glare_ratio, 4),
        "elongated_glare_count": elongated_glare_count,
    }
    if ocr_metrics is not None:
        metrics["ocr"] = ocr_metrics
    if id_card_scope is not None:
        metrics["id_card_scope"] = id_card_scope
    if driver_license_scope is not None:
        metrics["driver_license_scope"] = driver_license_scope
    if document_type == "driver_license":
        metrics["driver_license_electronic"] = driver_electronic or {
            "detected": False,
            "matched_keywords": [],
            "candidate_keywords": [],
            "physical_page_side": None,
            "strategy": "no_ppocr_evidence",
        }
        metrics["driver_license_quality_disposition"] = (
            driver_quality_disposition
            or {
                "quality_applicable": True,
                "exclude_from_quality_rejected": False,
                "reason": None,
                "recognizable_physical_license": False,
                "label_backed_value_fields": [],
                "driver_specific_label_fields": [],
                "weak_physical_evidence": None,
            }
        )
        metrics["driver_license_document_scale"] = (
            driver_document_scale
            or {
                "document_too_small": False,
                "decision_basis": "no_ppocr_evidence",
                "median_text_height_ratio": None,
                "minimum_text_height_ratio": (
                    _DRIVER_MIN_STANDALONE_TEXT_HEIGHT_RATIO
                ),
                "token_count": 0,
            }
        )
        metrics["driver_license_elongated_text_glare"] = (
            driver_elongated_text_glare
            or {
                "detected": False,
                "component_count": 0,
                "total_area_ratio": 0.0,
                "minimum_component_count": (
                    _DRIVER_ELONGATED_GLARE_MIN_COMPONENT_COUNT
                ),
                "minimum_total_area_ratio": (
                    _DRIVER_ELONGATED_GLARE_MIN_TOTAL_AREA_RATIO
                ),
                "components": [],
            }
        )
        metrics["driver_license_physical_glare_fallback"] = (
            driver_physical_glare_fallback
            or {
                "detected": False,
                "decision_basis": None,
                "strong_text_glare": False,
                "multiple_streaks": False,
            }
        )
        metrics["driver_license_large_area_glare_fallback"] = (
            driver_large_area_glare_fallback
            or {
                "detected": False,
                "decision_basis": None,
                "recognized_document_glare": False,
                "degraded_document_glare": False,
            }
        )
        metrics["driver_license_weak_physical_evidence"] = (
            driver_weak_physical
            or {
                "detected": False,
                "vehicle_classes": [],
                "nationality_markers": [],
                "date_like_values": [],
                "minimum_date_like_count": 2,
                "strategy": "no_ppocr_evidence",
            }
        )
    if driver_field_glare is not None:
        metrics["driver_license_field_glare"] = driver_field_glare
    if driver_semantic_glare is not None:
        metrics["driver_license_semantic_glare"] = driver_semantic_glare
    if vehicle_license_scope is not None:
        metrics["vehicle_license_scope"] = vehicle_license_scope
    if document_type == "vehicle_license":
        metrics["vehicle_license_electronic"] = vehicle_electronic or {
            "detected": False,
            "explicit_title": False,
            "matched_keywords": [],
            "strategy": "no_ppocr_evidence",
        }
    if vehicle_field_glare is not None:
        metrics["vehicle_license_field_glare"] = vehicle_field_glare
    if vehicle_semantic_glare is not None:
        metrics["vehicle_license_semantic_glare"] = vehicle_semantic_glare
    return ImageQualityReport(tuple(issues), metrics)
