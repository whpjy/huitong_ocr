"""Fast, conservative visual correction for driver archive-number 0/8 errors.

The correction is deliberately independent of OCR/model rechecks.  It locates
the archive-number row on a normalized physical licence page and compares the
closed-hole stability of glyphs that HunyuanOCR returned as ``8``.  A glyph is
changed to ``0`` only when the same row contains a stable, closed reference 8
and the candidate remains completely open under several adaptive thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np


_ARCHIVE_LENGTH = 12
_PAGE_WIDTH = 1600
_PAGE_HEIGHT = 1000
_REFERENCE_HOLE_SCORE = 0.025
_OPEN_GLYPH_SCORE = 0.003


@dataclass(frozen=True)
class ArchiveVisualDecision:
    original: str
    corrected: str
    changed_indexes: tuple[int, ...] = ()
    hole_scores: tuple[float, ...] = ()
    page: np.ndarray | None = None
    digit_patches: tuple[np.ndarray, ...] = ()
    elapsed_seconds: float = 0.0
    reason: str = ""


def _read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None
    return image if image is not None and image.size else None


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def _warp_landscape(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    ordered = _order_points(points)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    width = max(top, bottom)
    height = max(left, right)
    if height > width:
        ordered = np.roll(ordered, -1, axis=0)
        width, height = height, width
    aspect = float(np.clip(width / max(height, 1.0), 1.2, 3.6))
    target_height = _PAGE_HEIGHT
    target_width = max(_PAGE_WIDTH, round(target_height * aspect))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))


def _green_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (28, 20, 45), (100, 255, 255))


def _largest_green_rectangle(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    detector = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    mask = _green_mask(detector)
    kernel_width = max(9, round(mask.shape[1] * 0.015)) | 1
    kernel_height = max(7, round(mask.shape[0] * 0.012)) | 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, kernel_height),
        ),
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < mask.shape[0] * mask.shape[1] * 0.025:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32) / scale


def _split_union_pages(union: np.ndarray) -> list[np.ndarray]:
    height, width = union.shape[:2]
    if width / max(height, 1) < 2.15:
        return [cv2.resize(union, (_PAGE_WIDTH, _PAGE_HEIGHT))]

    mask = _green_mask(union)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    individual_contours = [
        contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(contour) >= height * width * 0.08
    ][:4]
    if len(individual_contours) >= 2:
        return [
            cv2.resize(
                _warp_landscape(
                    union,
                    cv2.boxPoints(cv2.minAreaRect(contour)),
                ),
                (_PAGE_WIDTH, _PAGE_HEIGHT),
            )
            for contour in individual_contours
        ]

    density = (mask > 0).mean(axis=0)
    window = max(11, round(width * 0.015)) | 1
    smoothed = np.convolve(density, np.ones(window) / window, mode="same")
    left = round(width * 0.35)
    right = round(width * 0.65)
    split = left + int(np.argmin(smoothed[left:right]))
    halves = [
        union[:, :split],
        union[:, split:],
    ]
    pages: list[np.ndarray] = []
    for half in halves:
        rectangle = _largest_green_rectangle(half)
        if rectangle is None:
            continue
        page = _warp_landscape(half, rectangle)
        pages.append(cv2.resize(page, (_PAGE_WIDTH, _PAGE_HEIGHT)))
    return pages


def _physical_page_candidates(image: np.ndarray) -> list[np.ndarray]:
    rectangle = _largest_green_rectangle(image)
    if rectangle is None:
        return []
    union = _warp_landscape(image, rectangle)
    return _split_union_pages(union)


def _projection_groups(values: np.ndarray, minimum_width: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_width:
                groups.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= minimum_width:
        groups.append((start, len(values) - 1))
    return groups


def _regular_group_window(
    groups: list[tuple[int, int]],
    count: int,
) -> tuple[list[tuple[int, int]], float] | None:
    if len(groups) < count:
        return None
    best: tuple[list[tuple[int, int]], float] | None = None
    for start in range(len(groups) - count + 1):
        window = groups[start : start + count]
        centers = np.array([(left + right) / 2 for left, right in window])
        pitches = np.diff(centers)
        median_pitch = float(np.median(pitches))
        if median_pitch <= 0:
            continue
        pitch_cv = float(np.std(pitches) / median_pitch)
        span_ratio = float((centers[-1] - centers[0]) / median_pitch)
        score = pitch_cv + abs(span_ratio - (count - 1)) * 0.02
        if pitch_cv > 0.32:
            continue
        if best is None or score < best[1]:
            best = (window, score)
    return best


def _archive_digit_patches(
    page: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], float] | None:
    height, width = page.shape[:2]
    x1, x2 = round(width * 0.58), round(width * 0.90)
    y1, y2 = round(height * 0.315), round(height * 0.395)
    roi = page[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    green = roi[:, :, 1]
    background = cv2.GaussianBlur(green, (0, 0), 7)
    darkness = np.clip(
        background.astype(np.int16) - green.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)
    band_top = round(darkness.shape[0] * 0.16)
    band_bottom = round(darkness.shape[0] * 0.90)
    band = darkness[band_top:band_bottom]
    threshold = max(20.0, float(np.percentile(band, 85)))
    column_counts = (band > threshold).sum(axis=0)
    smoothed = np.convolve(column_counts, np.ones(3) / 3, mode="same")
    groups = _projection_groups(
        smoothed > max(4.0, band.shape[0] * 0.085),
        minimum_width=max(4, round(width * 0.003)),
    )
    selected = _regular_group_window(groups, _ARCHIVE_LENGTH)
    if selected is None:
        return None
    digit_groups, regularity = selected
    centers = np.array([(left + right) / 2 for left, right in digit_groups])
    pitch = float(np.median(np.diff(centers)))
    if not width * 0.010 <= pitch <= width * 0.030:
        return None

    patch_top = max(0, round(height * 0.325) - y1)
    patch_bottom = min(roi.shape[0], round(height * 0.395) - y1)
    patches: list[np.ndarray] = []
    for center in centers:
        left = max(0, round(center - pitch * 0.56))
        right = min(roi.shape[1], round(center + pitch * 0.56))
        patch = roi[patch_top:patch_bottom, left:right]
        if patch.size == 0:
            return None
        patches.append(patch.copy())
    return tuple(patches), regularity


def _closed_hole_score(patch: np.ndarray) -> float:
    normalized = cv2.resize(patch, (64, 128), interpolation=cv2.INTER_CUBIC)
    green = normalized[:, :, 1]
    scores: list[float] = []
    for block_size, constant in ((31, 5), (41, 7), (51, 9), (61, 11)):
        mask = cv2.adaptiveThreshold(
            green,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            constant,
        )
        mask[:3, :] = 0
        mask[-3:, :] = 0
        mask[:, :3] = 0
        mask[:, -3:] = 0
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        largest_hole = 0.0
        if hierarchy is not None:
            for index, contour in enumerate(contours):
                if hierarchy[0][index][3] < 0:
                    continue
                area = float(cv2.contourArea(contour))
                if area >= 15.0:
                    largest_hole = max(largest_hole, area)
        scores.append(largest_hole / float(mask.size))
    return max(scores, default=0.0)


def _correct_ambiguous_eights(
    value: str,
    scores: Sequence[float],
) -> tuple[str, tuple[int, ...]]:
    if len(value) != _ARCHIVE_LENGTH or len(scores) != _ARCHIVE_LENGTH:
        return value, ()
    eight_indexes = [index for index, char in enumerate(value) if char == "8"]
    if len(eight_indexes) < 2:
        return value, ()
    reference_score = max(scores[index] for index in eight_indexes)
    if reference_score < _REFERENCE_HOLE_SCORE:
        return value, ()
    changed = tuple(
        index
        for index in eight_indexes
        if scores[index] <= _OPEN_GLYPH_SCORE
        and reference_score - scores[index] >= 0.02
    )
    if not changed or len(changed) > 2:
        return value, ()
    corrected = list(value)
    for index in changed:
        corrected[index] = "0"
    return "".join(corrected), changed


def inspect_archive_number(
    image_paths: Sequence[Path],
    archive_number: str,
) -> ArchiveVisualDecision:
    """Inspect an extracted archive number without invoking another model."""

    started = perf_counter()
    value = "".join(char for char in str(archive_number or "") if char.isdigit())
    if len(value) != _ARCHIVE_LENGTH or "8" not in value:
        return ArchiveVisualDecision(
            original=value,
            corrected=value,
            elapsed_seconds=perf_counter() - started,
            reason="not_applicable",
        )

    best: tuple[float, np.ndarray, tuple[np.ndarray, ...], tuple[float, ...]] | None = None
    for path in image_paths:
        image = _read_image(Path(path))
        if image is None:
            continue
        for page in _physical_page_candidates(image):
            located = _archive_digit_patches(page)
            if located is None:
                continue
            patches, regularity = located
            scores = tuple(_closed_hole_score(patch) for patch in patches)
            candidate = (regularity, page, patches, scores)
            if best is None or regularity < best[0]:
                best = candidate

    if best is None:
        return ArchiveVisualDecision(
            original=value,
            corrected=value,
            elapsed_seconds=perf_counter() - started,
            reason="archive_row_not_found",
        )
    _, page, patches, scores = best
    corrected, changed = _correct_ambiguous_eights(value, scores)
    return ArchiveVisualDecision(
        original=value,
        corrected=corrected,
        changed_indexes=changed,
        hole_scores=scores,
        page=page,
        digit_patches=patches,
        elapsed_seconds=perf_counter() - started,
        reason="corrected" if changed else "no_high_confidence_change",
    )


def refine_driver_archive_visually(
    image_paths: Sequence[Path],
    fields: dict[str, Any],
) -> dict[str, str]:
    """Return fields with a high-confidence archive-number correction."""

    refined = {str(key): str(value or "") for key, value in fields.items()}
    try:
        decision = inspect_archive_number(
            image_paths,
            refined.get("档案编号", ""),
        )
    except (ValueError, TypeError, cv2.error):
        return refined
    if decision.changed_indexes:
        refined["档案编号"] = decision.corrected
    return refined


__all__ = [
    "ArchiveVisualDecision",
    "inspect_archive_number",
    "refine_driver_archive_visually",
]
