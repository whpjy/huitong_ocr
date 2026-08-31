"""PP-OCRv6 provider migrated from the existing batch script."""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

from ..models import ProviderResult
from .base import OCRProvider


def _strip_data_uri(value: str) -> str:
    value = value.strip()
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _sorted_layout_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(block: dict[str, Any]) -> tuple[int, int]:
        order = block.get("block_order")
        block_id = block.get("block_id")
        return (
            int(order) if order is not None else 10**9,
            int(block_id) if block_id is not None else 10**9,
        )

    return sorted(blocks, key=key)


def prepare_request_image(
    source_path: Path,
    max_side: int | None,
) -> tuple[bytes, tuple[float, float], tuple[int, int]]:
    """Return request bytes, inverse coordinate scales, and request size."""

    raw = source_path.read_bytes()
    if max_side is None:
        with Image.open(io.BytesIO(raw)) as source:
            return raw, (1.0, 1.0), source.size

    with Image.open(io.BytesIO(raw)) as source:
        orientation = source.getexif().get(274, 1)
        if (
            source.format == "JPEG"
            and source.mode == "RGB"
            and orientation in (None, 1)
            and max(source.size) <= max_side
        ):
            # The shared mobile preprocessing has already produced exactly the
            # model-sized JPEG. Avoid competing with Hunyuan for a duplicate
            # decode/encode on the request thread.
            return raw, (1.0, 1.0), source.size
        image = ImageOps.exif_transpose(source).convert("RGB")
        source_width, source_height = image.size
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        request_width, request_height = image.size
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
    return (
        output.getvalue(),
        (
            source_width / request_width,
            source_height / request_height,
        ),
        (request_width, request_height),
    )


def _extract_preprocessor_angle(data: dict[str, Any]) -> int:
    """Return the rotation PP applied before producing OCR coordinates."""

    result = data.get("result") or {}
    for ocr_result in result.get("ocrResults") or []:
        pruned = ocr_result.get("prunedResult") or {}
        preprocessor = pruned.get("doc_preprocessor_res") or {}
        raw_angle = preprocessor.get("angle")
        if isinstance(raw_angle, (int, float)):
            angle = int(raw_angle) % 360
            if angle in {90, 180, 270}:
                return angle
    return 0


def _inverse_rotate_bbox(
    bbox: list[float],
    request_size: tuple[int, int],
    angle: int,
) -> list[float]:
    """Map a box on PP's counterclockwise-rotated canvas to the request."""

    width, height = request_size
    left, top, right, bottom = bbox
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    if angle == 90:
        restored = [(width - y, x) for x, y in corners]
    elif angle == 180:
        restored = [(width - x, height - y) for x, y in corners]
    elif angle == 270:
        restored = [(y, height - x) for x, y in corners]
    else:
        return bbox
    xs = [point[0] for point in restored]
    ys = [point[1] for point in restored]
    return [min(xs), min(ys), max(xs), max(ys)]


def _restore_token_coordinates(
    result: ProviderResult,
    coordinate_scale: tuple[float, float],
    request_size: tuple[int, int],
    preprocessor_angle: int = 0,
) -> ProviderResult:
    """Undo PP orientation and map request boxes back to source coordinates."""

    scale_x, scale_y = coordinate_scale
    if (
        scale_x == 1.0
        and scale_y == 1.0
        and preprocessor_angle == 0
    ):
        return result
    tokens: list[dict[str, Any]] = []
    for token in result.tokens:
        normalized = dict(token)
        bbox = token.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            restored_bbox = _inverse_rotate_bbox(
                [float(value) for value in bbox],
                request_size,
                preprocessor_angle,
            )
            normalized["bbox"] = [
                restored_bbox[0] * scale_x,
                restored_bbox[1] * scale_y,
                restored_bbox[2] * scale_x,
                restored_bbox[3] * scale_y,
            ]
        tokens.append(normalized)
    return ProviderResult(
        text=result.text,
        page_texts=result.page_texts,
        visualizations=result.visualizations,
        tokens=tuple(tokens),
    )


class PPOCRv6Provider(OCRProvider):
    """Call a PP-OCRv6 compatible layout-parsing service."""

    original_prefix = "原图"

    def check_health(self, timeout: float | None = None) -> None:
        response = self.get_session().get(
            self.config.health_url,
            timeout=timeout or min(self.config.timeout, 30),
        )
        response.raise_for_status()
        data = response.json()
        if str(data.get("status", "")).lower() != "ready":
            raise RuntimeError(
                f"OCR 服务未就绪：{json.dumps(data, ensure_ascii=False)}"
            )

    @staticmethod
    def _extract_texts(result: dict[str, Any]) -> list[str]:
        texts: list[str] = []

        for ocr_result in result.get("ocrResults") or []:
            pruned_result = ocr_result.get("prunedResult") or {}
            for text in pruned_result.get("rec_texts") or []:
                normalized = str(text).strip()
                if normalized:
                    texts.append(normalized)

        if texts:
            return texts

        for layout_result in result.get("layoutParsingResults") or []:
            pruned_result = layout_result.get("prunedResult") or {}
            blocks = pruned_result.get("parsing_res_list") or []
            for block in _sorted_layout_blocks(blocks):
                content = str(block.get("block_content") or "").strip()
                if content:
                    texts.append(content)

        return texts

    @staticmethod
    def _extract_tokens(result: dict[str, Any]) -> list[dict[str, Any]]:
        tokens: list[dict[str, Any]] = []
        for ocr_result in result.get("ocrResults") or []:
            pruned = ocr_result.get("prunedResult") or {}
            texts = pruned.get("rec_texts") or []
            boxes = pruned.get("rec_boxes") or pruned.get("rec_polys") or []
            scores = pruned.get("rec_scores") or []
            for index, value in enumerate(texts):
                text = str(value).strip()
                if not text or index >= len(boxes):
                    continue
                box = boxes[index]
                points: list[tuple[float, float]] = []
                if (
                    isinstance(box, list)
                    and len(box) == 4
                    and all(isinstance(item, (int, float)) for item in box)
                ):
                    left, top, right, bottom = map(float, box)
                elif isinstance(box, list):
                    for point in box:
                        if (
                            isinstance(point, list)
                            and len(point) >= 2
                            and all(
                                isinstance(item, (int, float))
                                for item in point[:2]
                            )
                        ):
                            points.append((float(point[0]), float(point[1])))
                    if not points:
                        continue
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue
                score = (
                    float(scores[index])
                    if index < len(scores)
                    and isinstance(scores[index], (int, float))
                    else None
                )
                tokens.append(
                    {
                        "text": text,
                        "score": score,
                        "bbox": [left, top, right, bottom],
                    }
                )
        return tokens

    @staticmethod
    def _extract_visualization(result: dict[str, Any]) -> bytes | None:
        for layout_result in result.get("layoutParsingResults") or []:
            output_images = layout_result.get("outputImages") or {}
            encoded = output_images.get("layout_det_res")
            if isinstance(encoded, str) and encoded.strip():
                return base64.b64decode(_strip_data_uri(encoded))

        for ocr_result in result.get("ocrResults") or []:
            encoded = ocr_result.get("ocrImage")
            if isinstance(encoded, str) and encoded.strip():
                return base64.b64decode(_strip_data_uri(encoded))

        return None

    @classmethod
    def parse_response(cls, data: dict[str, Any]) -> ProviderResult:
        if data.get("errorCode") not in (None, 0):
            raise RuntimeError(
                f"OCR errorCode={data.get('errorCode')}: "
                f"{data.get('errorMsg', '')}"
            )
        result = data.get("result") or {}
        text = "\n".join(cls._extract_texts(result)).strip()
        visualization = cls._extract_visualization(result)
        return ProviderResult(
            text=text,
            page_texts=(text,),
            visualizations=(visualization,) if visualization is not None else (),
            tokens=tuple(cls._extract_tokens(result)),
        )

    def recognize(self, source_path: Path) -> ProviderResult:
        if not source_path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{source_path}")
        suffix = source_path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"不支持的文件类型：{suffix or '(无扩展名)'}")

        request_image, coordinate_scale, request_size = prepare_request_image(
            source_path,
            self.config.max_image_side,
        )
        payload = {
            "file": base64.b64encode(request_image).decode("ascii"),
            "fileType": 1,
            # Text is sufficient for the API; visualization generation is
            # expensive for high-resolution documents.
            "visualize": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self.get_session().post(
                    self.config.service_url,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                response_data = response.json()
                return _restore_token_coordinates(
                    self.parse_response(response_data),
                    coordinate_scale,
                    request_size,
                    _extract_preprocessor_angle(response_data),
                )
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(min(2**attempt, 5))
        if last_error is None:
            raise RuntimeError("PP-OCRv6 请求失败，未获得具体错误")
        raise last_error
