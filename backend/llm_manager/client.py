"""OpenAI-compatible chat-completions client and JSON parsing."""

from __future__ import annotations

import json
import base64
import io
import logging
import mimetypes
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from pathlib import Path

import requests

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:  # Optional unless a provider requests resizing.
    Image = ImageOps = None  # type: ignore[assignment]

from .config import LLMProviderConfig
from .hunyuan_ocr_adapter import (
    contains_spotting_coordinates,
    id_card_side,
    merge_document_pages,
    merge_id_card_pages,
    normalize_document_page,
    normalize_id_card_page,
    sanitize_document_records,
    spotting_markdown,
    spotting_text,
)
from .hunyuan_ocr_prompts import document_prompt, id_card_prompt
from .driver_refiner import refine_driver_fields
from .driver_archive_visual_refiner import refine_driver_archive_visually
from .profiles import ExtractionProfile
from .vehicle_refiner import refine_vehicle_fields, sync_vehicle_plate_fields
from .vision_text_adapter import adapt_profile_text


LOGGER = logging.getLogger("huitong.api")


JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _repair_truncated_final_json_string(raw: str) -> dict[str, Any] | None:
    """Repair only an object whose final string value ended before ``\"}``."""

    candidates = [raw]
    start = raw.find("{")
    if start > 0:
        candidates.append(raw[start:])

    for candidate in candidates:
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            if not exc.msg.startswith("Unterminated string"):
                continue
        else:
            continue

        try:
            repaired = json.loads(candidate + '\"}')
        except json.JSONDecodeError:
            continue
        if isinstance(repaired, dict):
            return repaired
    return None


def parse_json_value(
    text: str,
    *,
    repair_truncated_final_string: bool = False,
) -> Any:
    """Accept plain/fenced JSON or explanatory text around a JSON value."""

    raw = text.strip()
    fence_match = JSON_FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            if repair_truncated_final_string:
                repaired = _repair_truncated_final_json_string(raw)
                if repaired is not None:
                    return repaired
            raise
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            if repair_truncated_final_string:
                repaired = _repair_truncated_final_json_string(raw)
                if repaired is not None:
                    return repaired
            raise
    return value


def parse_json_object(
    text: str,
    *,
    repair_truncated_final_string: bool = False,
) -> dict[str, Any]:
    """Parse a response that is required to contain one JSON object."""

    value = parse_json_value(
        text,
        repair_truncated_final_string=repair_truncated_final_string,
    )
    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON object")
    return value


def parse_streaming_content(response: requests.Response) -> str:
    """Collect final answer content from an OpenAI-compatible SSE response."""

    content_parts: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = raw_line or ""
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            content_parts.append(str(content))
    content = "".join(content_parts)
    if not content.strip():
        raise RuntimeError("模型流式响应未返回最终 content")
    return content


class OpenAICompatibleClient:
    """Thread-safe model client using one requests.Session per worker."""

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        session_factory: Callable[[], requests.Session] = requests.Session,
    ) -> None:
        self.config = config
        self._session_factory = session_factory
        self._thread_local = threading.local()

    def _image_part(self, image_path: Path) -> dict[str, Any]:
        """Encode one image, optionally constraining its longest side."""

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        raw = image_path.read_bytes()
        max_side = self.config.vision_max_image_side
        if max_side is not None:
            if Image is None or ImageOps is None:
                raise RuntimeError(
                    f"Provider {self.config.name} 需要 Pillow 缩放图片"
                )
            with Image.open(io.BytesIO(raw)) as source:
                orientation = source.getexif().get(274, 1)
                if (
                    source.format == "JPEG"
                    and source.mode == "RGB"
                    and orientation in (None, 1)
                    and max(source.size) <= max_side
                ):
                    # Mobile uploads are already normalized to the model input
                    # size. Reusing those bytes avoids a full JPEG decode and
                    # encode on every Hunyuan request.
                    mime_type = "image/jpeg"
                else:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    if max(image.size) > max_side:
                        scale = max_side / max(image.size)
                        image = image.resize(
                            (
                                max(1, round(image.width * scale)),
                                max(1, round(image.height * scale)),
                            ),
                            Image.Resampling.LANCZOS,
                        )
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=95)
                    raw = output.getvalue()
                    mime_type = "image/jpeg"
        encoded = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }

    def get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self._session_factory()
            self._thread_local.session = session
        return session

    def _map_images_concurrently(
        self,
        image_paths: list[Path],
        worker: Callable[[Path], Any],
    ) -> list[Any]:
        """Run independent per-image calls concurrently, preserving input order."""

        if len(image_paths) <= 1:
            return [worker(image_path) for image_path in image_paths]
        worker_count = min(len(image_paths), max(1, self.config.concurrency))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"{self.config.name}-images",
        ) as executor:
            return list(executor.map(worker, image_paths))

    def build_payload(
        self,
        profile: ExtractionProfile,
        ocr_text: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": profile.build_messages(ocr_text),
            "temperature": self.config.temperature,
        }
        if self.config.response_format:
            payload["response_format"] = {
                "type": self.config.response_format,
            }
        # OpenAI SDK 的 extra_body 会合并到 HTTP JSON 顶层；这里保持同样语义。
        payload.update(self.config.extra_body)
        return payload

    def build_vision_payload(
        self,
        profile: ExtractionProfile,
        image_paths: list[Path],
        *,
        system_prefix: str = "",
        user_prefix: str = "",
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible multimodal chat-completions request."""

        if self.config.input_mode != "vision":
            raise ValueError(f"Provider {self.config.name} 不是多模态模型")
        if not image_paths:
            raise ValueError("多模态抽取至少需要一张图片")

        text_messages = profile.build_messages(
            "图片已作为本消息的视觉输入提供。请直接阅读所有图片，综合提取字段。"
        )
        for message in text_messages:
            message["content"] = message["content"].replace(
                "OCR 文本", "证件图片中的可见信息"
            ).replace("OCR", "图片")
        if system_prefix:
            text_messages[0]["content"] = (
                f"{system_prefix.strip()}\n\n{text_messages[0]['content']}"
            )
        if user_prefix:
            text_messages[1]["content"] = (
                f"{user_prefix.strip()}\n\n{text_messages[1]['content']}"
            )
        image_parts: list[dict[str, Any]] = []
        for image_path in image_paths:
            image_parts.append(self._image_part(image_path))
        text_part = {"type": "text", "text": text_messages[1]["content"]}
        if self.config.vision_content_order == "text_first":
            text_messages[1]["content"] = [text_part, *image_parts]
        else:
            text_messages[1]["content"] = [*image_parts, text_part]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": text_messages,
            "temperature": self.config.temperature,
        }
        if self.config.response_format:
            payload["response_format"] = {"type": self.config.response_format}
        payload.update(self.config.extra_body)
        return payload

    def build_general_vision_payload(
        self,
        image_paths: list[Path],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Build a free-form image-to-text request without JSON constraints."""

        if self.config.input_mode != "vision":
            raise ValueError(f"Provider {self.config.name} 不是多模态模型")
        if not image_paths:
            raise ValueError("图片识别至少需要一张图片")
        image_parts: list[dict[str, Any]] = []
        for image_path in image_paths:
            image_parts.append(self._image_part(image_path))
        text_part = {"type": "text", "text": user_prompt}
        content = (
            [text_part, *image_parts]
            if self.config.vision_content_order == "text_first"
            else [*image_parts, text_part]
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
        }
        payload.update(self.config.extra_body)
        return payload

    def build_labeled_text_payload(
        self,
        profile: ExtractionProfile,
        image_path: Path,
    ) -> dict[str, Any]:
        """Ask an OCR-oriented VLM for complete labelled document text."""

        if profile.key == "driver_license":
            prompt = (
                "请识别图片中驾驶证的全部文字，完整输出证号、姓名、性别、"
                "国籍、住址、出生日期、初次领证日期、准驾车型、有效期限和"
                "档案编号。逐行输出，不要省略。"
            )
        elif profile.key == "id_card":
            prompt = (
                "请识别图片中身份证的全部文字，完整输出姓名、性别、民族、"
                "出生日期、住址、公民身份号码、签发机关和有效期限。"
                "逐行输出，不要省略。"
            )
        else:
            field_names = "、".join(profile.field_names)
            prompt = (
                "请完整识别图片中证件的全部可见文字，保留字段标签和值，"
                f"重点不要遗漏：{field_names}。逐行输出，不要省略。"
            )
        text_part = {"type": "text", "text": prompt}
        image_part = self._image_part(image_path)
        content = (
            [text_part, image_part]
            if self.config.vision_content_order == "text_first"
            else [image_part, text_part]
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
        }
        payload.update(self.config.extra_body)
        return payload

    def build_hunyuan_id_card_payload(
        self,
        profile: ExtractionProfile,
        image_path: Path,
    ) -> dict[str, Any]:
        """Build one official-style HunyuanOCR information extraction call."""

        image_part = self._image_part(image_path)
        side = id_card_side(image_path)
        text_part = {
            "type": "text",
            "text": id_card_prompt(side),
        }
        content = (
            [text_part, image_part]
            if self.config.vision_content_order == "text_first"
            else [image_part, text_part]
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
        }
        payload.update(self.config.extra_body)
        return payload

    def build_hunyuan_document_payload(
        self,
        profile: ExtractionProfile,
        image_path: Path,
        *,
        prompt_key: str | None = None,
    ) -> dict[str, Any]:
        """Build one HunyuanOCR structured document request."""

        image_part = self._image_part(image_path)
        text_part = {
            "type": "text",
            "text": document_prompt(prompt_key or profile.key),
        }
        content = (
            [text_part, image_part]
            if self.config.vision_content_order == "text_first"
            else [image_part, text_part]
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
        }
        payload.update(self.config.extra_body)
        return payload

    def extract(
        self,
        profile: ExtractionProfile,
        ocr_text: str,
    ) -> dict[str, str]:
        """Call the model, parse its JSON object, and normalize all fields."""

        api_key = self.config.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"LLM Provider {self.config.name} 缺少 API Key；"
                "请填写 llm.yaml 的 api_key 或配置 api_key_env"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self.build_payload(profile, ocr_text)
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.get_session().post(
                    self.config.service_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                    stream=bool(payload.get("stream")),
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                if payload.get("stream"):
                    try:
                        content = parse_streaming_content(response)
                    finally:
                        response.close()
                else:
                    body = response.json()
                    content = (
                        body["choices"][0]["message"].get("content") or ""
                    )
                if not content.strip():
                    raise RuntimeError("模型返回空内容")
                return profile.normalize_records(parse_json_object(content))[0]
            except Exception as exc:  # noqa: BLE001 - preserve batch retries.
                last_error = exc
                if attempt >= self.config.max_attempts:
                    break
                time.sleep(min(2**attempt, 10))

        raise RuntimeError(f"模型调用失败：{last_error}")

    def extract_images(
        self,
        profile: ExtractionProfile,
        image_paths: list[Path],
        *,
        system_prefix: str = "",
        user_prefix: str = "",
    ) -> dict[str, str]:
        """Read document images and return normalized structured fields."""

        if self.config.vision_response_adapter in {
            "labeled_text",
            "hunyuan_ocr",
        }:
            return self.extract_image_records(
                profile,
                image_paths,
                system_prefix=system_prefix,
                user_prefix=user_prefix,
            )[0]
        return self._extract_payload(
            profile,
            self.build_vision_payload(
                profile,
                image_paths,
                system_prefix=system_prefix,
                user_prefix=user_prefix,
            ),
        )

    def extract_records(
        self,
        profile: ExtractionProfile,
        ocr_text: str,
    ) -> list[dict[str, str]]:
        """Extract every distinct document subject from merged OCR text."""

        content = self._complete_payload(self.build_payload(profile, ocr_text))
        return profile.normalize_records(parse_json_value(content))

    def extract_image_records(
        self,
        profile: ExtractionProfile,
        image_paths: list[Path],
        *,
        system_prefix: str = "",
        user_prefix: str = "",
        hunyuan_prompt_key: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract every distinct document subject from a group of images."""

        if (
            self.config.vision_response_adapter == "hunyuan_ocr"
            and profile.key == "id_card"
        ):
            page_records: list[dict[str, str]] = []
            page_errors: list[str] = []
            side_indexes: dict[Path, int] = {}
            side_counts: dict[str, int] = {"front": 0, "back": 0}
            for image_path in image_paths:
                side = id_card_side(image_path)
                if side:
                    side_indexes[image_path] = side_counts[side]
                    side_counts[side] += 1
            ordered_paths = sorted(
                image_paths,
                key=lambda path: (
                    len(image_paths) > 2 and id_card_side(path) != "back",
                    path.name.startswith("方向校正_"),
                ),
            )

            def extract_id_card_page(
                image_path: Path,
            ) -> tuple[list[dict[str, str]], str | None]:
                page_started = time.perf_counter()
                request_count = 0
                page_error: Exception | None = None
                side = id_card_side(image_path)
                shape_attempts = self.config.max_attempts + int(side == "back")
                for _shape_attempt in range(shape_attempts):
                    try:
                        request_count += 1
                        content = self._complete_payload(
                            self.build_hunyuan_id_card_payload(
                                profile,
                                image_path,
                            )
                        )
                        normalized_pages = normalize_id_card_page(
                            profile,
                            parse_json_value(
                                content,
                                repair_truncated_final_string=True,
                            ),
                            side,
                        )
                        for record in normalized_pages:
                            record["__hunyuan_side"] = side
                            record["__hunyuan_side_index"] = str(
                                side_indexes.get(image_path, -1)
                            )
                        page_error = None
                        LOGGER.info(
                            "hunyuan id-card page side=%s requests=%s records=%s "
                            "elapsed=%.3fs",
                            side or "unknown",
                            request_count,
                            len(normalized_pages),
                            time.perf_counter() - page_started,
                        )
                        return normalized_pages, None
                    except Exception as exc:  # noqa: BLE001
                        page_error = exc
                if page_error is not None and side == "back":
                    try:
                        request_count += 1
                        fallback_text = self._complete_payload(
                            self.build_labeled_text_payload(
                                profile,
                                image_path,
                            )
                        )
                        normalized_pages = normalize_id_card_page(
                            profile,
                            adapt_profile_text(profile, fallback_text),
                            side,
                        )
                        for record in normalized_pages:
                            record["__hunyuan_side"] = side
                            record["__hunyuan_side_index"] = str(
                                side_indexes.get(image_path, -1)
                            )
                        LOGGER.info(
                            "hunyuan id-card page side=%s requests=%s records=%s "
                            "fallback=True elapsed=%.3fs",
                            side,
                            request_count,
                            len(normalized_pages),
                            time.perf_counter() - page_started,
                        )
                        return normalized_pages, None
                    except Exception as exc:  # noqa: BLE001
                        page_error = exc
                LOGGER.info(
                    "hunyuan id-card page side=%s requests=%s records=0 "
                    "failed=True elapsed=%.3fs",
                    side or "unknown",
                    request_count,
                    time.perf_counter() - page_started,
                )
                return [], (
                    None
                    if page_error is None
                    else (
                        f"{image_path.name}: "
                        f"{type(page_error).__name__}: {page_error}"
                    )
                )

            page_outcomes = self._map_images_concurrently(
                ordered_paths,
                extract_id_card_page,
            )
            for normalized_pages, page_error in page_outcomes:
                page_records.extend(normalized_pages)
                if page_error:
                    page_errors.append(page_error)
            if not page_records:
                detail = "; ".join(page_errors[:3])
                raise ValueError(
                    "HunyuanOCR 所有身份证图片抽取失败"
                    + (f"：{detail}" if detail else "")
                )
            return merge_id_card_pages(profile, page_records)

        if (
            self.config.vision_response_adapter == "hunyuan_ocr"
            and profile.key in {
                "driver_license",
                "vehicle_license",
                "registration_certificate",
            }
        ):
            page_records: list[dict[str, str]] = []
            page_contents: list[str] = []
            page_errors: list[str] = []

            def extract_document_page(
                image_path: Path,
            ) -> tuple[list[dict[str, str]], str, str | None]:
                page_error: Exception | None = None
                # Response-shape retries are separate from transport retries:
                # HunyuanOCR can return a valid HTTP response in spotting mode.
                for _shape_attempt in range(max(2, self.config.max_attempts)):
                    try:
                        content = self._complete_payload(
                            self.build_hunyuan_document_payload(
                                profile,
                                image_path,
                                prompt_key=hunyuan_prompt_key,
                            )
                        )
                        try:
                            parsed_content = parse_json_value(
                                content,
                                repair_truncated_final_string=True,
                            )
                        except (json.JSONDecodeError, ValueError):
                            if not contains_spotting_coordinates(content):
                                raise
                            parsed_content = content
                        if contains_spotting_coordinates(parsed_content):
                            page_text = spotting_text(parsed_content)
                            records = sanitize_document_records(
                                profile,
                                adapt_profile_text(profile, page_text),
                            )
                            if not any(any(record.values()) for record in records):
                                raise ValueError(
                                    "HunyuanOCR 坐标文本未解析出证件字段"
                                )
                            page_content = page_text
                        else:
                            records = normalize_document_page(
                                profile,
                                parsed_content,
                            )
                            page_content = content
                        return records, page_content, None
                    except Exception as exc:  # noqa: BLE001
                        page_error = exc
                if page_error is not None:
                    try:
                        fallback_text = self._complete_payload(
                            self.build_labeled_text_payload(profile, image_path)
                        )
                        if contains_spotting_coordinates(fallback_text):
                            fallback_text = spotting_text(fallback_text)
                        fallback_records = sanitize_document_records(
                            profile,
                            adapt_profile_text(profile, fallback_text),
                        )
                        if not any(
                            any(record.values()) for record in fallback_records
                        ):
                            raise ValueError(
                                "HunyuanOCR 纯文本回退未解析出证件字段"
                            )
                        return fallback_records, fallback_text, None
                    except Exception as exc:  # noqa: BLE001
                        page_error = exc
                return [], "", (
                    None
                    if page_error is None
                    else (
                        f"{image_path.name}: "
                        f"{type(page_error).__name__}: {page_error}"
                    )
                )

            page_outcomes = self._map_images_concurrently(
                image_paths,
                extract_document_page,
            )
            for records, page_content, page_error in page_outcomes:
                page_records.extend(records)
                if page_content:
                    page_contents.append(page_content)
                if page_error:
                    page_errors.append(page_error)
            if not page_records:
                detail = "; ".join(page_errors[:3])
                raise ValueError(
                    "HunyuanOCR 所有证件图片抽取失败"
                    + (f"：{detail}" if detail else "")
                )
            records = merge_document_pages(profile, page_records)
            records = sanitize_document_records(profile, records)
            evidence_text = "\n\n".join(page_contents)
            if profile.key == "vehicle_license":
                records = [sync_vehicle_plate_fields(record) for record in records]
            if len(records) == 1 and profile.key == "driver_license":
                records[0] = refine_driver_fields(evidence_text, records[0])
                records[0] = refine_driver_archive_visually(
                    image_paths,
                    records[0],
                )
            elif len(records) == 1 and profile.key == "vehicle_license":
                records[0] = refine_vehicle_fields(evidence_text, records[0])
            return records

        if self.config.vision_response_adapter in {
            "labeled_text",
            "hunyuan_ocr",
        }:
            pages = self._map_images_concurrently(
                image_paths,
                lambda image_path: self._complete_payload(
                    self.build_labeled_text_payload(profile, image_path)
                ),
            )
            text = "\n\n".join(
                f"[图片 {index}]\n{page}"
                for index, page in enumerate(pages, start=1)
            )
            return adapt_profile_text(profile, text)

        payload = self.build_vision_payload(
            profile,
            image_paths,
            system_prefix=system_prefix,
            user_prefix=user_prefix,
        )
        content = self._complete_payload(payload)
        return profile.normalize_records(parse_json_value(content))

    def recognize_images(
        self,
        image_paths: list[Path],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return faithful free-form text read directly from images."""

        if self.config.vision_response_adapter in {
            "labeled_text",
            "hunyuan_ocr",
        }:
            pages = []
            for image_path in image_paths:
                page = self._complete_payload(
                    self.build_general_vision_payload(
                        [image_path],
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )
                if (
                    self.config.vision_response_adapter == "hunyuan_ocr"
                    and contains_spotting_coordinates(page)
                ):
                    page = spotting_markdown(page)
                pages.append(page)
            return "\n\n".join(
                f"[图片 {index}]\n{page}"
                for index, page in enumerate(pages, start=1)
            )
        return self._complete_payload(
            self.build_general_vision_payload(
                image_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )

    def _extract_payload(
        self,
        profile: ExtractionProfile,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        content = self._complete_payload(payload)
        return profile.normalize_records(parse_json_object(content))[0]

    def _complete_payload(self, payload: dict[str, Any]) -> str:
        """Execute one request and return its final assistant content."""

        api_key = self.config.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"LLM Provider {self.config.name} 缺少 API Key，请配置 "
                f"{self.config.api_key_env or 'llm.yaml api_key'}"
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.get_session().post(
                    self.config.service_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                    stream=bool(payload.get("stream")),
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                if payload.get("stream"):
                    try:
                        content = parse_streaming_content(response)
                    finally:
                        response.close()
                else:
                    body = response.json()
                    content = body["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise RuntimeError("模型返回空内容")
                return content
            except Exception as exc:  # noqa: BLE001 - preserve request retries.
                last_error = exc
                if attempt >= self.config.max_attempts:
                    break
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"模型调用失败：{last_error}")
