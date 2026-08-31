"""Shared application service built on the existing OCR and LLM managers."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm_manager.client import OpenAICompatibleClient
from llm_manager.config import (
    DocumentCropConfig,
    LLMManagerConfig,
    load_multimodal_config,
)
from llm_manager.image_preprocess import (
    correct_document_orientation,
    crop_document,
)
from llm_manager.hybrid_pipeline import collect_ppocr_document, run_hybrid_document
from llm_manager.profiles import ExtractionProfiles, load_profiles
from llm_manager.pdf_conversion import recognition_images
from ocr_manager.config import OCRManagerConfig, load_config
from ocr_manager.orientation import detect_document_orientations
from ocr_manager.providers import create_provider
from ocr_manager.providers.base import OCRProvider

from .image_quality import ImageQualityReport


logger = logging.getLogger("huitong.api")


SUPPORTED_DOCUMENT_KEYS = (
    "id_card",
    "driver_license",
    "vehicle_license",
    "registration_certificate",
)

ID_CARD_FRONT_FIELDS = (
    "证件类型", "姓名", "身份证号", "出生日期", "性别", "民族", "住址",
)
ID_CARD_BACK_FIELDS = ("签发机关", "有效期")


def _merge_page_texts(results: list[object]) -> str:
    texts = [str(getattr(result, "text", "") or "") for result in results]
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(
        f"[第 {index} 页]\n{text}"
        for index, text in enumerate(texts, start=1)
        if text.strip()
    )


def _prepare_document_images(
    source_paths: list[Path],
    crop_config: DocumentCropConfig,
) -> list[Path]:
    """Prepare independent document sides concurrently and preserve order."""

    started = time.perf_counter()

    def expand_source(source: Path) -> list[tuple[Path, Path, int]]:
        return [
            (source, image_path, index)
            for index, image_path in enumerate(
                recognition_images(
                    source,
                    source.parent / f"{source.stem}_PDF页面",
                ),
                start=1,
            )
        ]

    if len(source_paths) <= 1:
        grouped = [expand_source(source) for source in source_paths]
    else:
        with ThreadPoolExecutor(
            max_workers=min(2, len(source_paths)),
            thread_name_prefix="mobile-document-preprocess",
        ) as executor:
            grouped = list(executor.map(expand_source, source_paths))
    expanded = [item for group in grouped for item in group]
    detections = detect_document_orientations([item[1] for item in expanded])

    def prepare_image(
        item: tuple[tuple[Path, Path, int], dict[str, object]],
    ) -> Path:
        (source, image_path, index), detection = item
        orientation_result = correct_document_orientation(
            image_path,
            source.parent / f"{source.stem}.page-{index:03d}.oriented.jpg",
            detection,
        )
        crop_result = crop_document(
            orientation_result.path,
            source.parent / f"{source.stem}.page-{index:03d}.document-crop.jpg",
            crop_config,
        )
        return crop_result.path

    work = list(zip(expanded, detections))
    if len(work) <= 1:
        prepared = [prepare_image(item) for item in work]
    else:
        with ThreadPoolExecutor(
            max_workers=min(2, len(work)),
            thread_name_prefix="mobile-document-transform",
        ) as executor:
            prepared = list(executor.map(prepare_image, work))
    logger.info(
        "mobile document preprocessing images=%s parallel=%s elapsed=%.3fs",
        len(prepared),
        len(source_paths) > 1,
        time.perf_counter() - started,
    )
    return prepared


@dataclass(frozen=True)
class OCRExecution:
    document_type: str
    document_name: str
    provider: str
    text: str
    page_count: int
    elapsed: float


@dataclass(frozen=True)
class LLMExecution:
    document_type: str
    document_name: str
    provider: str
    model: str
    fields: dict[str, str]
    elapsed: float


@dataclass(frozen=True)
class GeneralRecognitionExecution:
    engine_type: str
    provider: str
    model: str
    text: str
    elapsed: float


@dataclass(frozen=True)
class DocumentRecognitionExecution:
    document_type: str
    document_name: str
    pipeline_type: str
    provider: str
    model: str
    text: str
    fields: dict[str, str]
    ocr_elapsed: float
    extraction_elapsed: float
    elapsed: float


@dataclass(frozen=True)
class QualityImageContext:
    side: str
    side_label: str
    filename: str


class DocumentQualityError(Exception):
    """Stop the pipeline at the quality gate before Hunyuan is invoked."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        issues: list[dict[str, object]],
        reports: list[dict[str, object]],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = issues
        self.reports = reports

    def as_detail(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "issues": self.issues,
        }
        if self.reports:
            detail["reports"] = self.reports
        return detail


class ExtractionService:
    """Resolve document aliases and reuse configured provider clients."""

    def __init__(
        self,
        *,
        ocr_config: OCRManagerConfig | None = None,
        llm_config: LLMManagerConfig | None = None,
        profiles: ExtractionProfiles | None = None,
    ) -> None:
        self.ocr_config = ocr_config or load_config()
        # The slim service has no text-only LLM provider configuration.
        # Keep the optional injection hook for focused unit tests only.
        self.llm_config = llm_config
        self.profiles = profiles or load_profiles()
        self._ocr_providers: dict[str, OCRProvider] = {}
        self._ocr_semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._llm_clients: dict[str, OpenAICompatibleClient] = {}
        self._client_lock = threading.Lock()

    def resolve_document(self, value: str) -> tuple[str, str]:
        document_type = value.strip()
        for key in SUPPORTED_DOCUMENT_KEYS:
            document = self.ocr_config.get_document(key)
            if document_type in {key, document.name}:
                return key, document.name
        choices = "、".join(
            self.ocr_config.get_document(key).name
            for key in SUPPORTED_DOCUMENT_KEYS
        )
        raise ValueError(f"不支持的材料类型：{value}；可选：{choices}")

    def _get_ocr_provider(
        self,
        document_type: str,
        provider_name: str | None,
    ) -> OCRProvider:
        document = self.ocr_config.get_document(document_type)
        config = self.ocr_config.get_provider(provider_name)
        self.ocr_config.validate_pair(document, config)
        with self._client_lock:
            provider = self._ocr_providers.get(config.name)
            if provider is None:
                provider = create_provider(config)
                self._ocr_providers[config.name] = provider
        return provider

    def _get_llm_client(
        self,
        provider_name: str | None,
    ) -> OpenAICompatibleClient:
        if self.llm_config is None:
            raise ValueError("精简服务未配置文本 LLM，仅支持 HunyuanOCR 图片识别")
        config = self.llm_config.get_provider(provider_name)
        with self._client_lock:
            client = self._llm_clients.get(config.name)
            if client is None:
                client = OpenAICompatibleClient(config)
                self._llm_clients[config.name] = client
        return client

    def _multimodal_client(
        self,
        provider_name: str | None,
    ) -> tuple[OpenAICompatibleClient, LLMManagerConfig]:
        # Reload on every single-image request so edits to multimodal.yaml are
        # reflected without rebuilding the frontend or restarting the backend.
        config = load_multimodal_config()
        provider = config.get_provider(provider_name)
        if provider.input_mode != "vision":
            raise ValueError(f"模型 {provider.name} 不支持图片输入")
        return OpenAICompatibleClient(provider), config

    def model_options(self) -> dict[str, object]:
        multimodal = load_multimodal_config()
        pp_ocr = self.ocr_config.get_provider("pp_ocrv6")
        models = [
            {
                "key": f"ocr:{pp_ocr.name}",
                "kind": "ocr",
                "provider": pp_ocr.name,
                "model": "PP-OCRv6",
                "label": "PP-OCRv6",
            }
        ]
        models.extend(
            {
                "key": f"multimodal:{provider.name}",
                "kind": "multimodal",
                "provider": provider.name,
                "model": provider.model,
                "label": provider.display_name or provider.model,
            }
            for provider in multimodal.providers.values()
            if provider.enabled and provider.input_mode == "vision"
        )
        hunyuan = multimodal.providers.get("hunyuan_ocr")
        if (
            hunyuan is not None
            and hunyuan.enabled
            and hunyuan.input_mode == "vision"
        ):
            models.append({
                "key": "hybrid:hunyuan_ocr",
                "kind": "hybrid",
                "provider": "hunyuan_ocr",
                "model": f"{hunyuan.model} + PP-OCRv6",
                "label": "HunyuanOCR + PP-OCRv6（双路并行）",
            })
        documents = []
        for key in SUPPORTED_DOCUMENT_KEYS:
            profile = self.profiles.get(key)
            if key == "id_card":
                upload_slots = [
                    {
                        "key": "front",
                        "label": "身份证正面（人像面）",
                        "required": True,
                        "fields": list(ID_CARD_FRONT_FIELDS),
                    },
                    {
                        "key": "back",
                        "label": "身份证反面（国徽面）",
                        "required": True,
                        "fields": list(ID_CARD_BACK_FIELDS),
                    },
                ]
            else:
                upload_slots = [
                    {
                        "key": "document",
                        "label": f"{profile.name}图片",
                        "required": True,
                        "fields": list(profile.field_names),
                    }
                ]
            documents.append(
                {
                    "key": key,
                    "name": self.ocr_config.get_document(key).name,
                    "fields": list(profile.field_names),
                    "upload_slots": upload_slots,
                }
            )
        return {
            "models": models,
            "documents": documents,
            "default_multimodal_provider": multimodal.active_provider,
        }

    def recognize_general(
        self,
        source_path: Path,
        engine_type: str,
        provider_name: str,
    ) -> GeneralRecognitionExecution:
        started = time.perf_counter()
        image_paths = recognition_images(
            source_path, source_path.parent / f"{source_path.stem}_PDF页面"
        )
        if engine_type == "ocr":
            config = self.ocr_config.get_provider(provider_name)
            if config.name != "pp_ocrv6":
                raise ValueError("通用识别当前仅支持 PP-OCRv6")
            with self._client_lock:
                provider = self._ocr_providers.get(config.name)
                if provider is None:
                    provider = create_provider(config)
                    self._ocr_providers[config.name] = provider
            results = [provider.recognize(path) for path in image_paths]
            return GeneralRecognitionExecution(
                engine_type="ocr",
                provider=config.name,
                model="PP-OCRv6",
                text=_merge_page_texts(results),
                elapsed=time.perf_counter() - started,
            )
        if engine_type != "multimodal":
            raise ValueError("不支持的识别模型类型")
        client, config = self._multimodal_client(provider_name)
        prompt_prefix = (
            "hunyuan_general"
            if client.config.name == "hunyuan_ocr"
            else "general"
        )
        text = client.recognize_images(
            image_paths,
            system_prompt=config.prompts[f"{prompt_prefix}_system"],
            user_prompt=config.prompts[f"{prompt_prefix}_user"],
        )
        return GeneralRecognitionExecution(
            engine_type="multimodal",
            provider=client.config.name,
            model=client.config.model,
            text=text,
            elapsed=time.perf_counter() - started,
        )

    def _ppocr_resources(
        self,
        document_type: str,
    ) -> tuple[OCRProvider, threading.BoundedSemaphore]:
        provider = self._get_ocr_provider(document_type, "pp_ocrv6")
        config = self.ocr_config.get_provider("pp_ocrv6")
        with self._client_lock:
            semaphore = self._ocr_semaphores.get(config.name)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(max(1, config.concurrency))
                self._ocr_semaphores[config.name] = semaphore
        return provider, semaphore

    @staticmethod
    def _run_quality_gate(
        *,
        images: list[Path],
        document_type: str,
        ppocr_files: list[dict[str, Any]],
        checker: Callable[..., ImageQualityReport],
        contexts: list[QualityImageContext] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Evaluate prepared images; PP artifacts are ready for future rules."""

        failed_ocr = [item for item in ppocr_files if item.get("error")]
        if failed_ocr:
            raise DocumentQualityError(
                code="IMAGE_QUALITY_CHECK_FAILED",
                message="图片质量检测失败，请重新拍摄后再试",
                issues=[
                    {
                        "side": "document",
                        "side_label": "证件图片",
                        "code": "PPOCR_QUALITY_GATE_FAILED",
                        "message": str(failed_ocr[0].get("error") or "PP-OCR调用失败"),
                    }
                ],
                reports=[],
            )

        def check(indexed_path: tuple[int, Path]) -> tuple[int, ImageQualityReport]:
            index, image_path = indexed_path
            raw_ppocr_file = ppocr_files[index] if index < len(ppocr_files) else {}
            ppocr_file = dict(raw_ppocr_file)
            if contexts is not None and index < len(contexts):
                ppocr_file["quality_side"] = contexts[index].side
            content = image_path.read_bytes()
            try:
                inspect.signature(checker).bind(content, document_type, ppocr_file)
            except (TypeError, ValueError):
                # Keep compatibility with existing two-argument custom checkers.
                report = checker(content, document_type)
            else:
                report = checker(content, document_type, ppocr_file)
            return index, report

        indexed_images = list(enumerate(images))
        try:
            if len(indexed_images) <= 1:
                checked = [check(item) for item in indexed_images]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(2, len(indexed_images)),
                    thread_name_prefix="mobile-quality-gate",
                ) as executor:
                    checked = list(executor.map(check, indexed_images))
        except Exception as exc:
            raise DocumentQualityError(
                code="IMAGE_QUALITY_CHECK_FAILED",
                message="图片质量检测失败，请重新拍摄后再试",
                issues=[
                    {
                        "side": "document",
                        "side_label": "证件图片",
                        "code": "QUALITY_CHECK_FAILED",
                        "message": f"质量检测失败：{type(exc).__name__}",
                    }
                ],
                reports=[],
            ) from exc

        issues: list[dict[str, object]] = []
        reports: list[dict[str, object]] = []
        for index, report in checked:
            context = (
                contexts[index]
                if contexts is not None and index < len(contexts)
                else QualityImageContext("document", "证件图片", images[index].name)
            )
            metrics = dict(report.metrics)
            ocr_file = ppocr_files[index] if index < len(ppocr_files) else {}
            metrics["ppocr_token_count"] = len(ocr_file.get("tokens") or [])
            reports.append(
                {
                    "side": context.side,
                    "side_label": context.side_label,
                    "filename": context.filename,
                    "metrics": metrics,
                }
            )
            issues.extend(
                {
                    "side": context.side,
                    "side_label": context.side_label,
                    **issue.as_dict(),
                }
                for issue in report.issues
            )
        return issues, reports

    def recognize_document(
        self,
        source_path: Path | list[Path],
        document_type: str,
        engine_type: str,
        provider_name: str,
        *,
        quality_checker: Callable[..., ImageQualityReport] | None = None,
        quality_contexts: list[QualityImageContext] | None = None,
    ) -> DocumentRecognitionExecution:
        started = time.perf_counter()
        source_paths = source_path if isinstance(source_path, list) else [source_path]
        if not source_paths:
            raise ValueError("至少需要上传一个证件文件")
        key, name = self.resolve_document(document_type)
        if engine_type == "hybrid":
            if provider_name != "hunyuan_ocr":
                raise ValueError("混合识别当前仅支持 HunyuanOCR + PP-OCRv6")
            client, config = self._multimodal_client(provider_name)
            profile = self.profiles.get(key)
            processed_images = _prepare_document_images(
                source_paths,
                config.document_crop,
            )
            ocr_provider, ocr_semaphore = self._ppocr_resources(key)
            ppocr_result_callback = None
            if quality_checker is not None:
                def run_quality_gate(
                    ppocr_outcome: tuple[list[dict[str, Any]], float],
                ) -> None:
                    quality_issues, quality_reports = self._run_quality_gate(
                        images=processed_images,
                        document_type=key,
                        ppocr_files=ppocr_outcome[0],
                        checker=quality_checker,
                        contexts=quality_contexts,
                    )
                    if quality_issues:
                        raise DocumentQualityError(
                            code="IMAGE_QUALITY_REJECTED",
                            message="照片质量不符合要求，请重新拍摄",
                            issues=quality_issues,
                            reports=quality_reports,
                        )

                ppocr_result_callback = run_quality_gate
            result, process = run_hybrid_document(
                application_no="single-document",
                images=processed_images,
                profile=profile,
                client=client,
                ocr_provider=ocr_provider,
                artifact_root=source_paths[0].parent / "混合复核",
                hunyuan_prompt_key=key,
                ppocr_semaphore=ocr_semaphore,
                prefer_hunyuan_latency=True,
                ppocr_result_callback=ppocr_result_callback,
            )
            if not result.success:
                raise ValueError(result.error or "混合识别未抽取出有效字段")
            ocr_files = process.get("files", [])
            ocr_pages = [str(item.get("text") or "") for item in ocr_files]
            ocr_text = (
                ocr_pages[0]
                if len(ocr_pages) == 1
                else "\n\n".join(
                    f"[第 {index} 页]\n{text}"
                    for index, text in enumerate(ocr_pages, start=1)
                    if text.strip()
                )
            )
            primary = process.get("primary_execution", {})
            logger.info(
                "mobile hybrid cutoff=%s ppocr_status=%s ppocr_available=%s "
                "hunyuan_elapsed=%.3fs ppocr_elapsed=%.3fs",
                primary.get("cutoff_policy"),
                primary.get("ppocr_status"),
                primary.get("ppocr_available_for_fusion"),
                float(primary.get("multimodal_seconds") or 0.0),
                float(primary.get("ppocr_seconds") or 0.0),
            )
            return DocumentRecognitionExecution(
                document_type=key,
                document_name=name,
                pipeline_type="hybrid",
                provider="hunyuan_ocr+pp_ocrv6",
                model=f"{client.config.model} + PP-OCRv6",
                text=ocr_text,
                fields=result.fields,
                ocr_elapsed=float(primary.get("ppocr_seconds") or 0.0),
                extraction_elapsed=float(
                    primary.get("multimodal_seconds") or 0.0
                ),
                elapsed=time.perf_counter() - started,
            )
        if engine_type == "ocr":
            if len(source_paths) <= 1:
                ocr_results = [
                    self.recognize(path, key, provider_name)
                    for path in source_paths
                ]
            else:
                provider_config = self.ocr_config.get_provider(provider_name)
                with ThreadPoolExecutor(
                    max_workers=min(
                        len(source_paths),
                        max(1, provider_config.concurrency),
                    ),
                    thread_name_prefix="document-ocr-pages",
                ) as executor:
                    ocr_results = list(
                        executor.map(
                            lambda path: self.recognize(
                                path,
                                key,
                                provider_name,
                            ),
                            source_paths,
                        )
                    )
            ocr_text = _merge_page_texts(ocr_results)
            if not ocr_text.strip():
                raise ValueError("OCR 未识别出文本，无法进行字段抽取")
            llm_result = self.extract(ocr_text, key)
            return DocumentRecognitionExecution(
                document_type=key,
                document_name=name,
                pipeline_type="ocr_llm",
                provider=ocr_results[0].provider,
                model=f"PP-OCRv6 + {llm_result.model}",
                text=ocr_text,
                fields=llm_result.fields,
                ocr_elapsed=sum(result.elapsed for result in ocr_results),
                extraction_elapsed=llm_result.elapsed,
                elapsed=time.perf_counter() - started,
            )
        if engine_type != "multimodal":
            raise ValueError("不支持的识别模型类型")
        client, config = self._multimodal_client(provider_name)
        profile = self.profiles.get(key)
        processed_images = _prepare_document_images(
            source_paths,
            config.document_crop,
        )
        gate_ppocr_elapsed = 0.0
        if quality_checker is not None:
            ocr_provider, ocr_semaphore = self._ppocr_resources(key)
            ppocr_files, gate_ppocr_elapsed = collect_ppocr_document(
                images=processed_images,
                ocr_provider=ocr_provider,
                ppocr_semaphore=ocr_semaphore,
            )
            quality_issues, quality_reports = self._run_quality_gate(
                images=processed_images,
                document_type=key,
                ppocr_files=ppocr_files,
                checker=quality_checker,
                contexts=quality_contexts,
            )
            if quality_issues:
                raise DocumentQualityError(
                    code="IMAGE_QUALITY_REJECTED",
                    message="照片质量不符合要求，请重新拍摄",
                    issues=quality_issues,
                    reports=quality_reports,
                )
        extraction_started = time.perf_counter()
        fields = client.extract_images(
            profile,
            processed_images,
            system_prefix=config.prompts.get("document_system_prefix", ""),
            user_prefix=config.prompts.get("document_user_prefix", ""),
        )
        extraction_elapsed = time.perf_counter() - extraction_started
        return DocumentRecognitionExecution(
            document_type=key,
            document_name=name,
            pipeline_type="multimodal",
            provider=client.config.name,
            model=client.config.model,
            text="",
            fields=fields,
            ocr_elapsed=gate_ppocr_elapsed,
            extraction_elapsed=extraction_elapsed,
            elapsed=time.perf_counter() - started,
        )

    def recognize(
        self,
        source_path: Path,
        document_type: str,
        provider_name: str | None = None,
    ) -> OCRExecution:
        key, name = self.resolve_document(document_type)
        provider = self._get_ocr_provider(key, provider_name)
        started = time.perf_counter()
        image_paths = recognition_images(
            source_path, source_path.parent / f"{source_path.stem}_PDF页面"
        )
        results = [provider.recognize(path) for path in image_paths]
        text = _merge_page_texts(results)
        return OCRExecution(
            document_type=key,
            document_name=name,
            provider=provider.name,
            text=text,
            page_count=sum(result.page_count for result in results),
            elapsed=time.perf_counter() - started,
        )

    def extract(
        self,
        ocr_text: str,
        document_type: str,
        provider_name: str | None = None,
    ) -> LLMExecution:
        key, name = self.resolve_document(document_type)
        profile = self.profiles.get(key)
        client = self._get_llm_client(provider_name)
        started = time.perf_counter()
        fields = client.extract(profile, ocr_text)
        return LLMExecution(
            document_type=key,
            document_name=name,
            provider=client.config.name,
            model=client.config.model,
            fields=fields,
            elapsed=time.perf_counter() - started,
        )
