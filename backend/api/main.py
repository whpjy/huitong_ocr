"""FastAPI entry point for the HunyuanOCR + PP-OCRv6 web demo."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ocr_manager.providers.base import IMAGE_EXTENSIONS

from .image_quality import ImageQualityReport, analyze_image_quality
from .mobile_config import MobileRecognitionConfig, load_mobile_config
from .schemas import (
    DocumentRecognitionResponse,
    ExtractionTiming,
    MobileRecognitionConfigResponse,
)
from .service import DocumentQualityError, ExtractionService, QualityImageContext


DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAVE_IMAGE_TTL_SECONDS = 10 * 60
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "api.log"
CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("huitong.api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


LOGGER = _configure_logging()


def _parse_model_key(model_key: str) -> tuple[str, str]:
    engine_type, separator, provider = model_key.strip().partition(":")
    if (
        separator != ":"
        or engine_type not in {"multimodal", "hybrid"}
        or provider != "hunyuan_ocr"
    ):
        raise ValueError("仅支持 HunyuanOCR 或 HunyuanOCR + PP-OCRv6")
    return engine_type, provider


def _max_upload_bytes() -> int:
    raw = os.getenv("OCR_API_MAX_UPLOAD_BYTES", "")
    if not raw:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("OCR_API_MAX_UPLOAD_BYTES 必须是整数") from exc
    if value < 1:
        raise RuntimeError("OCR_API_MAX_UPLOAD_BYTES 必须大于 0")
    return value


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "OCR_API_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cors_origin_regex() -> str | None:
    value = os.getenv(
        "OCR_API_CORS_ORIGIN_REGEX",
        (
            r"^https?://(?:10(?:\.\d{1,3}){3}|"
            r"192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}):5173$"
        ),
    ).strip()
    return value or None


def _upload_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if not suffix:
        suffix = CONTENT_TYPE_SUFFIXES.get(upload.content_type or "", "")
    if suffix not in {*IMAGE_EXTENSIONS, ".pdf"}:
        supported = "、".join(sorted({*IMAGE_EXTENSIONS, ".pdf"}))
        raise HTTPException(status_code=415, detail=f"不支持的文件格式；支持：{supported}")
    return suffix


async def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    suffix = _upload_suffix(upload)
    limit = _max_upload_bytes()
    content = await upload.read(limit + 1)
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"上传文件超过 {limit // (1024 * 1024)} MB 限制")
    return content, suffix


def _save_image_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "huitong-ocr-web-demo-images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_saved_images(now: float | None = None) -> None:
    current_time = now if now is not None else time.time()
    for path in _save_image_dir().iterdir():
        try:
            if path.is_file() and current_time - path.stat().st_mtime > SAVE_IMAGE_TTL_SECONDS:
                path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("failed to remove expired demo image path=%s", path)


def _saved_image_path(token: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        return None
    for suffix in IMAGE_EXTENSIONS:
        candidate = _save_image_dir() / f"{token}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service


def _client_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ValueError, KeyError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=502, detail=f"上游识别服务调用失败：{type(exc).__name__}: {exc}")


def create_app(
    service: ExtractionService | None = None,
    mobile_config: MobileRecognitionConfig | None = None,
    image_quality_checker: Callable[..., ImageQualityReport] = analyze_image_quality,
) -> FastAPI:
    application = FastAPI(
        title="Huitong OCR API",
        version="1.0.0",
        description="面向测试页面的 HunyuanOCR + PP-OCRv6 证件识别接口。",
    )
    application.state.extraction_service = service or ExtractionService()
    application.state.mobile_recognition_config = mobile_config or load_mobile_config()
    application.state.image_quality_checker = image_quality_checker

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_origin_regex=_cors_origin_regex(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/v1/mobile/save-image")
    async def prepare_save_image(file: UploadFile = File(...)) -> dict[str, object]:
        content, suffix = await _read_upload(file)
        if suffix not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=415, detail="仅支持保存图片文件")
        await run_in_threadpool(_cleanup_saved_images)
        token = uuid.uuid4().hex
        path = _save_image_dir() / f"{token}{suffix}"
        await run_in_threadpool(path.write_bytes, content)
        return {
            "token": token,
            "url": f"/api/v1/mobile/save-image/{token}",
            "expires_in_seconds": SAVE_IMAGE_TTL_SECONDS,
        }

    @application.get("/api/v1/mobile/save-image/{token}")
    async def download_saved_image(token: str) -> FileResponse:
        await run_in_threadpool(_cleanup_saved_images)
        path = _saved_image_path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="图片不存在或已过期")
        return FileResponse(path, filename=path.name)

    @application.get(
        "/api/v1/mobile/config",
        response_model=MobileRecognitionConfigResponse,
    )
    async def mobile_recognition_config(request: Request) -> MobileRecognitionConfigResponse:
        config = request.app.state.mobile_recognition_config
        return MobileRecognitionConfigResponse(
            name=config.name,
            model_key=config.model_key,
            label=config.label,
            pipeline_type=config.pipeline_type,
            image_quality_enabled=config.image_quality_enabled,
        )

    @application.post(
        "/api/v1/recognition/document",
        response_model=DocumentRecognitionResponse,
    )
    async def recognize_document(
        request: Request,
        file: UploadFile | None = File(default=None),
        front_file: UploadFile | None = File(default=None),
        back_file: UploadFile | None = File(default=None),
        model_key: str | None = Form(default=None),
        document_type: str = Form(...),
    ) -> DocumentRecognitionResponse:
        uploads: list[tuple[str, UploadFile]] = []
        if front_file is not None:
            uploads.append(("DG12", front_file))
        if back_file is not None:
            uploads.append(("DG13", back_file))
        if file is not None:
            uploads.append(("document", file))
        if not uploads:
            raise HTTPException(status_code=400, detail="至少需要上传一个证件文件")
        if document_type != "id_card" and len(uploads) > 1:
            raise HTTPException(status_code=422, detail="当前证件类型只接受一个文件")

        prepared = [(slot, upload, *(await _read_upload(upload))) for slot, upload in uploads]
        side_labels = {"DG12": "人像面", "DG13": "国徽面", "document": "证件图片"}
        mobile_config = request.app.state.mobile_recognition_config
        try:
            selected_model_key = model_key or mobile_config.model_key
            engine_type, provider = _parse_model_key(selected_model_key)
            with tempfile.TemporaryDirectory(prefix="document-recognition-") as directory:
                source_paths: list[Path] = []
                for slot, _upload, content, suffix in prepared:
                    parent = Path(directory) if slot == "document" else Path(directory) / slot
                    await run_in_threadpool(parent.mkdir, parents=True, exist_ok=True)
                    source_path = parent / f"upload{suffix}"
                    await run_in_threadpool(source_path.write_bytes, content)
                    source_paths.append(source_path)
                recognition_kwargs = {}
                if mobile_config.image_quality_enabled:
                    recognition_kwargs = {
                        "quality_checker": request.app.state.image_quality_checker,
                        "quality_contexts": [
                            QualityImageContext(
                                side=slot,
                                side_label=side_labels[slot],
                                filename=upload.filename or f"upload{suffix}",
                            )
                            for slot, upload, _content, suffix in prepared
                        ],
                    }
                result = await run_in_threadpool(
                    _service(request).recognize_document,
                    source_paths[0] if len(source_paths) == 1 else source_paths,
                    document_type,
                    engine_type,
                    provider,
                    **recognition_kwargs,
                )
        except DocumentQualityError as exc:
            raise HTTPException(status_code=422, detail=exc.as_detail()) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise _client_error(exc) from exc

        return DocumentRecognitionResponse(
            filename=" / ".join(
                upload.filename or f"upload{suffix}"
                for _slot, upload, _content, suffix in prepared
            ),
            document_type=result.document_type,
            document_name=result.document_name,
            pipeline_type=result.pipeline_type,
            provider=result.provider,
            model=result.model,
            source_text=result.text,
            fields=result.fields,
            timing=ExtractionTiming(
                ocr_seconds=round(result.ocr_elapsed, 6),
                llm_seconds=round(result.extraction_elapsed, 6),
                total_seconds=round(result.elapsed, 6),
            ),
        )

    return application


app = create_app()
