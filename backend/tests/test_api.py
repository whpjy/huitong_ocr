from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.image_quality import ImageQualityReport
from api.main import create_app
from api.mobile_config import MobileRecognitionConfig
from api.service import DocumentRecognitionExecution


class FakeRecognitionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int]] = []

    def recognize_document(
        self,
        source_path: Path | list[Path],
        document_type: str,
        engine_type: str,
        provider_name: str,
        **_kwargs,
    ) -> DocumentRecognitionExecution:
        paths = source_path if isinstance(source_path, list) else [source_path]
        assert all(path.read_bytes() == b"fake-image" for path in paths)
        self.calls.append((document_type, engine_type, provider_name, len(paths)))
        return DocumentRecognitionExecution(
            document_type=document_type,
            document_name="身份证" if document_type == "id_card" else "驾驶证",
            pipeline_type=engine_type,
            provider=provider_name,
            model="HunyuanOCR + PP-OCRv6" if engine_type == "hybrid" else "HunyuanOCR",
            text="姓名 张三" if engine_type == "hybrid" else "",
            fields={"姓名": "张三"},
            ocr_elapsed=0.1,
            extraction_elapsed=0.2,
            elapsed=0.25,
        )


def _client(*, model_key: str = "hybrid:hunyuan_ocr") -> tuple[TestClient, FakeRecognitionService]:
    service = FakeRecognitionService()
    config = MobileRecognitionConfig(
        name="hunyuan_ocr_ppocr" if model_key.startswith("hybrid") else "hunyuan_ocr",
        model_key=model_key,
        label="HunyuanOCR + PP-OCRv6" if model_key.startswith("hybrid") else "HunyuanOCR",
    )
    app = create_app(
        service=service,
        mobile_config=config,
        image_quality_checker=lambda *_args: ImageQualityReport(issues=(), metrics={}),
    )
    return TestClient(app), service


def test_health_and_mobile_config() -> None:
    client, _service = _client()
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/api/v1/mobile/config")
    assert response.status_code == 200
    assert response.json()["model_key"] == "hybrid:hunyuan_ocr"


def test_hybrid_document_recognition() -> None:
    client, service = _client()
    response = client.post(
        "/api/v1/recognition/document",
        data={"document_type": "driver_license"},
        files={"file": ("driver.jpg", b"fake-image", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["fields"] == {"姓名": "张三"}
    assert service.calls == [("driver_license", "hybrid", "hunyuan_ocr", 1)]


def test_hunyuan_id_card_front_and_back() -> None:
    client, service = _client(model_key="multimodal:hunyuan_ocr")
    response = client.post(
        "/api/v1/recognition/document",
        data={"document_type": "id_card"},
        files={
            "front_file": ("front.jpg", b"fake-image", "image/jpeg"),
            "back_file": ("back.jpg", b"fake-image", "image/jpeg"),
        },
    )
    assert response.status_code == 200
    assert service.calls == [("id_card", "multimodal", "hunyuan_ocr", 2)]


def test_other_models_are_rejected() -> None:
    client, service = _client()
    response = client.post(
        "/api/v1/recognition/document",
        data={"document_type": "driver_license", "model_key": "multimodal:qwen"},
        files={"file": ("driver.jpg", b"fake-image", "image/jpeg")},
    )
    assert response.status_code == 422
    assert service.calls == []


def test_save_image_round_trip() -> None:
    client, _service = _client()
    created = client.post(
        "/api/v1/mobile/save-image",
        files={"file": ("capture.jpg", b"fake-image", "image/jpeg")},
    )
    assert created.status_code == 200
    downloaded = client.get(created.json()["url"])
    assert downloaded.status_code == 200
    assert downloaded.content == b"fake-image"


def test_non_id_document_rejects_multiple_files() -> None:
    client, _service = _client()
    response = client.post(
        "/api/v1/recognition/document",
        data={"document_type": "driver_license"},
        files={
            "front_file": ("front.jpg", b"fake-image", "image/jpeg"),
            "back_file": ("back.jpg", b"fake-image", "image/jpeg"),
        },
    )
    assert response.status_code == 422
