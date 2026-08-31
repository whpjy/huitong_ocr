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


class FakeApplicationRecognitionService:
    def list_source_images(self, application_no: str) -> dict[str, object]:
        assert application_no == "1108011200"
        return {
            "application_no": application_no,
            "groups": [
                {
                    "material": "id_card_front",
                    "label": "身份证正面",
                    "material_code": "DG12",
                    "files": [
                        {"name": "front.jpg", "material_code": "DG12"}
                    ],
                }
            ],
        }

    def recognize(
        self,
        application_no: str,
        engine_type: str,
        provider: str,
    ) -> dict[str, object]:
        assert application_no == "1108011200"
        assert (engine_type, provider) == ("hybrid", "hunyuan_ocr")
        return {
            "application_no": application_no,
            "status": "completed",
            "documents": {
                "id_cards": [],
                "driver_licenses": [],
                "vehicle_licenses": [],
            },
            "validations": [],
            "errors": [],
            "summary": {
                "id_card_count": 0,
                "driver_license_count": 0,
                "vehicle_license_count": 0,
                "person_count": 0,
                "duplicate_file_count": 0,
                "error_count": 0,
                "missing_documents": ["身份证", "驾驶证", "行驶证"],
                "elapsed_seconds": 0.01,
            },
        }


class FakeApplicationFileService(FakeApplicationRecognitionService):
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def source_file(
        self,
        application_no: str,
        material_code: str,
        file_name: str,
    ) -> Path:
        assert (application_no, material_code, file_name) == (
            "1108011200",
            "DG14",
            "驾驶证.jpg",
        )
        return self.source_path

    def source_thumbnail(
        self,
        application_no: str,
        material_code: str,
        file_name: str,
    ) -> bytes:
        self.source_file(application_no, material_code, file_name)
        return b"normalized-thumbnail"


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


def test_application_recognition_endpoint() -> None:
    service = FakeRecognitionService()
    config = MobileRecognitionConfig(
        name="hunyuan_ocr_ppocr",
        model_key="hybrid:hunyuan_ocr",
        label="HunyuanOCR + PP-OCRv6",
    )
    app = create_app(
        service=service,
        mobile_config=config,
        application_service=FakeApplicationRecognitionService(),
    )

    response = TestClient(app).post(
        "/api/v1/applications/1108011200/recognition"
    )

    assert response.status_code == 200
    assert response.json()["application_no"] == "1108011200"
    assert response.json()["summary"]["missing_documents"] == [
        "身份证",
        "驾驶证",
        "行驶证",
    ]


def test_application_source_images_endpoint() -> None:
    service = FakeRecognitionService()
    config = MobileRecognitionConfig(
        name="hunyuan_ocr_ppocr",
        model_key="hybrid:hunyuan_ocr",
        label="HunyuanOCR + PP-OCRv6",
    )
    app = create_app(
        service=service,
        mobile_config=config,
        application_service=FakeApplicationRecognitionService(),
    )

    response = TestClient(app).get("/api/v1/applications/1108011200/files")

    assert response.status_code == 200
    assert response.json()["groups"][0] == {
        "material": "id_card_front",
        "label": "身份证正面",
        "material_code": "DG12",
        "files": [{"name": "front.jpg", "material_code": "DG12"}],
    }


def test_application_source_file_can_be_viewed_inline(tmp_path: Path) -> None:
    source = tmp_path / "驾驶证.jpg"
    source.write_bytes(b"source-image")
    service = FakeRecognitionService()
    config = MobileRecognitionConfig(
        name="hunyuan_ocr_ppocr",
        model_key="hybrid:hunyuan_ocr",
        label="HunyuanOCR + PP-OCRv6",
    )
    app = create_app(
        service=service,
        mobile_config=config,
        application_service=FakeApplicationFileService(source),
    )

    response = TestClient(app).get(
        "/api/v1/applications/1108011200/files/DG14/%E9%A9%BE%E9%A9%B6%E8%AF%81.jpg"
    )

    assert response.status_code == 200
    assert response.content == b"source-image"
    assert response.headers["content-type"] == "image/jpeg"
    assert "content-disposition" not in response.headers


def test_application_source_thumbnail_is_normalized(tmp_path: Path) -> None:
    source = tmp_path / "驾驶证.jpg"
    source.write_bytes(b"source-image")
    service = FakeRecognitionService()
    config = MobileRecognitionConfig(
        name="hunyuan_ocr_ppocr",
        model_key="hybrid:hunyuan_ocr",
        label="HunyuanOCR + PP-OCRv6",
    )
    app = create_app(
        service=service,
        mobile_config=config,
        application_service=FakeApplicationFileService(source),
    )

    response = TestClient(app).get(
        "/api/v1/applications/1108011200/files/DG14/%E9%A9%BE%E9%A9%B6%E8%AF%81.jpg?thumbnail=true"
    )

    assert response.status_code == 200
    assert response.content == b"normalized-thumbnail"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, max-age=3600"
