from api.main import app
from llm_manager.config import load_multimodal_config
from llm_manager.profiles import load_profiles
from ocr_manager.config import load_config


def test_only_required_providers_and_documents_are_configured() -> None:
    assert set(load_config().providers) == {"pp_ocrv6"}
    assert set(load_multimodal_config().providers) == {"hunyuan_ocr"}
    assert set(load_config().documents) == {
        "id_card",
        "driver_license",
        "registration_certificate",
        "vehicle_license",
    }
    assert set(load_profiles().profiles) == set(load_config().documents)


def test_only_web_demo_business_routes_are_exposed() -> None:
    paths = {
        route.path
        for route in app.routes
        if route.path.startswith("/api/") or route.path == "/health"
    }
    assert paths == {
        "/health",
        "/api/v1/mobile/config",
        "/api/v1/mobile/save-image",
        "/api/v1/mobile/save-image/{token}",
        "/api/v1/recognition/document",
        "/api/v1/applications/{application_no}/recognition",
        "/api/v1/applications/{application_no}/files",
        "/api/v1/applications/{application_no}/files/{material_code}/{file_name}",
    }


def test_service_urls_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("PPOCR_BASE_URL", "http://ppocr.test:8280")
    monkeypatch.setenv("HUNYUAN_OCR_BASE_URL", "http://hunyuan.test:4406/v1")
    assert load_config().get_provider().base_url == "http://ppocr.test:8280"
    assert (
        load_multimodal_config().get_provider().base_url
        == "http://hunyuan.test:4406/v1"
    )
