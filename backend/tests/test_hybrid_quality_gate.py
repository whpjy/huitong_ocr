from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.image_quality import ImageQualityIssue, ImageQualityReport
from api.service import DocumentQualityError, ExtractionService, QualityImageContext
from api import service as service_module
from llm_manager import hybrid_pipeline
from llm_manager.hybrid_pipeline import collect_ppocr_document, run_hybrid_document
from ocr_manager.models import ProviderResult


class FakeProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = 0
        self.config = SimpleNamespace(concurrency=1)

    def recognize(self, _image: Path) -> ProviderResult:
        self.events.append("ppocr")
        self.calls += 1
        return ProviderResult(
            text="姓名 张三",
            page_texts=("姓名 张三",),
            visualizations=(),
            tokens=({"text": "张三", "score": 0.99, "box": [1, 1, 9, 9]},),
        )


class FakeClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def extract_image_records(self, *_args, **_kwargs):
        self.events.append("hunyuan")
        return [{"姓名": "张三"}]


def _stub_fusion(**kwargs):
    result = kwargs["result"]
    return {
        "files": kwargs["ocr_files"],
        "primary_execution": {
            "parallel": kwargs["primary_parallel"],
            "multimodal_seconds": result.elapsed,
            "ppocr_seconds": kwargs["ppocr_elapsed"],
            "parallel_wall_seconds": kwargs["parallel_elapsed"],
        },
    }


def test_quality_gate_ppocr_result_is_reused_before_hunyuan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "prepared.jpg"
    image.write_bytes(b"prepared")
    events: list[str] = []
    provider = FakeProvider(events)
    precomputed = collect_ppocr_document(images=[image], ocr_provider=provider)
    monkeypatch.setattr(hybrid_pipeline, "fuse_hybrid_result", _stub_fusion)

    _result, process = run_hybrid_document(
        application_no="one",
        images=[image],
        profile=SimpleNamespace(key="id_card", field_names=("姓名",)),
        client=FakeClient(events),
        ocr_provider=provider,
        artifact_root=tmp_path,
        precomputed_ppocr=precomputed,
    )

    assert events == ["ppocr", "hunyuan"]
    assert provider.calls == 1
    assert process["files"] is precomputed[0]
    assert process["primary_execution"]["parallel"] is False
    assert process["primary_execution"]["cutoff_policy"] == "quality_gate_then_hunyuan"
    assert process["primary_execution"]["ppocr_status"] == "reused_quality_gate"


def test_default_hybrid_path_remains_parallel(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "prepared.jpg"
    image.write_bytes(b"prepared")
    events: list[str] = []
    provider = FakeProvider(events)
    monkeypatch.setattr(hybrid_pipeline, "fuse_hybrid_result", _stub_fusion)

    _result, process = run_hybrid_document(
        application_no="one",
        images=[image],
        profile=SimpleNamespace(key="id_card", field_names=("姓名",)),
        client=FakeClient(events),
        ocr_provider=provider,
        artifact_root=tmp_path,
        prefer_hunyuan_latency=False,
    )

    assert provider.calls == 1
    assert sorted(events) == ["hunyuan", "ppocr"]
    assert process["primary_execution"]["parallel"] is True
    assert process["primary_execution"]["cutoff_policy"] == "wait_for_both"


def test_quality_gate_runs_in_parallel_with_hunyuan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "prepared.jpg"
    image.write_bytes(b"prepared")
    events: list[str] = []
    hunyuan_started = threading.Event()
    ppocr_started = threading.Event()

    class ConcurrentProvider(FakeProvider):
        def recognize(self, image_path: Path) -> ProviderResult:
            events.append("ppocr_started")
            ppocr_started.set()
            assert hunyuan_started.wait(timeout=2)
            return super().recognize(image_path)

    class ConcurrentClient(FakeClient):
        def extract_image_records(self, *_args, **_kwargs):
            events.append("hunyuan_started")
            hunyuan_started.set()
            assert ppocr_started.wait(timeout=2)
            return super().extract_image_records(*_args, **_kwargs)

    provider = ConcurrentProvider(events)
    monkeypatch.setattr(hybrid_pipeline, "fuse_hybrid_result", _stub_fusion)

    def quality_gate(_outcome) -> None:
        events.append("quality")

    _result, process = run_hybrid_document(
        application_no="one",
        images=[image],
        profile=SimpleNamespace(key="id_card", field_names=("姓名",)),
        client=ConcurrentClient(events),
        ocr_provider=provider,
        artifact_root=tmp_path,
        prefer_hunyuan_latency=True,
        ppocr_result_callback=quality_gate,
    )

    assert "quality" in events
    assert process["primary_execution"]["parallel"] is True
    assert process["primary_execution"]["cutoff_policy"] == (
        "quality_gate_parallel_hunyuan"
    )
    assert process["primary_execution"]["ppocr_status"] == "quality_gate_passed"


def test_rejection_checks_prepared_image_after_hunyuan_starts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.jpg"
    prepared = tmp_path / "prepared.jpg"
    source.write_bytes(b"raw-image")
    prepared.write_bytes(b"prepared-image")
    events: list[str] = []
    service = object.__new__(ExtractionService)
    service.profiles = SimpleNamespace(
        get=lambda _key: SimpleNamespace(key="id_card", field_names=("姓名",))
    )
    service.resolve_document = lambda _value: ("id_card", "身份证")
    service._multimodal_client = lambda _name: (
        SimpleNamespace(config=SimpleNamespace(name="hunyuan", model="model")),
        SimpleNamespace(document_crop=SimpleNamespace()),
    )
    service._ppocr_resources = lambda _key: (SimpleNamespace(), None)
    monkeypatch.setattr(
        service_module,
        "_prepare_document_images",
        lambda _paths, _config: [prepared],
    )

    def fake_collect(**_kwargs):
        events.append("ppocr")
        return ([{"input_path": str(prepared), "text": "姓名", "tokens": []}], 0.1)

    monkeypatch.setattr(service_module, "collect_ppocr_document", fake_collect)
    def fake_hybrid(**kwargs):
        events.append("hunyuan")
        kwargs["ppocr_result_callback"](fake_collect()[0:2])
        pytest.fail("rejected quality gate returned normally")

    monkeypatch.setattr(service_module, "run_hybrid_document", fake_hybrid)

    def reject(content: bytes, _document_type: str) -> ImageQualityReport:
        events.append("quality")
        assert content == b"prepared-image"
        return ImageQualityReport(
            issues=(ImageQualityIssue("TOO_BLURRY", "图片模糊"),),
            metrics={},
        )

    with pytest.raises(DocumentQualityError) as captured:
        service.recognize_document(
            source,
            "id_card",
            "hybrid",
            "hunyuan_ocr",
            quality_checker=reject,
            quality_contexts=[QualityImageContext("document", "证件图片", "source.jpg")],
        )

    assert captured.value.code == "IMAGE_QUALITY_REJECTED"
    assert events == ["hunyuan", "ppocr", "quality"]
