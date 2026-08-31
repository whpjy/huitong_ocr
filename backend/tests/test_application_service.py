from __future__ import annotations

from pathlib import Path

import pytest

from api.application_config import ApplicationRecognitionConfig
from api.application_service import (
    ApplicationNotFoundError,
    ApplicationRecognitionService,
)
from api.service import DocumentRecognitionExecution


class FakeExtractionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def recognize_document(
        self,
        source_path: Path | list[Path],
        document_type: str,
        engine_type: str,
        provider: str,
    ) -> DocumentRecognitionExecution:
        paths = source_path if isinstance(source_path, list) else [source_path]
        names = [path.name for path in paths]
        self.calls.append((document_type, names))
        paths[0].with_name(
            f"{paths[0].stem}.page-001.oriented.jpg"
        ).write_bytes(b"generated-artifact")
        if document_type == "id_card":
            second_person = any("200" in name for name in names)
            fields = {
                "姓名": "李四" if second_person else "张三",
                "身份证号": "110101199202023456" if second_person else "110101199001011237",
                "住址": "北京市东城区二号" if second_person else "北京市东城区一号",
            }
            name = "身份证"
        elif document_type == "driver_license":
            fields = {
                "姓名": "张三",
                "证号": "110101199001011237",
                "住址": "上海市浦东新区九号",
            }
            name = "驾驶证"
        else:
            fields = {
                "所有人": "张三",
                "号牌号码": "京A12345",
                "车辆识别代号": "TESTVIN1234567890",
            }
            name = "行驶证"
        return DocumentRecognitionExecution(
            document_type=document_type,
            document_name=name,
            pipeline_type=engine_type,
            provider=provider,
            model="fake-model",
            text="",
            fields=fields,
            ocr_elapsed=0.1,
            extraction_elapsed=0.2,
            elapsed=0.3,
            records=[fields],
        )


def _create_application(root: Path, application_no: str = "1108011200") -> Path:
    application = root / application_no
    for code in ("DG12", "DG13", "DG14", "Z002"):
        (application / code).mkdir(parents=True)
    files = {
        "DG12/20250101_100_front.jpg": b"front-1",
        "DG13/20250101_100_back.jpg": b"back-1",
        "DG12/20250101_200_front.jpg": b"front-2",
        "DG13/20250101_200_back.jpg": b"back-2",
        "DG14/driver.jpg": b"driver",
        "Z002/vehicle.jpg": b"vehicle",
        "Z002/vehicle-copy.jpg": b"vehicle",
        "Z002/ignore.txt": b"not a document",
    }
    for relative_path, content in files.items():
        (application / relative_path).write_bytes(content)
    (application / "Z002" / "node_modules").mkdir()
    (application / "Z002" / "node_modules" / "bad.jpg").write_bytes(b"nested")
    return application


def _service(tmp_path: Path) -> tuple[ApplicationRecognitionService, FakeExtractionService]:
    _create_application(tmp_path)
    extraction = FakeExtractionService()
    config = ApplicationRecognitionConfig(
        data_root=tmp_path.resolve(),
        material_codes={
            "id_card_front": "DG12",
            "id_card_back": "DG13",
            "driver_license": "DG14",
            "vehicle_license": "Z002",
        },
        max_workers=2,
    )
    return ApplicationRecognitionService(extraction, config), extraction


def test_scan_only_direct_supported_files_and_deduplicates(tmp_path: Path) -> None:
    service, _extraction = _service(tmp_path)

    scanned = service.scan("1108011200")

    assert len(scanned["materials"]["vehicle_license"]) == 1
    assert scanned["duplicates"]["vehicle_license"] == 1


def test_source_image_list_keeps_every_original_image_grouped_by_directory(
    tmp_path: Path,
) -> None:
    service, _extraction = _service(tmp_path)

    result = service.list_source_images("1108011200")

    assert [group["material_code"] for group in result["groups"]] == [
        "DG12",
        "DG13",
        "DG14",
        "Z002",
    ]
    vehicle_group = next(
        group for group in result["groups"] if group["material_code"] == "Z002"
    )
    assert [file["name"] for file in vehicle_group["files"]] == [
        "vehicle-copy.jpg",
        "vehicle.jpg",
    ]
    assert all(file["material_code"] == "Z002" for file in vehicle_group["files"])


def test_recognize_application_links_people_and_keeps_address_warning(
    tmp_path: Path,
) -> None:
    service, extraction = _service(tmp_path)

    result = service.recognize("1108011200", "hybrid", "hunyuan_ocr")

    assert result["status"] == "completed"
    assert result["summary"]["id_card_count"] == 2
    assert result["summary"]["driver_license_count"] == 1
    assert result["summary"]["vehicle_license_count"] == 1
    assert result["summary"]["person_count"] == 2
    assert result["summary"]["duplicate_file_count"] == 1
    first_id = next(
        item
        for item in result["documents"]["id_cards"]
        if item["fields"]["姓名"] == "张三"
    )
    driver = result["documents"]["driver_licenses"][0]
    assert {item["material_code"] for item in first_id["source_file_refs"]} == {
        "DG12",
        "DG13",
    }
    assert driver["source_file_refs"] == [
        {"name": "driver.jpg", "material_code": "DG14"}
    ]
    assert first_id["person_id"] == driver["person_id"]
    assert any(
        item["field"] == "住址" and item["status"] == "different"
        for item in result["validations"]
    )
    id_calls = [call for call in extraction.calls if call[0] == "id_card"]
    assert len(id_calls) == 2
    assert all(len(names) == 2 for _document_type, names in id_calls)
    assert not list(tmp_path.rglob("*.page-001.oriented.jpg"))


def test_application_number_is_validated_and_must_exist(tmp_path: Path) -> None:
    service, _extraction = _service(tmp_path)

    with pytest.raises(ValueError, match="只能包含数字"):
        service.scan("../secret")
    with pytest.raises(ApplicationNotFoundError):
        service.scan("9999999999")


def test_source_file_is_resolved_only_inside_configured_materials(tmp_path: Path) -> None:
    service, _extraction = _service(tmp_path)

    source = service.source_file("1108011200", "DG14", "driver.jpg")

    assert source.read_bytes() == b"driver"
    with pytest.raises(ValueError, match="材料目录不受支持"):
        service.source_file("1108011200", "OTHER", "driver.jpg")
    with pytest.raises(ValueError, match="文件名非法"):
        service.source_file("1108011200", "DG14", "../driver.jpg")
    with pytest.raises(ApplicationNotFoundError, match="原文件不存在"):
        service.source_file("1108011200", "DG14", "missing.jpg")
