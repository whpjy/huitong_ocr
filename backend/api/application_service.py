"""Application-number recognition orchestration for the three supported documents."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ocr_manager.providers.base import IMAGE_EXTENSIONS

from .application_config import (
    ApplicationRecognitionConfig,
    load_application_config,
)
from .service import DocumentRecognitionExecution, ExtractionService


SUPPORTED_FILE_SUFFIXES = {*IMAGE_EXTENSIONS, ".pdf"}
APPLICATION_NUMBER_PATTERN = re.compile(r"^[0-9]{1,32}$")
MATERIAL_LABELS = {
    "id_card_front": "身份证正面",
    "id_card_back": "身份证反面",
    "driver_license": "驾驶证",
    "vehicle_license": "行驶证",
}


class ApplicationNotFoundError(FileNotFoundError):
    """The requested application directory does not exist."""


@dataclass(frozen=True)
class RecognitionWorkItem:
    document_type: str
    files: tuple[Path, ...]
    pairing_confidence: str = ""


@dataclass
class ApplicationDocument:
    document_type: str
    document_name: str
    instance_id: str
    source_files: list[str]
    source_file_refs: list[dict[str, str]]
    fields: dict[str, str]
    pipeline_type: str
    provider: str
    model: str
    elapsed_seconds: float
    pairing_confidence: str = ""
    person_id: str = ""
    vehicle_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "document_name": self.document_name,
            "instance_id": self.instance_id,
            "source_files": self.source_files,
            "source_file_refs": self.source_file_refs,
            "fields": self.fields,
            "pipeline_type": self.pipeline_type,
            "provider": self.provider,
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "pairing_confidence": self.pairing_confidence or None,
            "person_id": self.person_id or None,
            "vehicle_id": self.vehicle_id or None,
            "warnings": self.warnings,
        }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_files(paths: list[Path]) -> tuple[list[Path], int]:
    unique: list[Path] = []
    digests: set[str] = set()
    for path in sorted(paths, key=lambda item: item.name.lower()):
        digest = _file_digest(path)
        if digest in digests:
            continue
        digests.add(digest)
        unique.append(path)
    return unique, len(paths) - len(unique)


def _batch_token(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return parts[1]
    return ""


def _pair_id_card_files(
    fronts: list[Path],
    backs: list[Path],
) -> list[RecognitionWorkItem]:
    """Pair ID-card sides without ever merging two front sides together."""

    remaining_fronts = list(fronts)
    remaining_backs = list(backs)
    pairs: list[RecognitionWorkItem] = []

    back_by_token: dict[str, list[Path]] = {}
    for path in remaining_backs:
        token = _batch_token(path)
        if token:
            back_by_token.setdefault(token, []).append(path)
    for front in list(remaining_fronts):
        token = _batch_token(front)
        candidates = back_by_token.get(token, []) if token else []
        if not candidates:
            continue
        back = candidates.pop(0)
        pairs.append(RecognitionWorkItem("id_card", (front, back), "high"))
        remaining_fronts.remove(front)
        remaining_backs.remove(back)

    while remaining_fronts and remaining_backs:
        pairs.append(
            RecognitionWorkItem(
                "id_card",
                (remaining_fronts.pop(0), remaining_backs.pop(0)),
                "medium",
            )
        )
    pairs.extend(
        RecognitionWorkItem("id_card", (path,), "unpaired_front")
        for path in remaining_fronts
    )
    pairs.extend(
        RecognitionWorkItem("id_card", (path,), "unpaired_back")
        for path in remaining_backs
    )
    return pairs


def _clean_key(value: str) -> str:
    return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", value.upper())


def _record_key(document_type: str, fields: dict[str, str]) -> str:
    candidates = {
        "id_card": ("身份证号",),
        "driver_license": ("证号",),
        "vehicle_license": ("车辆识别代号", "车牌号码", "号牌号码"),
    }[document_type]
    return next(
        (_clean_key(fields.get(name, "")) for name in candidates if fields.get(name)),
        "",
    )


def _merge_fields(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    merged = dict(left)
    for name, value in right.items():
        current = merged.get(name, "")
        if not current or (value and len(value) > len(current)):
            merged[name] = value
    return merged


def _merge_documents(documents: list[ApplicationDocument]) -> list[ApplicationDocument]:
    merged: list[ApplicationDocument] = []
    by_key: dict[tuple[str, str], ApplicationDocument] = {}
    for document in documents:
        key = _record_key(document.document_type, document.fields)
        if not key:
            merged.append(document)
            continue
        lookup = (document.document_type, key)
        existing = by_key.get(lookup)
        if existing is None:
            by_key[lookup] = document
            merged.append(document)
            continue
        existing.fields = _merge_fields(existing.fields, document.fields)
        existing.source_files.extend(
            name for name in document.source_files if name not in existing.source_files
        )
        existing_refs = {
            (item["material_code"], item["name"])
            for item in existing.source_file_refs
        }
        for item in document.source_file_refs:
            key = (item["material_code"], item["name"])
            if key not in existing_refs:
                existing.source_file_refs.append(item)
                existing_refs.add(key)
        existing.elapsed_seconds += document.elapsed_seconds
        existing.warnings.append("检测到同一证件的多份材料，结果已合并")
    for index, document in enumerate(merged, start=1):
        document.instance_id = f"{document.document_type}-{index}"
    return merged


def _normalized_address(value: str) -> str:
    return re.sub(r"[\s,，。;；、]", "", value)


def _address_validation(
    id_document: ApplicationDocument,
    driver_document: ApplicationDocument,
) -> dict[str, Any] | None:
    left = id_document.fields.get("住址", "")
    right = driver_document.fields.get("住址", "")
    if not left or not right:
        return None
    normalized_left = _normalized_address(left)
    normalized_right = _normalized_address(right)
    similarity = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    if normalized_left == normalized_right:
        status, severity, message = "consistent", "info", "身份证与驾驶证住址一致"
    elif similarity >= 0.72:
        status, severity, message = "similar", "info", "身份证与驾驶证住址基本一致"
    else:
        status, severity, message = (
            "different",
            "warning",
            "身份证与驾驶证住址不同，可能是证件换发时间不同，请人工核验",
        )
    return {
        "field": "住址",
        "left_instance_id": id_document.instance_id,
        "right_instance_id": driver_document.instance_id,
        "status": status,
        "severity": severity,
        "similarity": round(similarity, 4),
        "message": message,
    }


class ApplicationRecognitionService:
    """Scan one application directory, recognize documents, and link people."""

    def __init__(
        self,
        extraction_service: ExtractionService | None = None,
        config: ApplicationRecognitionConfig | None = None,
    ) -> None:
        self.extraction_service = extraction_service or ExtractionService()
        self.config = config or load_application_config()

    def application_path(self, application_no: str) -> Path:
        if not APPLICATION_NUMBER_PATTERN.fullmatch(application_no):
            raise ValueError("申请单号只能包含数字，长度不能超过 32 位")
        path = (self.config.data_root / application_no).resolve()
        if path.parent != self.config.data_root:
            raise ValueError("申请单号路径非法")
        if not path.is_dir():
            raise ApplicationNotFoundError(f"申请单号不存在：{application_no}")
        return path

    def source_file(
        self,
        application_no: str,
        material_code: str,
        file_name: str,
    ) -> Path:
        """Resolve one configured source file without allowing directory traversal."""

        application_path = self.application_path(application_no)
        allowed_codes = set(self.config.material_codes.values())
        if material_code not in allowed_codes:
            raise ValueError("材料目录不受支持")
        if not file_name or Path(file_name).name != file_name:
            raise ValueError("原文件名非法")
        path = (application_path / material_code / file_name).resolve()
        material_path = (application_path / material_code).resolve()
        if path.parent != material_path:
            raise ValueError("原文件路径非法")
        if path.suffix.lower() not in SUPPORTED_FILE_SUFFIXES or not path.is_file():
            raise ApplicationNotFoundError("原文件不存在")
        return path

    def list_source_images(self, application_no: str) -> dict[str, Any]:
        """List every original image in each configured material directory."""

        application_path = self.application_path(application_no)
        groups: list[dict[str, Any]] = []
        for material, label in MATERIAL_LABELS.items():
            code = self.config.material_codes[material]
            images = [
                path
                for path in self._material_files(application_path, material)
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            groups.append({
                "material": material,
                "label": label,
                "material_code": code,
                "files": [
                    {"name": path.name, "material_code": code}
                    for path in sorted(images, key=lambda item: item.name.lower())
                ],
            })
        return {"application_no": application_no, "groups": groups}

    def source_thumbnail(
        self,
        application_no: str,
        material_code: str,
        file_name: str,
    ) -> bytes:
        """Render a metadata-free 4:3 thumbnail containing the complete image."""

        path = self.source_file(application_no, material_code, file_name)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (640, 480), "#eef2f7")
            left = (canvas.width - image.width) // 2
            top = (canvas.height - image.height) // 2
            canvas.paste(image, (left, top))
            output = io.BytesIO()
            canvas.save(output, format="JPEG", quality=86, optimize=True)
            return output.getvalue()

    def _material_files(self, application_path: Path, material: str) -> list[Path]:
        code = self.config.material_codes[material]
        directory = application_path / code
        if not directory.is_dir():
            return []
        files: list[Path] = []
        for path in directory.iterdir():
            lower_name = path.name.lower()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
                continue
            if re.search(r"\.page-\d+\.(?:oriented|document-crop)\.", lower_name):
                continue
            files.append(path)
        return files

    def scan(self, application_no: str) -> dict[str, Any]:
        path = self.application_path(application_no)
        materials: dict[str, list[Path]] = {}
        duplicates: dict[str, int] = {}
        for material in self.config.material_codes:
            unique, duplicate_count = _deduplicate_files(
                self._material_files(path, material)
            )
            materials[material] = unique
            duplicates[material] = duplicate_count
        return {
            "path": path,
            "materials": materials,
            "duplicates": duplicates,
        }

    def _work_items(self, materials: dict[str, list[Path]]) -> list[RecognitionWorkItem]:
        items = _pair_id_card_files(
            materials["id_card_front"],
            materials["id_card_back"],
        )
        items.extend(
            RecognitionWorkItem("driver_license", (path,))
            for path in materials["driver_license"]
        )
        items.extend(
            RecognitionWorkItem("vehicle_license", (path,))
            for path in materials["vehicle_license"]
        )
        return items

    def _recognize_item(
        self,
        item: RecognitionWorkItem,
        engine_type: str,
        provider: str,
    ) -> tuple[RecognitionWorkItem, DocumentRecognitionExecution]:
        # Existing single-document preprocessing writes artifacts beside its
        # input. Work on temporary copies so application source data remains
        # immutable and repeated scans never ingest generated images.
        with tempfile.TemporaryDirectory(prefix="application-recognition-") as directory:
            working_paths: list[Path] = []
            for index, path in enumerate(item.files, start=1):
                parent = Path(directory) / f"source-{index:02d}"
                parent.mkdir(parents=True, exist_ok=True)
                working_path = parent / path.name
                shutil.copy2(path, working_path)
                working_paths.append(working_path)
            source: Path | list[Path] = working_paths[0]
            if len(working_paths) > 1:
                source = working_paths
            result = self.extraction_service.recognize_document(
                source,
                item.document_type,
                engine_type,
                provider,
            )
            return item, result

    def recognize(
        self,
        application_no: str,
        engine_type: str,
        provider: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        scanned = self.scan(application_no)
        items = self._work_items(scanned["materials"])
        documents: list[ApplicationDocument] = []
        errors: list[dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=min(self.config.max_workers, max(1, len(items))),
            thread_name_prefix="application-recognition",
        ) as executor:
            futures = {
                executor.submit(
                    self._recognize_item,
                    item,
                    engine_type,
                    provider,
                ): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    _item, result = future.result()
                except Exception as exc:  # Keep partial application results.
                    errors.append({
                        "document_type": item.document_type,
                        "source_files": [path.name for path in item.files],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                records = result.records or [result.fields]
                for record in records:
                    if not any(record.values()):
                        continue
                    documents.append(ApplicationDocument(
                        document_type=result.document_type,
                        document_name=result.document_name,
                        instance_id="",
                        source_files=[path.name for path in item.files],
                        source_file_refs=[
                            {
                                "name": path.name,
                                "material_code": path.parent.name,
                            }
                            for path in item.files
                        ],
                        fields=dict(record),
                        pipeline_type=result.pipeline_type,
                        provider=result.provider,
                        model=result.model,
                        elapsed_seconds=result.elapsed,
                        pairing_confidence=item.pairing_confidence,
                    ))

        documents = _merge_documents(documents)
        ids = [item for item in documents if item.document_type == "id_card"]
        drivers = [item for item in documents if item.document_type == "driver_license"]
        vehicles = [item for item in documents if item.document_type == "vehicle_license"]

        person_counter = 0
        people_by_number: dict[str, str] = {}
        for document in ids:
            person_counter += 1
            document.person_id = f"person-{person_counter}"
            number = _clean_key(document.fields.get("身份证号", ""))
            if number:
                people_by_number[number] = document.person_id
        for document in drivers:
            number = _clean_key(document.fields.get("证号", ""))
            person_id = people_by_number.get(number)
            if not person_id:
                person_counter += 1
                person_id = f"person-{person_counter}"
                if number:
                    people_by_number[number] = person_id
            document.person_id = person_id
        for index, document in enumerate(vehicles, start=1):
            document.vehicle_id = f"vehicle-{index}"

        validations: list[dict[str, Any]] = []
        ids_by_person = {item.person_id: item for item in ids}
        for driver in drivers:
            identity = ids_by_person.get(driver.person_id)
            if identity is None:
                continue
            validations.append({
                "field": "身份证号/驾驶证号",
                "left_instance_id": identity.instance_id,
                "right_instance_id": driver.instance_id,
                "status": "consistent",
                "severity": "info",
                "similarity": 1.0,
                "message": "身份证号与驾驶证号一致，已关联为同一人",
            })
            address_check = _address_validation(identity, driver)
            if address_check:
                validations.append(address_check)

        grouped = {
            "id_cards": [item.to_dict() for item in ids],
            "driver_licenses": [item.to_dict() for item in drivers],
            "vehicle_licenses": [item.to_dict() for item in vehicles],
        }
        missing = [
            label
            for key, label in (
                ("id_cards", "身份证"),
                ("driver_licenses", "驾驶证"),
                ("vehicle_licenses", "行驶证"),
            )
            if not grouped[key]
        ]
        return {
            "application_no": application_no,
            "status": "completed" if not errors else "partial",
            "documents": grouped,
            "validations": validations,
            "errors": errors,
            "summary": {
                "id_card_count": len(ids),
                "driver_license_count": len(drivers),
                "vehicle_license_count": len(vehicles),
                "person_count": person_counter,
                "duplicate_file_count": sum(scanned["duplicates"].values()),
                "error_count": len(errors),
                "missing_documents": missing,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            },
        }
