"""Shared HunyuanOCR + PP-OCR document pipeline for batch and API use."""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ocr_manager.providers.base import OCRProvider

from .client import OpenAICompatibleClient
from .driver_refiner import detect_driver_page_type
from .hybrid_validation import (
    arbitrate_fast,
    compact,
    compare_fields_fast,
    missing_field_supplements,
)
from .models import ExtractionResult
from .profiles import ExtractionProfile


OCRArtifactCallback = Callable[[Path, dict[str, Any]], None]
PPOCRResultCallback = Callable[[tuple[list[dict[str, Any]], float]], None]
LOGGER = logging.getLogger("huitong.api")

# Mobile hybrid requests use Hunyuan's completion as a hard latency cutoff.
# Keep PP work outside a per-request executor context, whose shutdown would
# otherwise wait for a slow or unavailable PP service before returning.
_MOBILE_PPOCR_WORKERS = 16
_MOBILE_PPOCR_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MOBILE_PPOCR_WORKERS,
    thread_name_prefix="hybrid-mobile-ppocr",
)
_MOBILE_PPOCR_ADMISSION = threading.BoundedSemaphore(_MOBILE_PPOCR_WORKERS)

_CHINA_NATIONALITY_VALUES = {"中国", "中国CHN", "CHN", "中华人民共和国"}
_CHINA_PROVINCE_NAMES = (
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "香港",
    "澳门", "台湾",
)


def _looks_like_domestic_address(value: Any) -> bool:
    address = compact(value)
    if any(address.startswith(name) for name in _CHINA_PROVINCE_NAMES):
        return True
    # Some licences omit the province and start directly with a prefecture or
    # county-level city, for example "深圳市南山区" or "中山市古镇镇".
    return bool(
        re.match(r"^[\u3400-\u9fff]{2,12}市", address)
        and any(marker in address for marker in ("区", "县", "镇", "乡", "街道", "村"))
    )


def _apply_driver_nationality_rule(
    *,
    result: ExtractionResult,
    records: list[dict[str, str]],
    process: dict[str, Any],
    ocr_files: list[dict[str, Any]],
) -> None:
    """Normalize or infer Chinese nationality from either model's evidence."""

    initial_text = "\n".join(
        str(value)
        for record in process.get("initial_records") or []
        for value in record.values()
        if value
    )
    ppocr_text = "\n".join(str(item.get("text") or "") for item in ocr_files)
    model_has_china = any(
        marker in text
        for text in (initial_text, ppocr_text)
        for marker in ("中国", "中华人民共和国")
    )
    for index, record in enumerate(records):
        current = str(record.get("国籍") or "")
        normalized = compact(current)
        source = ""
        if normalized in _CHINA_NATIONALITY_VALUES:
            source = "nationality_value_normalization"
        elif not normalized and model_has_china:
            source = "model_text_china_evidence"
        elif not normalized and _looks_like_domestic_address(record.get("住址")):
            source = "domestic_address_rule"
        else:
            continue
        if current == "中国":
            continue
        record["国籍"] = "中国"
        if result.records:
            result.records[index]["国籍"] = "中国"
        process["final_records"][index]["国籍"] = "中国"
        if index == 0:
            result.fields["国籍"] = "中国"
            process["final_fields"]["国籍"] = "中国"
        process["supplements"].append(
            {"field": "国籍", "value": "中国", "record_index": index, "source": source}
        )
        process["corrected_fields"].append(
            {"record_index": index, "field": "国籍", "source": "driver_nationality_rule"}
        )


def _record_image_evidence(
    profile_key: str,
    records: list[dict[str, str]],
    record_index: int,
    image_evidence: list[tuple[Path, list[dict[str, Any]], dict[str, Any]]],
) -> list[tuple[Path, list[dict[str, Any]], dict[str, Any]]]:
    """Keep PP evidence from another subject out of the current record."""

    if len(records) <= 1:
        return image_evidence
    key_fields = {
        "vehicle_license": ("车辆识别代号", "号牌号码", "车牌号码"),
        "driver_license": ("证号",),
        "id_card": ("身份证号",),
    }.get(profile_key, ())
    identifiers = [
        compact(records[record_index].get(field, ""))
        for field in key_fields
        if compact(records[record_index].get(field, ""))
    ]
    matched = []
    for evidence in image_evidence:
        stream = "".join(compact(token.get("text")) for token in evidence[1])
        if any(identifier in stream for identifier in identifiers):
            matched.append(evidence)
    if matched:
        return matched
    if len(records) == len(image_evidence) and record_index < len(image_evidence):
        return [image_evidence[record_index]]
    # Ambiguous multi-record evidence must not be shared across subjects.
    return []


def run_parallel_primary(
    multimodal_call: Callable[[], Any],
    ppocr_call: Callable[[], Any],
) -> tuple[Any, Any]:
    """Run the two independent primary-model branches concurrently."""

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-primary") as executor:
        multimodal_future = executor.submit(multimodal_call)
        ppocr_future = executor.submit(ppocr_call)
        return multimodal_future.result(), ppocr_future.result()


def run_hunyuan_cutoff_primary(
    multimodal_call: Callable[[], Any],
    ppocr_call: Callable[[], Any],
    *,
    ppocr_admission: threading.BoundedSemaphore = _MOBILE_PPOCR_ADMISSION,
) -> tuple[Any, Any | None, str]:
    """Return at Hunyuan completion and consume PP only when already ready."""

    # ThreadPoolExecutor uses an unbounded internal queue. Reserve execution
    # capacity before submission so PP failures can never create stale queued
    # work. A rejected PP branch is discarded before Hunyuan starts.
    if not ppocr_admission.acquire(blocking=False):
        LOGGER.info("mobile ppocr discarded reason=no_execution_capacity")
        return multimodal_call(), None, "discarded_no_execution_capacity"
    try:
        ppocr_future = _MOBILE_PPOCR_EXECUTOR.submit(ppocr_call)
    except Exception:
        ppocr_admission.release()
        raise
    LOGGER.info("mobile ppocr submitted")

    def release_and_report(future: Any) -> None:
        ppocr_admission.release()
        try:
            outcome = future.result()
            files = outcome[0] if isinstance(outcome, tuple) and outcome else None
            if isinstance(files, list):
                errors = [str(item.get("error") or "") for item in files]
                available = any(not error for error in errors)
                LOGGER.info(
                    "mobile ppocr finished available=%s errors=%s",
                    available,
                    " | ".join(error for error in errors if error) or "none",
                )
            else:
                LOGGER.info("mobile ppocr finished")
        except Exception as exc:  # noqa: BLE001 - report detached PP failure
            LOGGER.info(
                "mobile ppocr finished available=False error=%s: %s",
                type(exc).__name__,
                exc,
            )

    ppocr_future.add_done_callback(release_and_report)
    multimodal_outcome = multimodal_call()
    if ppocr_future.done():
        try:
            return multimodal_outcome, ppocr_future.result(), "completed_before_hunyuan"
        except Exception:  # noqa: BLE001 - PP failure must not fail Hunyuan
            return multimodal_outcome, None, "failed_before_hunyuan"

    # Cancel queued work where possible. Running provider calls cannot be
    # force-cancelled safely, so let them finish in the shared background pool.
    if not ppocr_future.cancel():
        def consume_background_result(future: Any) -> None:
            try:
                future.result()
            except Exception:  # noqa: BLE001 - deliberately detached fallback
                pass

        ppocr_future.add_done_callback(consume_background_result)
    return multimodal_outcome, None, "slower_than_hunyuan"


def run_ppocr_with_slot(
    ppocr_call: Callable[[], Any],
    semaphore: threading.BoundedSemaphore | None,
    *,
    wait_for_slot: bool,
) -> Any:
    """Run PP once without allowing mobile fallback work to form a backlog."""

    if semaphore is None:
        return ppocr_call()
    acquired = semaphore.acquire(blocking=wait_for_slot)
    if not acquired:
        raise RuntimeError("PP-OCR concurrency slot unavailable")
    try:
        return ppocr_call()
    finally:
        semaphore.release()


def collect_ppocr_document(
    *,
    images: list[Path],
    ocr_provider: OCRProvider,
    ppocr_semaphore: threading.BoundedSemaphore | None = None,
    ocr_artifact_callback: OCRArtifactCallback | None = None,
    wait_for_slot: bool = True,
) -> tuple[list[dict[str, Any]], float]:
    """Run PP-OCR once and return reusable page artifacts."""

    started = time.perf_counter()

    def recognize_page(indexed_path: tuple[int, Path]) -> dict[str, Any]:
        image_index, image_path = indexed_path
        try:
            ocr_result = run_ppocr_with_slot(
                lambda: ocr_provider.recognize(image_path),
                ppocr_semaphore,
                wait_for_slot=wait_for_slot,
            )
            item = {
                "image_index": image_index,
                "input_path": str(image_path),
                "text": ocr_result.text,
                "tokens": list(ocr_result.tokens),
            }
            if ocr_artifact_callback is not None:
                ocr_artifact_callback(image_path, item)
            return item
        except Exception as exc:  # noqa: BLE001
            return {
                "image_index": image_index,
                "input_path": str(image_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    indexed_images = list(enumerate(images))
    if len(indexed_images) <= 1:
        files = [recognize_page(item) for item in indexed_images]
    else:
        with ThreadPoolExecutor(
            max_workers=min(
                len(indexed_images),
                max(1, ocr_provider.config.concurrency),
            ),
            thread_name_prefix="hybrid-ppocr-pages",
        ) as executor:
            files = list(executor.map(recognize_page, indexed_images))
    return files, time.perf_counter() - started


def run_hybrid_document(
    *,
    application_no: str,
    images: list[Path],
    profile: ExtractionProfile,
    client: OpenAICompatibleClient,
    ocr_provider: OCRProvider,
    artifact_root: Path,
    hunyuan_prompt_key: str | None = None,
    ppocr_semaphore: threading.BoundedSemaphore | None = None,
    ocr_artifact_callback: OCRArtifactCallback | None = None,
    prefer_hunyuan_latency: bool = False,
    precomputed_ppocr: tuple[list[dict[str, Any]], float] | None = None,
    ppocr_result_callback: PPOCRResultCallback | None = None,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Recognize one document with both models, then fuse their evidence."""

    def extract_multimodal() -> tuple[
        list[dict[str, str]] | None, float, Exception | None
    ]:
        started = time.perf_counter()
        try:
            records = client.extract_image_records(
                profile,
                images,
                hunyuan_prompt_key=hunyuan_prompt_key,
            )
            return records, time.perf_counter() - started, None
        except Exception as exc:  # noqa: BLE001
            return None, time.perf_counter() - started, exc

    def collect_ppocr() -> tuple[list[dict[str, Any]], float]:
        outcome = collect_ppocr_document(
            images=images,
            ocr_provider=ocr_provider,
            ppocr_semaphore=ppocr_semaphore,
            ocr_artifact_callback=ocr_artifact_callback,
            # A mobile request that has already fallen back to Hunyuan must
            # never leave stale PP work queued on the shared semaphore.
            wait_for_slot=not prefer_hunyuan_latency,
        )
        if ppocr_result_callback is not None:
            ppocr_result_callback(outcome)
        return outcome

    parallel_started = time.perf_counter()
    primary_parallel = precomputed_ppocr is None
    if precomputed_ppocr is not None:
        multimodal_outcome = extract_multimodal()
        ppocr_outcome = precomputed_ppocr
        ppocr_status = "reused_quality_gate"
    elif ppocr_result_callback is not None:
        multimodal_outcome, ppocr_outcome = run_parallel_primary(
            extract_multimodal,
            collect_ppocr,
        )
        ppocr_status = "quality_gate_passed"
    elif prefer_hunyuan_latency:
        (
            multimodal_outcome,
            ppocr_outcome,
            ppocr_status,
        ) = run_hunyuan_cutoff_primary(extract_multimodal, collect_ppocr)
    else:
        multimodal_outcome, ppocr_outcome = run_parallel_primary(
            extract_multimodal,
            collect_ppocr,
        )
        ppocr_status = "completed"
    parallel_elapsed = time.perf_counter() - parallel_started
    records, multimodal_elapsed, multimodal_error = multimodal_outcome
    if ppocr_outcome is None:
        ocr_files, ppocr_elapsed = [], 0.0
    else:
        ocr_files, ppocr_elapsed = ppocr_outcome
        if ocr_files and all(item.get("error") for item in ocr_files):
            ppocr_status = "failed_before_hunyuan" if prefer_hunyuan_latency else "failed"
    if multimodal_error is None and records is not None:
        result = ExtractionResult(
            application_no=application_no,
            success=True,
            status="success",
            fields=records[0],
            elapsed=multimodal_elapsed,
            records=records,
        )
    else:
        exc = multimodal_error or RuntimeError("多模态模型未返回抽取结果")
        result = ExtractionResult(
            application_no=application_no,
            success=False,
            status="failed",
            fields={field: "" for field in profile.field_names},
            elapsed=multimodal_elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
    fusion_started = time.perf_counter()
    process = fuse_hybrid_result(
        application_no=application_no,
        profile=profile,
        client=client,
        artifact_root=artifact_root,
        result=result,
        ocr_files=ocr_files,
        ppocr_elapsed=ppocr_elapsed,
        parallel_elapsed=parallel_elapsed,
        primary_parallel=primary_parallel,
    )
    fusion_elapsed = time.perf_counter() - fusion_started
    process["primary_execution"]["fusion_seconds"] = fusion_elapsed
    LOGGER.info(
        "hybrid fusion document_type=%s records=%s ppocr_files=%s elapsed=%.3fs",
        profile.key,
        len(result.records or [result.fields]),
        len(ocr_files),
        fusion_elapsed,
    )
    ppocr_available = bool(ocr_files) and any(
        not item.get("error") for item in ocr_files
    )
    process["primary_execution"].update({
        "cutoff_policy": (
            "quality_gate_then_hunyuan"
            if precomputed_ppocr is not None
            else "quality_gate_parallel_hunyuan"
            if ppocr_result_callback is not None
            else "hunyuan_completion"
            if prefer_hunyuan_latency
            else "wait_for_both"
        ),
        "ppocr_status": ppocr_status,
        "ppocr_available_for_fusion": ppocr_available,
        "hybrid_applied": ppocr_available,
    })
    if not result.success and any(result.fields.values()):
        result.success = True
        result.status = "success_with_ppocr_fallback"
    return result, process


def fuse_hybrid_result(
    *,
    application_no: str,
    profile: ExtractionProfile,
    client: OpenAICompatibleClient,
    artifact_root: Path,
    result: ExtractionResult,
    ocr_files: list[dict[str, Any]],
    ppocr_elapsed: float,
    parallel_elapsed: float,
    primary_parallel: bool = True,
) -> dict[str, Any]:
    """Apply the same supplementation and conflict rules for every caller."""

    process: dict[str, Any] = {
        "application_no": application_no,
        "initial_fields": dict(result.fields),
        "final_fields": dict(result.fields),
        "initial_records": [dict(item) for item in (result.records or [result.fields])],
        "final_records": [dict(item) for item in (result.records or [result.fields])],
        "files": ocr_files,
        "conflicts": [],
        "supplements": [],
        "corrected_fields": [],
        "validation_mode": "ppocr_fast_rules",
        "secondary_model_recheck": False,
        "primary_execution": {
            "parallel": primary_parallel,
            "multimodal_seconds": result.elapsed,
            "ppocr_seconds": ppocr_elapsed,
            "parallel_wall_seconds": parallel_elapsed,
        },
    }
    image_evidence = [
        (Path(str(item["input_path"])), item["tokens"], item)
        for item in ocr_files
        if isinstance(item.get("tokens"), list) and item.get("input_path")
    ]
    records = result.records or [result.fields]
    if profile.key == "driver_license":
        detected_types = []
        for item in ocr_files:
            detected_type = detect_driver_page_type(str(item.get("text") or ""))
            item["document_type"] = detected_type
            detected_types.append(detected_type)
        document_type = next(
            (value for value in reversed(detected_types) if value),
            "",
        )
        type_source = "ppocr_latest_image_rule"
        if not document_type:
            type_source = "initial_hunyuan_field"
            document_type = next(
                (
                    str(record.get("证件类型") or "")
                    for record in records
                    if record.get("证件类型")
                ),
                "",
            )
        for index, record in enumerate(records):
            record["证件类型"] = document_type
            process["final_records"][index]["证件类型"] = document_type
        result.fields["证件类型"] = document_type
        process["final_fields"]["证件类型"] = document_type
        if document_type:
            process["supplements"].append({
                "field": "证件类型", "value": document_type, "source": type_source
            })
    if len(records) == 1:
        for supplement in missing_field_supplements(profile, records[0], ocr_files):
            field, value = supplement["field"], supplement["value"]
            records[0][field] = value
            result.fields[field] = value
            process["final_records"][0][field] = value
            process["final_fields"][field] = value
            process["supplements"].append(supplement)
            process["corrected_fields"].append({
                "record_index": 0, "field": field, "source": "ppocr_missing_field"
            })
    record_items = [
        (index, field, value)
        for index, record in enumerate(result.records or [result.fields])
        for field, value in record.items()
    ]
    for record_index, field, initial_value in record_items:
        if field == "证件类型":
            continue
        if (
            field == "号牌号码"
            and compact(records[record_index].get("车牌号码", ""))
        ):
            continue
        candidates = []
        record_evidence = _record_image_evidence(
            profile.key,
            records,
            record_index,
            image_evidence,
        )
        for evidence_index, (image_path, tokens, _item) in enumerate(record_evidence):
            compared = compare_fields_fast({field: initial_value}, tokens)
            if compared and compared[0].get("status") not in {
                "ppocr_no_evidence",
                "ppocr_no_labeled_evidence",
            }:
                candidates.append((compared[0], image_path, evidence_index))
        if not candidates:
            continue
        comparison, evidence_image, image_index = max(
            candidates,
            key=lambda item: (
                item[0].get("status") == "consistent",
                float(item[0].get("similarity") or 0),
                float(item[0].get("ppocr_confidence") or 0),
            ),
        )
        if comparison.get("status") != "conflict":
            continue
        final_value, reason, changed = arbitrate_fast(
            comparison,
            records[record_index],
        )
        process["conflicts"].append({
            **comparison,
            "record_index": record_index,
            "image_index": image_index,
            "evidence_image": str(evidence_image),
            "crop_paths": [],
            "recheck": {
                "skipped": True,
                "reason": "fast_ppocr_confidence_rules",
            },
            "decision_mode": "ppocr_fast_rules",
            "final_value": final_value,
            "changed": changed,
            "decision_reason": reason,
        })
        if changed:
            if result.records:
                result.records[record_index][field] = final_value
            if record_index == 0:
                result.fields[field] = final_value
                process["final_fields"][field] = final_value
            process["final_records"][record_index][field] = final_value
            if field in {"号牌号码", "车牌号码"}:
                for alias in ("号牌号码", "车牌号码"):
                    if result.records:
                        result.records[record_index][alias] = final_value
                    records[record_index][alias] = final_value
                    process["final_records"][record_index][alias] = final_value
                    if record_index == 0:
                        result.fields[alias] = final_value
                        process["final_fields"][alias] = final_value
            process["corrected_fields"].append({
                "record_index": record_index,
                "field": field,
                "source": "ppocr_fast_rules",
            })
    if profile.key == "driver_license":
        _apply_driver_nationality_rule(
            result=result,
            records=records,
            process=process,
            ocr_files=ocr_files,
        )
    return process
