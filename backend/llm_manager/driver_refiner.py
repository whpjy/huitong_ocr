"""Conservative, evidence-based repairs for driver-license extraction."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from llm_manager.address_corrector import correct_address_divisions
from llm_manager.unmasked_preference import (
    contains_mask,
    prefer_normalized_driver_address,
    unique_unmasked,
)


_DATE = (
    r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*"
    r"(\d{1,2})(?:日)?(?!\d)"
)
_ARCHIVE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
_DRIVING_CLASS = re.compile(
    r"(?:A[123]|B[12]|C[1-6]|[DEFMNP])"
    r"(?:A[123]|B[12]|C[1-6]|[DEFMNP]){0,2}",
    re.IGNORECASE,
)
_CLASS_EXPLANATION_MARKERS = (
    "准驾车型代号规定",
    "大型客车",
    "重型牵引挂车",
    "小型汽车和",
)
_IDENTITY_NUMBER = re.compile(r"(?<![0-9A-Z])([1-9]\d{16}[0-9X])(?![0-9A-Z])", re.IGNORECASE)
_ADDRESS_THETA_ZERO = re.compile(
    r"(?<=[\u4e00-\u9fff0-9])[θΘ](?=\d{1,5}"
    r"(?:号|组|栋|幢|座|单元|楼|层|室|房|户|弄|巷))"
)
_PROVINCE_LEVEL_PREFIX = (
    r"(?:(?:北京|天津|上海|重庆)市|"
    r"(?:河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|"
    r"河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾)省|"
    r"(?:内蒙古|广西壮族|西藏|宁夏回族|新疆维吾尔)自治区|"
    r"(?:香港|澳门)特别行政区)"
)
_TRAILING_AUTHORITY_LOCATION = re.compile(
    rf"^(.+?(?:\d+|[一二三四五六七八九十百]+)"
    rf"(?:组|号|室|户))(?={_PROVINCE_LEVEL_PREFIX})"
)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _label_window(
    text: str,
    labels: tuple[str, ...],
    *,
    before: int = 0,
    after: int = 4,
) -> list[str]:
    lines = _lines(text)
    result: list[str] = []
    for index, line in enumerate(lines):
        if any(label.lower() in line.lower() for label in labels):
            result.extend(lines[max(0, index - before) : index + after + 1])
    return list(dict.fromkeys(result))


def _valid_date(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        return ""
    try:
        parsed = datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return ""
    return digits if 1900 <= parsed.year <= 2099 else ""


def _date_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for year, month, day in re.findall(_DATE, text):
        candidate = _valid_date(f"{year}{int(month):02d}{int(day):02d}")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _date_after_label(text: str, labels: tuple[str, ...]) -> str:
    window = "\n".join(_label_window(text, labels, after=3))
    candidates = _date_candidates(window)
    return candidates[0] if candidates else ""


def _date_range_from_ocr(text: str) -> str:
    window = "\n".join(
        _label_window(
            text,
            ("有效期限", "有效期", "Valid Period"),
            before=1,
            after=5,
        )
    )
    dates = _date_candidates(window)
    if dates and "长期" in window:
        return f"{dates[0]}-长期"
    if len(dates) >= 2:
        return f"{dates[0]}-{dates[1]}"
    return ""


def _explicit_duration(text: str) -> str:
    """Return duration only when OCR explicitly prints years as a value."""

    for line in _label_window(
        text,
        ("有效期限", "有效年限"),
        before=0,
        after=2,
    ):
        # A date interval is not an independent duration value.
        if len(_date_candidates(line)) >= 1 or "至" in line:
            continue
        match = re.search(r"(?<!\d)(6|10)\s*年(?!\d)", line)
        if match:
            return f"{match.group(1)}年"
    return ""


def _driving_class_from_ocr(text: str) -> str:
    """Read the holder's class near its label, not from the code table."""

    candidates: list[str] = []
    sections = re.split(r"(?=\[(?:图片|局部复识别)\s)", text)
    stop_labels = (
        "累积记分",
        "初次领证日期",
        "有效期限",
        "出生日期",
        "证号",
        "姓名",
        "住址",
        "国籍",
        "档案编号",
    )
    for section in sections:
        marker_positions = [
            section.find(marker)
            for marker in _CLASS_EXPLANATION_MARKERS
            if marker in section
        ]
        if marker_positions:
            section = section[: min(marker_positions)]
        lines = _lines(section)
        section_candidates: list[str] = []
        for index, line in enumerate(lines):
            is_local_header = line.startswith("[局部复识别") and line.endswith(
                "/准驾车型]"
            )
            if (
                "准驾车型" not in line
                and line.lower() != "class"
                and not is_local_header
            ):
                continue
            nearby_lines = lines[max(0, index - 2) : index] + lines[
                index : index + 4
            ]
            for candidate_line in nearby_lines:
                if candidate_line != line and any(
                    label in candidate_line for label in stop_labels
                ):
                    continue
                compact = re.sub(r"[\s:：,，、.]+", "", candidate_line).upper()
                compact = compact.replace("准驾车型", "").replace("CLASS", "")
                compact = compact.strip("[]")
                # Common PP-OCRv6 confusions for the digit 1 inside a
                # labelled class value. Never apply these substitutions to
                # unlabelled free text or the class-code explanation table.
                compact = re.sub(r"^C[I|LJ](?=$|[DEFMNP])", "C1", compact)
                compact = re.sub(r"^B[I|L](?=$|[DEFMNP])", "B1", compact)
                if _DRIVING_CLASS.fullmatch(compact):
                    section_candidates.append(compact)
                    break
        if not section_candidates and (
            "准驾车型" in section or re.search(r"(?im)^Class\s*$", section)
        ):
            # PP-OCRv6 can emit the value later than its label on a rotated
            # physical licence. Accept only a complete standalone A/B/C class
            # token from the same non-explanation image section.
            for candidate_line in lines:
                compact = re.sub(r"[\s:：,，、.]+", "", candidate_line).upper()
                compact = re.sub(r"^C[I|LJ](?=$|[DEFMNP])", "C1", compact)
                compact = re.sub(r"^B[I|L](?=$|[DEFMNP])", "B1", compact)
                if (
                    compact.startswith(("A", "B", "C"))
                    and _DRIVING_CLASS.fullmatch(compact)
                ):
                    section_candidates.append(compact)
        candidates.extend(dict.fromkeys(section_candidates))
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else ""


def _archive_evidence(text: str) -> tuple[str, bool]:
    """Prefer full-image archive evidence over enhanced local OCR.

    Local enhancement is useful when the full image misses a value, but
    security backgrounds can also distort individual digits. A unique
    full-image 12-digit value therefore wins over conflicting local rechecks.
    """

    full_image_candidates: list[str] = []
    local_candidates: list[str] = []
    sections = re.split(r"(?=\[(?:图片|局部复识别)\s)", text)
    for section in sections:
        if "档案编号" not in section and "档案号码" not in section:
            continue
        candidates = list(dict.fromkeys(_ARCHIVE.findall(section)))
        if not candidates:
            continue
        lines = _lines(section)
        is_local = bool(lines and lines[0].startswith("[局部复识别"))
        target = local_candidates if is_local else full_image_candidates
        target.extend(candidate for candidate in candidates if candidate not in target)

    if len(full_image_candidates) == 1:
        return full_image_candidates[0], False
    if len(full_image_candidates) > 1:
        return "", True
    if len(local_candidates) == 1:
        return local_candidates[0], False
    return "", len(local_candidates) > 1


def detect_driver_document_type(text: str) -> str:
    """Classify a driver licence from explicit full-page OCR evidence."""

    compact = re.sub(r"\s+", "", text)
    electronic_markers = sum(
        marker in compact
        for marker in (
            "累积记分",
            "同意他人扫码获取驾驶证信息",
            "生成日期",
            "当前时间",
            "主页",
            "刷新",
            "换照片",
            "下载",
        )
    )
    # PP-OCRv6 may lose the page title while retaining the stable controls
    # and status fields of the electronic-licence screen.
    electronic = (
        "电子驾驶证" in compact
        or electronic_markers >= 2
    )
    physical = "中华人民共和国机动车驾驶证" in compact or (
        "DrivingLicense" in compact and "RepublicofChina" in compact
    )
    if electronic and not physical:
        return "电子驾驶证"
    if physical and not electronic:
        return "中华人民共和国机动车驾驶证"
    return ""


def detect_driver_page_type(text: str) -> str:
    """Classify one image, preferring stable electronic-screen controls."""

    compact = re.sub(r"\s+", "", text)
    electronic_markers = sum(
        marker in compact
        for marker in (
            "累积记分",
            "同意他人扫码获取驾驶证信息",
            "生成日期",
            "当前时间",
            "主页",
            "刷新",
            "换照片",
            "下载",
        )
    )
    if electronic_markers >= 2:
        return "电子驾驶证"
    return detect_driver_document_type(text)


def _valid_identity_number(value: str) -> bool:
    value = value.upper()
    if not re.fullmatch(r"[1-9]\d{16}[0-9X]", value):
        return False
    if not _valid_date(value[6:14]):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    expected = checks[sum(int(digit) * weight for digit, weight in zip(value[:17], weights)) % 11]
    return value[-1] == expected


def _complete_identity_number(text: str) -> str:
    candidates = [
        match.upper()
        for match in _IDENTITY_NUMBER.findall(text)
        if _valid_identity_number(match)
    ]
    return unique_unmasked(candidates)


def _repair_address_digit_confusions(value: str) -> str:
    """Repair theta only when it is structurally the leading digit zero."""

    return _ADDRESS_THETA_ZERO.sub("0", value or "")


def _trim_trailing_authority_location(value: str) -> str:
    """Drop a new province fragment after a complete village address."""

    compact = re.sub(r"\s+", "", value or "")
    match = _TRAILING_AUTHORITY_LOCATION.match(compact)
    return match.group(1) if match else compact


def refine_driver_fields(
    ocr_text: str,
    fields: dict[str, Any],
) -> dict[str, str]:
    """Repair only fields supported by clear OCR or format evidence."""

    refined = {str(key): str(value or "") for key, value in fields.items()}

    current_identity = re.sub(r"\s+", "", refined.get("证号", "")).upper()
    if contains_mask(current_identity):
        complete_identity = _complete_identity_number(ocr_text)
        if complete_identity:
            current_identity = complete_identity
    refined["证号"] = current_identity
    refined["住址"] = _trim_trailing_authority_location(
        _repair_address_digit_confusions(
            prefer_normalized_driver_address(
                refined.get("住址", ""),
                ocr_text,
            )
        )
    )
    refined["住址"] = correct_address_divisions(refined["住址"])

    document_type = detect_driver_document_type(ocr_text)
    for field in ("证件类型", "类型"):
        if document_type and field in refined:
            refined[field] = document_type

    class_value = re.sub(
        r"[\s,，、]+",
        "",
        refined.get("准驾车型", ""),
    ).upper()
    ocr_class = _driving_class_from_ocr(ocr_text)
    if ocr_class:
        if not _DRIVING_CLASS.fullmatch(class_value):
            class_value = ocr_class
    elif not _DRIVING_CLASS.fullmatch(class_value):
        class_value = ""
    refined["准驾车型"] = class_value

    archive = re.sub(r"\D", "", refined.get("档案编号", ""))
    ocr_archive, archive_conflict = _archive_evidence(ocr_text)
    if ocr_archive:
        archive = ocr_archive
    elif archive_conflict:
        archive = ""
    elif len(archive) != 12:
        archive = ""
    refined["档案编号"] = archive

    initial_date = _valid_date(refined.get("初次领证日期", ""))
    ocr_initial_date = _date_after_label(
        ocr_text,
        ("初次领证日期", "Date of First Issue"),
    )
    if not initial_date and ocr_initial_date:
        initial_date = ocr_initial_date
    refined["初次领证日期"] = initial_date

    birth_date = _valid_date(refined.get("出生日期", ""))
    ocr_birth_date = _date_after_label(
        ocr_text,
        ("出生日期", "Date of Birth"),
    )
    if not birth_date and ocr_birth_date:
        birth_date = ocr_birth_date
    refined["出生日期"] = birth_date

    date_range = refined.get("有效期限", "")
    ocr_date_range = _date_range_from_ocr(ocr_text)
    if (
        not re.fullmatch(r"\d{8}-(?:\d{8}|长期)", date_range)
        and ocr_date_range
    ):
        date_range = ocr_date_range
    elif not re.fullmatch(r"\d{8}-(?:\d{8}|长期)", date_range):
        date_range = ""
    refined["有效期限"] = date_range

    if refined.get("性别") not in {"男", "女"}:
        refined["性别"] = ""

    return refined
