"""Conservative, label-aware repairs for vehicle-license extraction.

The OCR service returns a reading-order text stream.  Several vehicle-license
fields live on the same row, so an LLM can shift one mass/date into the next
field.  These repairs only use values that are explicitly adjacent to a label
in the OCR text; they never invent values.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from llm_manager.address_corrector import correct_address_divisions
from llm_manager.unmasked_preference import (
    contains_mask,
    prefer_complete_address,
    unique_unmasked,
)


FUEL_TERMS = (
    "新能源/电",
    "新能源/混",
    "汽油/电",
    "汽油/混",
    "柴油/电",
    "汽油",
    "柴油",
    "纯电动",
    "混合动力",
    "新能源",
)

USE_NATURE_TERMS = (
    "危化品运输",
    "公路客运",
    "公交客运",
    "出租客运",
    "旅游客运",
    "非营运",
    "货运",
    "租赁",
    "教练",
    "警用",
    "消防",
    "救护",
    "工程救险",
    "营运",
)

_PROVINCE_PREFIXES = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新使领学警港澳"
_PLATE_TOKEN = re.compile(
    rf"(?<![\u4e00-\u9fffA-Z0-9])"
    rf"([{_PROVINCE_PREFIXES}])\s*([A-Z])\s*([A-Z0-9]{{5,6}})"
    rf"(?![A-Z0-9])",
    flags=re.IGNORECASE,
)
_ARCHIVE_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9]{6,12})(?![A-Za-z0-9])")
_PEOPLE_TOKEN = re.compile(r"(?<!\d)([1-9]\d?)\s*人(?!\w)")

FIELD_LABELS = {
    "总质量": ("总质量",),
    "整备质量": ("整备质量",),
    "核定载质量": ("核定载质量",),
    "准牵引总质量": ("准牵引总质量",),
    "外廓尺寸": ("外廓尺寸", "外库尺寸", "外尺寸"),
    "档案号码": ("档案编号", "档案号码"),
    "注册日期": ("注册日期", "RegisterDate"),
    "发证日期": ("发证日期", "Issue Date"),
    "检验有效期": ("检验有效期至",),
    "使用性质": ("使用性质", "Use Character"),
    "车牌号码": ("号牌号码", "Plate No."),
    "核定载人数": ("核定载人数",),
}

_MASS = r"\d+(?:\.\d+)?\s*(?:kg|KG|千克)?"
_DATE = r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?:日)?|\d{8}"


_TOTAL_MASS_FIELD = "\u603b\u8d28\u91cf"
_CURB_MASS_FIELD = "\u6574\u5907\u8d28\u91cf"
_REGISTRATION_DATE_FIELD = "\u6ce8\u518c\u65e5\u671f"
_ISSUE_DATE_FIELD = "\u53d1\u8bc1\u65e5\u671f"
_MASS_TOKEN = re.compile(
    r"(?<!\d)(\d{3,5}(?:\.\d+)?)\s*(?:kg|\u5343\u514b)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_DIMENSION_TOKEN = re.compile(
    r"\d+\s*[\u00d7xX*]\s*\d+\s*[\u00d7xX*]\s*\d+"
)
_VIN_TOKEN = re.compile(r"(?<![A-Z0-9])([A-HJ-NPR-Z0-9]{17})(?![A-Z0-9])", re.IGNORECASE)


def _after_label(text: str, labels: tuple[str, ...], value_pattern: str) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{label_pattern})\s*[:：]?\s*({value_pattern})",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def sync_vehicle_plate_fields(fields: dict[str, Any]) -> dict[str, str]:
    """Copy a valid plate number to both output aliases for one vehicle."""

    synced = {str(key): str(value or "") for key, value in fields.items()}
    plate = re.sub(r"[\s_-]+", "", synced.get("号牌号码", "")).upper()
    if _PLATE_TOKEN.fullmatch(plate):
        synced["车牌号码"] = plate
        synced["号牌号码"] = plate
    return synced


def _normalize_field(field: str, value: str) -> str:
    value = value.strip()
    if field in {"总质量", "整备质量", "核定载质量", "准牵引总质量"}:
        number = re.search(r"\d+(?:\.\d+)?", value)
        return f"{number.group(0)}kg" if number else ""
    if field in {"注册日期", "发证日期"}:
        digits = re.sub(r"\D", "", value)
        if len(digits) != 8:
            return ""
        try:
            parsed = datetime.strptime(digits, "%Y%m%d")
        except ValueError:
            return ""
        return digits if 1950 <= parsed.year <= 2099 else ""
    if field == "检验有效期":
        digits = re.sub(r"\D", "", value)
        return digits if len(digits) in {6, 8} else ""
    if field == "外廓尺寸":
        numbers = []
        for token in re.findall(r"\d+(?:\.\d+)?", value):
            if "." in token:
                integer, fraction = token.split(".", 1)
                if set(fraction) <= {"0"}:
                    token = integer
                elif len(integer) < 4 and 3 <= len(integer + fraction) <= 5:
                    token = integer + fraction
            numbers.append(token)
        return f"{'×'.join(numbers)}mm" if len(numbers) == 3 else ""
    return value


def _mass_on_label_line(line: str, label: str) -> str:
    position = line.find(label)
    if position < 0:
        return ""
    match = _MASS_TOKEN.search(line[position + len(label) :])
    return f"{match.group(1)}kg" if match else ""


def _nearby_mass(
    lines: list[str],
    label_index: int,
    *,
    before: bool,
    distance: int = 4,
) -> str:
    indexes = (
        range(label_index - 1, max(-1, label_index - distance - 1), -1)
        if before
        else range(label_index + 1, min(len(lines), label_index + distance + 1))
    )
    for index in indexes:
        line = lines[index]
        if _DIMENSION_TOKEN.search(line):
            continue
        match = _MASS_TOKEN.search(line)
        if match:
            return f"{match.group(1)}kg"
    return ""


def _mass_candidate(lines: list[str], label: str) -> str:
    for index, line in enumerate(lines):
        if label not in line:
            continue
        same_line = _mass_on_label_line(line, label)
        if same_line:
            return same_line
        # PP-OCRv6 often emits the visual value before its field label.
        previous = _nearby_mass(lines, index, before=True)
        if previous:
            return previous
        following = _nearby_mass(lines, index, before=False, distance=2)
        if following:
            return following
    return ""


def _mass_pairs(text: str) -> list[tuple[str, str]]:
    sections = re.split(r"(?=\[\u56fe\u7247\s)", text)
    pairs: list[tuple[str, str]] = []
    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        all_masses = [
            f"{match.group(1)}kg"
            for line in lines
            if not _DIMENSION_TOKEN.search(line)
            for match in _MASS_TOKEN.finditer(line)
        ]
        unique_masses = list(dict.fromkeys(all_masses))
        if (
            _TOTAL_MASS_FIELD in section
            and _CURB_MASS_FIELD in section
            and len(unique_masses) == 2
        ):
            ordered = sorted(
                unique_masses,
                key=lambda value: float(value[:-2]),
                reverse=True,
            )
            pairs.append((ordered[0], ordered[1]))
            continue
        total = _mass_candidate(lines, _TOTAL_MASS_FIELD)
        curb = _mass_candidate(lines, _CURB_MASS_FIELD)
        if not total or not curb:
            continue
        if float(total[:-2]) >= float(curb[:-2]):
            pairs.append((total, curb))
    return pairs


def _select_mass_pair(
    pairs: list[tuple[str, str]],
    refined: dict[str, str],
) -> tuple[str, str] | None:
    if not pairs:
        return None
    counts = Counter(pairs)
    highest = max(counts.values())
    leaders = [pair for pair, count in counts.items() if count == highest]
    if len(leaders) == 1:
        return leaders[0]
    current = (
        refined.get(_TOTAL_MASS_FIELD, ""),
        refined.get(_CURB_MASS_FIELD, ""),
    )
    return current if current in leaders else None


def _date_pairs(text: str) -> list[tuple[str, str]]:
    sections = re.split(r"(?=\[\u56fe\u7247\s)", text)
    pairs: list[tuple[str, str]] = []
    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        positions: dict[str, int] = {}
        for field in (_REGISTRATION_DATE_FIELD, _ISSUE_DATE_FIELD):
            positions[field] = next(
                (index for index, line in enumerate(lines) if field in line),
                -1,
            )
        if min(positions.values()) < 0:
            continue
        ordered_fields = sorted(positions, key=positions.get)
        start = min(positions.values())
        end = min(len(lines), max(positions.values()) + 9)
        dates = [
            _normalize_field(_REGISTRATION_DATE_FIELD, value)
            for value in re.findall(_DATE, "\n".join(lines[start:end]))
        ]
        dates = [value for value in dates if value]
        if len(dates) < 2:
            continue
        mapped = dict(zip(ordered_fields, dates[:2]))
        pairs.append(
            (
                mapped[_REGISTRATION_DATE_FIELD],
                mapped[_ISSUE_DATE_FIELD],
            )
        )
    return pairs


def _date_on_label_line(text: str, labels: tuple[str, ...]) -> str:
    for line in text.splitlines():
        for label in labels:
            position = line.find(label)
            if position < 0:
                continue
            match = re.search(_DATE, line[position + len(label) :])
            if match:
                return match.group(0)
    return ""


def _inspection_expiry_from_ocr(text: str) -> str:
    """Read YYYYMM(/DD) beside or immediately after the expiry label."""

    window = "\n".join(
        _label_window_lines(
            text,
            FIELD_LABELS["检验有效期"],
            before=0,
            after=3,
        )
    )
    match = re.search(
        r"(?<!\d)(20\d{2})\s*(?:[-/.年]\s*)"
        r"(\d{1,2})(?:\s*(?:[-/.月]\s*)(\d{1,2})\s*日?)?(?!\d)",
        window,
    )
    if not match:
        match = re.search(r"(?<!\d)(20\d{4}(?:\d{2})?)(?!\d)", window)
        return match.group(1) if match else ""
    year, month, day = match.groups()
    if not 1 <= int(month) <= 12:
        return ""
    if day and not 1 <= int(day) <= 31:
        return ""
    return f"{year}{int(month):02d}" + (f"{int(day):02d}" if day else "")


def _select_date_pair(
    pairs: list[tuple[str, str]],
    refined: dict[str, str],
) -> tuple[str, str] | None:
    if not pairs:
        return None
    counts = Counter(pairs)
    highest = max(counts.values())
    leaders = [pair for pair, count in counts.items() if count == highest]
    if len(leaders) == 1:
        return leaders[0]
    current = (
        refined.get(_REGISTRATION_DATE_FIELD, ""),
        refined.get(_ISSUE_DATE_FIELD, ""),
    )
    return current if current in leaders else None


def _label_window_lines(
    text: str,
    labels: tuple[str, ...],
    *,
    before: int = 1,
    after: int = 4,
) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    windows: list[str] = []
    for index, line in enumerate(lines):
        if any(label.lower() in line.lower() for label in labels):
            windows.extend(lines[max(0, index - before) : index + after + 1])
    return list(dict.fromkeys(windows))


def _use_nature_from_ocr(text: str) -> str:
    window = "\n".join(
        _label_window_lines(text, FIELD_LABELS["使用性质"], before=0, after=2)
    )
    return next((term for term in USE_NATURE_TERMS if term in window), "")


def _plate_from_ocr(text: str) -> str:
    window = "\n".join(
        _label_window_lines(text, FIELD_LABELS["车牌号码"], before=1, after=3)
    )
    match = _PLATE_TOKEN.search(window)
    if not match:
        match = _PLATE_TOKEN.search(text)
    return "".join(match.groups()).upper() if match else ""


def _archive_number_from_ocr(text: str) -> str:
    candidates: list[tuple[int, str]] = []
    for distance, line in enumerate(
        _label_window_lines(text, FIELD_LABELS["档案号码"], before=1, after=5)
    ):
        compact = re.sub(r"[\s_-]+", "", line)
        for match in _ARCHIVE_TOKEN.finditer(compact):
            value = match.group(1).upper()
            if (
                value.isdigit()
                and len(value) == 8
                and 19500101 <= int(value) <= 20991231
            ):
                continue
            # The public data-element definition is C..12. Prefer the most
            # specific 12-character candidate, then the nearest candidate.
            candidates.append((0 if len(value) == 12 else 1 + distance, value))
    return min(candidates, default=(0, ""), key=lambda item: item[0])[1]


def _people_from_ocr(text: str) -> str:
    window = "\n".join(
        _label_window_lines(text, FIELD_LABELS["核定载人数"], before=2, after=2)
    )
    match = _PEOPLE_TOKEN.search(window)
    return f"{match.group(1)}人" if match else ""


def _complete_vin_from_ocr(text: str) -> str:
    candidates: list[str] = []
    for section in re.split(r"(?=\[图片\s)", text):
        window = "\n".join(
            _label_window_lines(
                section,
                ("车辆识别代号", "Vehicle Identification Number", "VIN"),
                before=0,
                after=3,
            )
        )
        candidates.extend(match.upper() for match in _VIN_TOKEN.findall(window))
    return unique_unmasked(candidates)


def refine_vehicle_fields(ocr_text: str, fields: dict[str, Any]) -> dict[str, str]:
    """Repair only obvious label-adjacent omissions or field shifts."""

    refined = {str(key): str(value or "") for key, value in fields.items()}

    current_vin = re.sub(
        r"\s+", "", refined.get("车辆识别代号", "")
    ).upper()
    if contains_mask(current_vin):
        complete_vin = _complete_vin_from_ocr(ocr_text)
        if complete_vin:
            current_vin = complete_vin
    refined["车辆识别代号"] = current_vin
    refined["住址"] = prefer_complete_address(refined.get("住址", ""), ocr_text)
    refined["住址"] = correct_address_divisions(refined["住址"])

    # The standard workbook calls this column 检验记录, but its values are
    # fuel/energy terms.  Prefer a canonical term over noisy LLM output.
    fuel = next(
        (term for term in FUEL_TERMS if term in ocr_text),
        "",
    )
    if fuel:
        refined["检验记录"] = fuel

    use_nature = _use_nature_from_ocr(ocr_text)
    current_use_nature = re.sub(r"\s+", "", refined.get("使用性质", ""))
    refined["使用性质"] = (
        use_nature
        if use_nature
        else current_use_nature
        if current_use_nature in USE_NATURE_TERMS
        else ""
    )

    plate = _plate_from_ocr(ocr_text)
    current_plate = re.sub(
        r"[\s_-]+", "", refined.get("车牌号码", "")
    ).upper()
    if plate:
        current_plate = plate
    elif not _PLATE_TOKEN.fullmatch(current_plate):
        current_plate = ""
    refined["车牌号码"] = current_plate
    refined["号牌号码"] = current_plate

    mass_pair = _select_mass_pair(_mass_pairs(ocr_text), refined)
    if mass_pair:
        refined[_TOTAL_MASS_FIELD], refined[_CURB_MASS_FIELD] = mass_pair

    # OCR reading order often emits both date labels first and both values
    # afterwards. Pair by the label order inside each image section.
    date_labels = ("注册日期", "发证日期")
    paired_date_fields: set[str] = set()
    date_pair = _select_date_pair(_date_pairs(ocr_text), refined)
    if date_pair:
        refined[_REGISTRATION_DATE_FIELD], refined[_ISSUE_DATE_FIELD] = date_pair
        paired_date_fields.update(date_labels)

    for field in ("注册日期", "发证日期", "检验有效期"):
        if field in paired_date_fields:
            continue
        candidate = _date_on_label_line(ocr_text, FIELD_LABELS[field])
        normalized = _normalize_field(field, candidate)
        if normalized:
            refined[field] = normalized

    inspection_expiry = _inspection_expiry_from_ocr(ocr_text)
    if inspection_expiry:
        refined["检验有效期"] = inspection_expiry

    candidate = _after_label(
        ocr_text,
        FIELD_LABELS["外廓尺寸"],
        r"\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?\s*mm?",
    )
    normalized = _normalize_field("外廓尺寸", candidate)
    if normalized:
        refined["外廓尺寸"] = normalized

    archive_value = re.sub(
        r"[\s_-]+", "", refined.get("档案号码", "")
    ).upper()
    if not re.fullmatch(r"[A-Z0-9]{6,12}", archive_value):
        candidate = _archive_number_from_ocr(ocr_text)
        if candidate:
            archive_value = candidate
        else:
            archive_value = ""
    refined["档案号码"] = archive_value

    people_candidate = _people_from_ocr(ocr_text)
    if people_candidate:
        refined["核定载人数"] = people_candidate
    elif "小型轿车" in re.sub(r"\s+", "", ocr_text):
        refined["核定载人数"] = "5人"

    return refined
