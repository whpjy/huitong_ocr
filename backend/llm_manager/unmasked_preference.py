"""Shared conservative helpers for preferring explicit unmasked OCR values."""

from __future__ import annotations

import re


_OCR_SECTION = re.compile(r"(?=\[(?:图片|局部复识别)\s)", re.IGNORECASE)
_MASK = re.compile(r"[*＊]")
_ADDRESS_STOP_MARKERS = (
    "姓名",
    "Name",
    "性别",
    "Sex",
    "国籍",
    "Nationality",
    "出生日期",
    "Date of Birth",
    "准驾车型",
    "Class",
    "发证机关",
    "记录",
    "证号",
    "所有人",
    "Owner",
    "使用性质",
    "Use Character",
    "品牌型号",
    "Model",
    "车辆识别代号",
    "Vehicle Identification Number",
    "发动机号码",
    "Engine No",
    "注册日期",
    "Register Date",
    "发证日期",
    "Issue Date",
)


def contains_mask(value: str) -> bool:
    return bool(_MASK.search(value or ""))


def unique_unmasked(values: list[str]) -> str:
    """Return a value only when one distinct, non-masked candidate exists."""

    candidates = list(
        dict.fromkeys(value.strip() for value in values if value.strip() and not contains_mask(value))
    )
    return candidates[0] if len(candidates) == 1 else ""


def address_candidates(text: str) -> list[str]:
    """Read complete address values only from an explicit address-label region."""

    candidates: list[str] = []
    for section in _OCR_SECTION.split(text):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            match = re.match(r"^(?:住址|Address)\s*[:：]?\s*(.*)$", line, re.IGNORECASE)
            if not match:
                continue
            parts: list[str] = []
            remainder = match.group(1).strip()
            if remainder and remainder.lower() != "address":
                parts.append(remainder)
            for following in lines[index + 1 : index + 4]:
                if following.lower() == "address":
                    continue
                if any(marker.lower() in following.lower() for marker in _ADDRESS_STOP_MARKERS):
                    break
                parts.append(following)
            value = re.sub(r"\s+", "", "".join(parts))
            if len(re.findall(r"[\u4e00-\u9fff]", value)) >= 4:
                candidates.append(value)
    return list(dict.fromkeys(candidates))


def prefer_complete_address(current: str, text: str) -> str:
    """Replace a masked address only with one unique explicit complete address."""

    if not contains_mask(current):
        return current
    complete = unique_unmasked(address_candidates(text))
    return complete or current


def normalized_driver_address_candidates(text: str) -> list[str]:
    """Read address candidates emitted by the driver layout normalizer."""

    values = re.findall(
        r"(?m)^\[驾驶证版式归一化/住址\]\s*\n([^\n]+)$",
        text,
    )
    return list(dict.fromkeys(re.sub(r"\s+", "", value) for value in values))


def _is_ordered_subsequence(shorter: str, longer: str) -> bool:
    iterator = iter(longer)
    return all(character in iterator for character in shorter)


def prefer_normalized_driver_address(current: str, text: str) -> str:
    """Prefer one normalized candidate only when it cannot contradict OCR."""

    candidate = unique_unmasked(normalized_driver_address_candidates(text))
    if not candidate:
        return prefer_complete_address(current, text)

    compact_current = re.sub(r"\s+", "", current or "")
    if (
        not compact_current
        or contains_mask(compact_current)
        or _is_ordered_subsequence(compact_current, candidate)
    ):
        return candidate
    return compact_current
