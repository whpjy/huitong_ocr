"""Conservative, deterministic repairs for resident ID-card extraction."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from llm_manager.address_corrector import correct_address_divisions


_SECTION = re.compile(r"(?=^\[(?:图片|局部复识别)\s)", re.MULTILINE)
_ADDRESS_LABEL = re.compile(r"^住址\s*[:：]?\s*(.*)$")
_IDENTITY_NUMBER = re.compile(
    r"(?<![0-9A-Z])([1-9]\d{16}[0-9X])(?![0-9A-Z])",
    re.IGNORECASE,
)
_ADDRESS_STOP_MARKERS = (
    "公民身份号码",
    "公民身份证号码",
    "身份号码",
    "签发机关",
    "有效期限",
    "中华人民共和国",
    "居民身份证",
    "中国CHINA",
    "中国/CHN",
)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _address_candidates(text: str) -> list[str]:
    """Read one explicit address region per image without bare-number noise."""

    candidates: list[str] = []
    for section in _SECTION.split(text):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            match = _ADDRESS_LABEL.match(line)
            if not match:
                continue
            parts: list[str] = []
            remainder = _compact(match.group(1))
            if remainder:
                parts.append(remainder)
            for following in lines[index + 1 : index + 7]:
                compact = _compact(following)
                if not compact or compact.lower() == "address":
                    continue
                if any(marker in compact for marker in _ADDRESS_STOP_MARKERS):
                    break
                # A standalone number immediately before the identity-number
                # label is commonly a layout artifact, not an address suffix.
                if re.fullmatch(r"\d+", compact):
                    break
                # Watermarks and English UI fragments are not address lines.
                if not re.search(r"[\u4e00-\u9fff]", compact):
                    break
                parts.append(compact)
            value = "".join(parts)
            if len(re.findall(r"[\u4e00-\u9fff]", value)) >= 5:
                candidates.append(value)
    return candidates


def _is_ordered_subsequence(shorter: str, longer: str) -> bool:
    iterator = iter(longer)
    return all(character in iterator for character in shorter)


def _common_suffix_length(first: str, second: str) -> int:
    length = 0
    for left, right in zip(reversed(first), reversed(second)):
        if left != right:
            break
        length += 1
    return length


def _common_prefix_length(first: str, second: str) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def _preferred_address(text: str, current: str) -> str:
    candidates = _address_candidates(text)
    if not candidates:
        return ""

    counts = Counter(candidates)
    ranked = counts.most_common()
    unique = list(counts)
    if len(unique) > 1:
        clean = [
            candidate
            for candidate in unique
            if all(
                candidate == other
                or (
                    len(candidate) < len(other)
                    and _is_ordered_subsequence(candidate, other)
                    and _common_prefix_length(candidate, other) >= 8
                    and _common_suffix_length(candidate, other) >= 3
                )
                for other in unique
            )
        ]
        return clean[0] if len(clean) == 1 else ""

    candidate, support = ranked[0]
    compact_current = _compact(current)
    if not compact_current and support >= 2:
        return candidate
    if compact_current == candidate:
        return candidate
    # Remove a middle insertion only when repeated OCR agrees and both values
    # retain the same meaningful address ending.
    if (
        support >= 2
        and len(candidate) < len(compact_current)
        and _is_ordered_subsequence(candidate, compact_current)
        and _common_suffix_length(candidate, compact_current) >= 4
    ):
        return candidate
    # A bare trailing number with no address unit is a known layout artifact.
    if (
        compact_current.startswith(candidate)
        and re.fullmatch(r"\d{2,4}", compact_current[len(candidate) :])
        and re.search(r"[村屯组号室户]$", candidate)
    ):
        return candidate
    return ""


def _valid_date(value: str) -> bool:
    if not re.fullmatch(r"\d{8}", value):
        return False
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return 1900 <= parsed.year <= 2099


def _valid_identity_number(value: str) -> bool:
    value = value.upper()
    if not re.fullmatch(r"[1-9]\d{16}[0-9X]", value):
        return False
    if not _valid_date(value[6:14]):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    expected = checks[
        sum(
            int(digit) * weight
            for digit, weight in zip(value[:17], weights)
        )
        % 11
    ]
    return value[-1] == expected


def _unique_identity_birth_date(text: str) -> str:
    identities = list(
        dict.fromkeys(
            value.upper()
            for value in _IDENTITY_NUMBER.findall(text)
            if _valid_identity_number(value)
        )
    )
    return identities[0][6:14] if len(identities) == 1 else ""


def refine_id_card_fields(
    ocr_text: str,
    fields: dict[str, Any],
) -> dict[str, str]:
    """Repair address layout and birth date using unambiguous OCR evidence."""

    refined = {str(key): str(value or "") for key, value in fields.items()}

    address = _preferred_address(ocr_text, refined.get("住址", ""))
    if address:
        refined["住址"] = address
    refined["住址"] = correct_address_divisions(refined.get("住址", ""))

    identity_birth = _unique_identity_birth_date(ocr_text)
    if identity_birth:
        refined["出生日期"] = identity_birth

    return refined
