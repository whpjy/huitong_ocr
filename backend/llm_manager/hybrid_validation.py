"""PP-OCR evidence matching and targeted multimodal conflict rechecks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from .client import OpenAICompatibleClient, parse_json_object
from .hunyuan_ocr_prompts import local_recheck_prompt
from .profiles import ExtractionProfile
from .vision_text_adapter import adapt_profile_text


_IGNORED = re.compile(r"[\s:：,，.。·/\\_\-—()（）]+")
_SPOTTING_COORDINATE = re.compile(
    r"\(\s*\d{1,5}\s*,\s*\d{1,5}\s*\)"
)
_SPOTTING_REF = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)

_FIELD_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "姓名": ("姓名", "Name"),
    "性别": ("性别", "Sex"),
    "国籍": ("国籍", "Nationality"),
    "住址": ("住址", "Address"),
    "身份证号": ("公民身份号码", "公民身份证号码", "身份证号"),
    "证号": ("证号", "Certificate No", "Document No"),
    "出生日期": ("出生日期", "出生", "Date of Birth"),
    "初次领证日期": ("初次领证日期", "Date of First Issue"),
    "准驾车型": ("准驾车型", "Class"),
    "有效期限": ("有效期限", "Valid Period"),
    "有效期": ("有效期限", "有效期", "Valid Period"),
    "档案编号": ("档案编号", "档案号码", "File Number"),
    "档案号码": ("档案编号", "档案号码", "File Number"),
    "号牌号码": ("号牌号码", "Plate No."),
    "车牌号码": ("号牌号码", "Plate No."),
    "车辆类型": ("车辆类型", "Vehicle Type"),
    "所有人": ("所有人", "Owner"),
    "使用性质": ("使用性质", "Use Character"),
    "品牌型号": ("品牌型号", "Model"),
    "车辆识别代号": ("车辆识别代号", "车辆识别代码", "VIN"),
    "发动机号码": ("发动机号码", "发动机号", "Engine No."),
    "注册日期": ("注册日期", "RegisterDate"),
    "发证日期": ("发证日期", "Issue Date"),
    "总质量": ("总质量",),
    "核定载人数": ("核定载人数",),
    "核定载质量": ("核定载质量",),
    "强制报废期止": ("强制报废期止",),
    "检验有效期": ("检验有效期至", "检验有效期"),
    "准牵引总质量": ("准牵引总质量",),
    "整备质量": ("整备质量",),
    "外廓尺寸": ("外廓尺寸", "外库尺寸", "外尺寸"),
    "检验记录": ("检验记录", "能源种类", "燃料类型"),
}

_FAST_OVERRIDE_FIELDS = {
    "住址",
    "身份证号",
    "证号",
    "出生日期",
    "初次领证日期",
    "有效期限",
    "有效期",
    "准驾车型",
    "档案编号",
    "档案号码",
    "号牌号码",
    "车牌号码",
    "车辆识别代号",
    "发动机号码",
    "注册日期",
    "发证日期",
    "总质量",
    "核定载人数",
    "核定载质量",
    "强制报废期止",
    "检验有效期",
    "准牵引总质量",
    "整备质量",
    "外廓尺寸",
}

_PLATE_RE = re.compile(
    r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新使领学警港澳]"
    r"[A-Z][A-Z0-9]{5,6}"
)
_VIN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
_ENGINE_RE = re.compile(r"[A-Z0-9][A-Z0-9*\-]{3,29}")
_ARCHIVE_RE = re.compile(r"[A-Z0-9]{6,12}")
_DRIVER_CLASS_RE = re.compile(r"(?:A[1-3]|B[12]|C[1-6])(?:[DEF])?|[DEFMNP]")
_MASS_RE = re.compile(r"(\d{2,6}(?:\.\d+)?)\s*(?:KG|千克)")
_PEOPLE_RE = re.compile(r"([1-9]\d?)\s*人")
_DIMENSION_RE = re.compile(
    r"(\d{3,5})\s*[×X*]\s*(\d{3,5})\s*[×X*]\s*(\d{3,5})\s*(?:MM)?"
)
_ADDRESS_ALLOWED_RE = re.compile(r"[\u3400-\u9fffA-Z0-9*#]+")
_ADDRESS_COMPONENTS = (
    "省", "市", "州", "盟", "区", "县", "旗", "镇", "乡", "街道",
    "村", "社区", "路", "街", "巷", "弄", "号", "组", "屯", "队",
    "庄", "里", "湾", "寨", "楼", "栋", "幢", "单元", "室", "户",
)
_ADDRESS_FORBIDDEN_LABELS = (
    "公民身份号码", "身份证号", "签发机关", "有效期限", "出生日期",
    "准驾车型", "车辆识别代号", "发动机号码", "注册日期", "发证日期",
    "记录", "实习",
)
_ADDRESS_TERMINAL_RE = re.compile(
    r"(?:省|市|州|盟|区|县|旗|镇|乡|街道|村|社区|路|街|巷|弄|号|组|"
    r"屯|队|庄|里|湾|寨|楼|栋|幢|单元|室|户|\*+)$"
)
_ADMINISTRATIVE_SUFFIX_RE = re.compile(
    r"自治区|自治州|地区|省|市|盟|县|区|旗"
)


def compact(value: Any) -> str:
    return _IGNORED.sub("", str(value or "")).upper()


def _date_range_key(value: str) -> str:
    """Canonicalize OCR variants such as ``2025-01-01 至 长期``."""

    text = str(value or "").upper()
    dates = re.findall(
        r"(\d{4})\s*[-/.年]?\s*(\d{1,2})\s*[-/.月]?\s*(\d{1,2})(?:日)?",
        text,
    )
    normalized = [f"{year}{int(month):02d}{int(day):02d}" for year, month, day in dates]
    if normalized and ("长期" in text or re.search(r"(?:至|-)\s*长\s*$", text)):
        return f"{normalized[0]}-长期"
    if len(normalized) >= 2:
        return f"{normalized[0]}-{normalized[1]}"
    compact_text = compact(text)
    compact_match = re.fullmatch(r"(\d{8})(?:至|-)?(\d{8})", compact_text)
    if compact_match:
        return f"{compact_match.group(1)}-{compact_match.group(2)}"
    long_match = re.fullmatch(r"(\d{8})(?:至|-)?(?:长期|至长)", compact_text)
    if long_match:
        return f"{long_match.group(1)}-长期"
    return ""


def _equivalent(left: str, right: str, field: str = "") -> bool:
    if field == "有效期限":
        left_range = _date_range_key(left)
        right_range = _date_range_key(right)
        if left_range and right_range:
            return left_range == right_range
    return compact(left) == compact(right)


def _supports_candidate(recheck_value: str, candidate: str, field: str) -> bool:
    """Return whether crop OCR contains the complete candidate as evidence."""

    if _equivalent(recheck_value, candidate, field):
        return True
    if field == "有效期限":
        candidate_range = _date_range_key(candidate)
        return bool(candidate_range and candidate_range == _date_range_key(recheck_value))
    if field == "准驾车型":
        extracted = _field_recheck_candidate(recheck_value, field)
        return bool(extracted and _equivalent(extracted, candidate, field))
    needle = compact(candidate)
    return len(needle) >= 2 and needle in compact(recheck_value)


@dataclass(frozen=True)
class EvidenceMatch:
    value: str
    score: float
    token_indexes: tuple[int, ...]
    bbox: tuple[float, float, float, float] | None
    token_confidence: float | None


def _field_aliases(field: str) -> tuple[str, ...]:
    return _FIELD_LABEL_ALIASES.get(field, (field,))


def _label_indexes(field: str, tokens: list[dict[str, Any]]) -> list[int]:
    aliases = tuple(compact(alias) for alias in _field_aliases(field))
    return [
        index
        for index, token in enumerate(tokens)
        if any(
            compact(token.get("text")) == alias
            or compact(token.get("text")).startswith(alias)
            for alias in aliases
            if alias
        )
    ]


def _looks_like_any_label(value: Any) -> bool:
    candidate = compact(value)
    return any(
        candidate == compact(alias)
        for aliases in _FIELD_LABEL_ALIASES.values()
        for alias in aliases
    )


def _nearby_value_indexes(
    tokens: list[dict[str, Any]],
    label_index: int,
    field: str,
) -> list[int]:
    """Return a small label-relative token window in visual reading order."""

    label_box = tokens[label_index].get("bbox")
    indexes: set[int] = set()
    if isinstance(label_box, list) and len(label_box) == 4:
        left, top, right, bottom = map(float, label_box)
        label_height = max(1.0, bottom - top)
        center_y = (top + bottom) / 2
        for index, token in enumerate(tokens):
            if index == label_index or _looks_like_any_label(token.get("text")):
                continue
            box = token.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            item_left, item_top, item_right, item_bottom = map(float, box)
            item_height = max(1.0, item_bottom - item_top)
            item_center_y = (item_top + item_bottom) / 2
            same_row = (
                abs(item_center_y - center_y)
                <= max(label_height, item_height) * 1.1
                and item_right >= left - label_height * 2
                and item_left <= right + label_height * 30
            )
            below = (
                item_top >= top - label_height * 0.5
                and item_top - bottom <= label_height * 6
                and item_right >= left - label_height * 3
                and item_left <= right + label_height * 18
            )
            if same_row or below:
                indexes.add(index)

    # The service normally returns reading order. Keep a bounded fallback for
    # labels or values whose boxes are absent or slightly distorted.
    for index in range(max(0, label_index - 3), min(len(tokens), label_index + 7)):
        if index == label_index or _looks_like_any_label(tokens[index].get("text")):
            continue
        indexes.add(index)

    label_text = compact(tokens[label_index].get("text"))
    if any(
        label_text.startswith(compact(alias))
        and label_text != compact(alias)
        for alias in _field_aliases(field)
    ):
        indexes.add(label_index)
    return sorted(indexes)


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def _token_stream(
    tokens: list[dict[str, Any]],
    target: str,
) -> tuple[str, list[int]]:
    text: list[str] = []
    owners: list[int] = []
    chinese_target = any(_is_cjk(char) for char in target)
    target_has_latin = any("A" <= char <= "Z" for char in target)
    for index, token in enumerate(tokens):
        value = compact(token.get("text"))
        for char in value:
            if chinese_target and not (
                _is_cjk(char)
                or char.isdigit()
                or char == "*"
                or (target_has_latin and "A" <= char <= "Z")
            ):
                # Chinese certificates often insert English duplicate labels
                # (Address/Name/Class) between wrapped value lines.
                continue
            text.append(char)
            owners.append(index)
    return "".join(text), owners


def _bbox(tokens: list[dict[str, Any]], indexes: tuple[int, ...]) -> tuple[float, float, float, float] | None:
    boxes = [tokens[index].get("bbox") for index in indexes]
    boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    if not boxes:
        return None
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _short_token_evidence(
    original_target: str,
    target: str,
    tokens: list[dict[str, Any]],
    field: str,
) -> EvidenceMatch | None:
    """Find a one-character conflict in short OCR tokens.

    Two-character values additionally require a nearby field label so an
    unrelated word is not treated as evidence.
    """

    if not 2 <= len(target) <= 4:
        return None
    label = compact(field)
    label_indexes = [
        index
        for index, token in enumerate(tokens)
        if label and label in compact(token.get("text"))
    ]
    candidates: list[tuple[int, float, int, str]] = []
    for index, token in enumerate(tokens):
        candidate = compact(token.get("text"))
        if len(candidate) != len(target):
            continue
        if sum(left != right for left, right in zip(target, candidate)) != 1:
            continue
        label_nearby = any(0 < index - label_index <= 4 for label_index in label_indexes)
        if len(target) == 2 and not label_nearby:
            continue
        raw_score = token.get("score")
        confidence = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        # Prefer a value adjacent to its field label. Confidence only breaks
        # ties, preventing a high-confidence unrelated short word elsewhere
        # on the certificate from winning.
        candidates.append((int(label_nearby), confidence, index, candidate))
    if not candidates:
        return None
    _has_label, confidence, index, candidate = max(candidates)
    candidate_chars = list(original_target)
    compact_positions = [position for position, char in enumerate(original_target) if compact(char)]
    display_value = candidate
    if len(compact_positions) == len(candidate):
        for position, char in zip(compact_positions, candidate):
            candidate_chars[position] = char
        display_value = "".join(candidate_chars)
    return EvidenceMatch(
        value=display_value,
        score=round((len(target) - 1) / len(target), 6),
        token_indexes=(index,),
        bbox=_bbox(tokens, (index,)),
        token_confidence=round(confidence, 6) if confidence else None,
    )


def find_evidence(
    value: str,
    tokens: list[dict[str, Any]],
    field: str = "",
) -> EvidenceMatch | None:
    """Find the closest same-length OCR span for one multimodal field value."""

    original_target = str(value or "")
    target = compact(original_target)
    if len(target) < 2:
        return None
    stream, owners = _token_stream(tokens, target)
    if len(stream) < len(target):
        return None
    best_start = -1
    best_score = -1.0
    best_candidate = ""
    for start in range(0, len(stream) - len(target) + 1):
        candidate = stream[start : start + len(target)]
        score = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
        if score > best_score:
            best_start, best_score, best_candidate = start, score, candidate
            if score == 1.0:
                break
    max_differences = 3 if len(target) >= 12 else 2 if len(target) >= 5 else 1
    differences = sum(left != right for left, right in zip(target, best_candidate))
    minimum = 0.80 if len(target) >= 8 else 0.84
    if best_start < 0 or best_score < minimum or differences > max_differences:
        return _short_token_evidence(original_target, target, tokens, field)
    indexes = tuple(dict.fromkeys(owners[best_start : best_start + len(target)]))
    scores = [tokens[index].get("score") for index in indexes]
    numeric_scores = [float(score) for score in scores if isinstance(score, (int, float))]
    candidate_chars = list(original_target)
    compact_positions = [
        index for index, char in enumerate(original_target) if compact(char)
    ]
    if len(compact_positions) == len(best_candidate):
        for position, char in zip(compact_positions, best_candidate):
            candidate_chars[position] = char
        display_value = "".join(candidate_chars)
    else:
        display_value = best_candidate
    return EvidenceMatch(
        value=display_value,
        score=round(best_score, 6),
        token_indexes=indexes,
        bbox=_bbox(tokens, indexes),
        token_confidence=(round(sum(numeric_scores) / len(numeric_scores), 6) if numeric_scores else None),
    )


def find_labeled_evidence(
    value: str,
    tokens: list[dict[str, Any]],
    field: str,
) -> tuple[EvidenceMatch, float | None, bool] | None:
    """Find a value only inside a bounded spatial window around its label."""

    matches: list[tuple[EvidenceMatch, float | None]] = []
    for label_index in _label_indexes(field, tokens):
        candidate_indexes = _nearby_value_indexes(tokens, label_index, field)
        if not candidate_indexes:
            continue
        local_tokens = [tokens[index] for index in candidate_indexes]
        if len(compact(value)) == 1:
            exact = [
                index
                for index, token in enumerate(local_tokens)
                if compact(token.get("text")) == compact(value)
            ]
            if exact:
                local_index = max(
                    exact,
                    key=lambda index: float(local_tokens[index].get("score") or 0),
                )
                raw_score = local_tokens[local_index].get("score")
                evidence = EvidenceMatch(
                    value=value,
                    score=1.0,
                    token_indexes=(local_index,),
                    bbox=_bbox(local_tokens, (local_index,)),
                    token_confidence=(
                        float(raw_score)
                        if isinstance(raw_score, (int, float))
                        else None
                    ),
                )
            else:
                evidence = None
        else:
            evidence = find_evidence(value, local_tokens, field)
        if evidence is None:
            continue
        original_indexes = tuple(
            candidate_indexes[index] for index in evidence.token_indexes
        )
        token_scores = [tokens[index].get("score") for index in original_indexes]
        numeric_scores = [
            float(score) for score in token_scores if isinstance(score, (int, float))
        ]
        label_score = tokens[label_index].get("score")
        matches.append(
            (
                EvidenceMatch(
                    value=evidence.value,
                    score=evidence.score,
                    token_indexes=original_indexes,
                    bbox=_bbox(tokens, original_indexes),
                    # A field is only as reliable as its weakest contributing
                    # OCR line; an average can hide one bad fragment.
                    token_confidence=(
                        round(min(numeric_scores), 6) if numeric_scores else None
                    ),
                ),
                (
                    round(float(label_score), 6)
                    if isinstance(label_score, (int, float))
                    else None
                ),
            )
        )
    if not matches:
        return None
    distinct = {compact(item[0].value) for item in matches}
    evidence, label_confidence = max(
        matches,
        key=lambda item: (
            item[0].score,
            float(item[0].token_confidence or 0),
            float(item[1] or 0),
        ),
    )
    return evidence, label_confidence, len(distinct) == 1


def compare_fields_fast(
    fields: dict[str, str],
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare fields using label-bound PP evidence only."""

    comparisons: list[dict[str, Any]] = []
    for field, value in fields.items():
        if not compact(value):
            continue
        labeled = find_labeled_evidence(value, tokens, field)
        if labeled is None:
            comparisons.append(
                {
                    "field": field,
                    "multimodal_value": value,
                    "ppocr_value": "",
                    "status": "ppocr_no_labeled_evidence",
                    "association": "none",
                }
            )
            continue
        evidence, label_confidence, unique = labeled
        comparisons.append(
            {
                "field": field,
                "multimodal_value": value,
                "ppocr_value": evidence.value,
                "status": (
                    "consistent"
                    if _equivalent(value, evidence.value, field)
                    else "conflict"
                ),
                "association": "label_spatial",
                "association_unique": unique,
                "similarity": evidence.score,
                "ppocr_confidence": evidence.token_confidence,
                "label_confidence": label_confidence,
                "bbox": list(evidence.bbox) if evidence.bbox else None,
                "token_indexes": list(evidence.token_indexes),
            }
        )
    return comparisons


def _valid_date(value: str, *, allow_month_only: bool = False) -> bool:
    digits = re.sub(r"\D", "", value)
    if allow_month_only and len(digits) == 6:
        try:
            datetime.strptime(digits, "%Y%m")
        except ValueError:
            return False
        return True
    if len(digits) != 8:
        return False
    try:
        parsed = datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return False
    return 1950 <= parsed.year <= 2099


def _valid_identity_number(value: str) -> bool:
    candidate = compact(value)
    if not re.fullmatch(r"\d{17}[0-9X]", candidate):
        return False
    if not _valid_date(candidate[6:14]):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    return checks[sum(int(char) * weight for char, weight in zip(candidate[:17], weights)) % 11] == candidate[-1]


def _valid_date_range(value: str) -> bool:
    key = _date_range_key(value)
    if not key:
        return False
    start, end = key.split("-", 1)
    return _valid_date(start) and (end == "长期" or _valid_date(end))


def _valid_address(value: str) -> bool:
    """Reject obvious label leakage and non-address OCR before replacement."""

    candidate = compact(value)
    cjk_count = sum("\u3400" <= char <= "\u9fff" for char in candidate)
    return bool(
        5 <= len(candidate) <= 100
        and cjk_count >= 4
        and _ADDRESS_ALLOWED_RE.fullmatch(candidate)
        and any(component in candidate for component in _ADDRESS_COMPONENTS)
        and _ADDRESS_TERMINAL_RE.search(candidate)
        and not any(label in candidate for label in _ADDRESS_FORBIDDEN_LABELS)
        and not re.search(r"\d{15,18}[0-9X]", candidate)
        and not re.search(r"[士土]组(?:\d|$)", candidate)
    )


def _address_administrative_prefix(value: str) -> str:
    candidate = compact(value)
    matches = list(_ADMINISTRATIVE_SUFFIX_RE.finditer(candidate))
    return candidate[: matches[-1].end()] if matches else ""


def valid_fast_field_value(field: str, value: str) -> bool:
    """Apply deterministic validation before PP can replace Hunyuan."""

    candidate = compact(value)
    if not candidate:
        return False
    if field == "住址":
        return _valid_address(candidate)
    if field in {"号牌号码", "车牌号码"}:
        return bool(_PLATE_RE.fullmatch(candidate))
    if field == "车辆识别代号":
        return bool(_VIN_RE.fullmatch(candidate))
    if field == "发动机号码":
        return bool(_ENGINE_RE.fullmatch(candidate)) and not (
            candidate.isdigit() and len(candidate) == 8
        )
    if field == "身份证号":
        return _valid_identity_number(candidate)
    if field == "证号":
        return _valid_identity_number(candidate) or bool(
            re.fullmatch(r"[A-Z0-9*]{15,20}", candidate)
        )
    if field in {"档案编号", "档案号码"}:
        return bool(_ARCHIVE_RE.fullmatch(candidate)) and not (
            candidate.isdigit() and len(candidate) == 8 and _valid_date(candidate)
        )
    if field in {"出生日期", "初次领证日期", "注册日期", "发证日期", "强制报废期止"}:
        return _valid_date(candidate)
    if field == "检验有效期":
        return _valid_date(candidate, allow_month_only=True)
    if field in {"有效期限", "有效期"}:
        return _valid_date_range(value)
    if field == "准驾车型":
        return bool(_DRIVER_CLASS_RE.fullmatch(candidate))
    if field in {"总质量", "整备质量", "核定载质量", "准牵引总质量"}:
        match = _MASS_RE.fullmatch(candidate)
        return bool(match and 10 <= float(match.group(1)) <= 100000)
    if field == "核定载人数":
        match = _PEOPLE_RE.fullmatch(candidate)
        return bool(match and 1 <= int(match.group(1)) <= 99)
    if field == "外廓尺寸":
        match = _DIMENSION_RE.fullmatch(candidate)
        return bool(
            match
            and all(300 <= int(number) <= 30000 for number in match.groups())
        )
    return False


def _record_update_is_valid(
    record: dict[str, str],
    field: str,
    value: str,
) -> bool:
    proposed = dict(record)
    proposed[field] = value
    if field in {"总质量", "整备质量"}:
        total = re.search(r"\d+(?:\.\d+)?", proposed.get("总质量", ""))
        curb = re.search(r"\d+(?:\.\d+)?", proposed.get("整备质量", ""))
        if total and curb and float(total.group()) < float(curb.group()):
            return False
    if field in {"注册日期", "发证日期"}:
        registered = re.sub(r"\D", "", proposed.get("注册日期", ""))
        issued = re.sub(r"\D", "", proposed.get("发证日期", ""))
        if len(registered) == len(issued) == 8 and registered > issued:
            return False
    return True


def arbitrate_fast(
    comparison: dict[str, Any],
    current_record: dict[str, str],
) -> tuple[str, str, bool]:
    """Resolve a conflict without another model call."""

    field = str(comparison.get("field") or "")
    initial = str(comparison.get("multimodal_value") or "")
    ppocr = str(comparison.get("ppocr_value") or "")
    if field not in _FAST_OVERRIDE_FIELDS:
        return initial, "文本语义字段冲突，快速模式保留Hunyuan结果", False
    if comparison.get("association") != "label_spatial":
        return initial, "PP缺少可靠标签空间关联，保留Hunyuan结果", False
    if comparison.get("association_unique") is not True:
        return initial, "PP标签邻域存在多个候选，保留Hunyuan结果", False
    label_confidence = comparison.get("label_confidence")
    if not isinstance(label_confidence, (int, float)) or label_confidence < 0.95:
        return initial, "PP字段标签置信度不足0.95，保留Hunyuan结果", False
    confidence = comparison.get("ppocr_confidence")
    initial_valid = valid_fast_field_value(field, initial)
    minimum = 0.985 if initial_valid else 0.97
    if not isinstance(confidence, (int, float)) or confidence < minimum:
        return initial, f"PP候选最低置信度不足{minimum:.3f}，保留Hunyuan结果", False
    if not valid_fast_field_value(field, ppocr):
        return initial, "PP候选未通过字段格式校验，保留Hunyuan结果", False
    if field == "住址" and SequenceMatcher(
        None,
        compact(initial),
        compact(ppocr),
        autojunk=False,
    ).ratio() < 0.90:
        return initial, "PP地址与Hunyuan地址差异过大，保留Hunyuan结果", False
    if field == "住址":
        initial_prefix = _address_administrative_prefix(initial)
        ppocr_prefix = _address_administrative_prefix(ppocr)
        if (
            initial_prefix
            and ppocr_prefix
            and initial_prefix != ppocr_prefix
            and confidence < 0.995
        ):
            return (
                initial,
                "PP与Hunyuan行政区划前缀冲突且最低置信度不足0.995，保留Hunyuan结果",
                False,
            )
    if not _record_update_is_valid(current_record, field, ppocr):
        return initial, "PP候选未通过跨字段关系校验，保留Hunyuan结果", False
    return ppocr, "PP标签空间关联可靠、候选唯一且通过高置信度与字段规则校验", True


def compare_fields(fields: dict[str, str], tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for field, value in fields.items():
        if not compact(value):
            continue
        evidence = find_evidence(value, tokens, field)
        if evidence is None:
            comparisons.append({
                "field": field,
                "multimodal_value": value,
                "ppocr_value": "",
                "status": "ppocr_no_evidence",
            })
            continue
        status = (
            "consistent"
            if _equivalent(value, evidence.value, field)
            else "conflict"
        )
        comparisons.append({
            "field": field,
            "multimodal_value": value,
            "ppocr_value": evidence.value,
            "status": status,
            "similarity": evidence.score,
            "ppocr_confidence": evidence.token_confidence,
            "bbox": list(evidence.bbox) if evidence.bbox else None,
            "token_indexes": list(evidence.token_indexes),
        })
    return comparisons


def missing_field_supplements(
    profile: ExtractionProfile,
    fields: dict[str, str],
    pages: list[dict[str, Any]],
    *,
    minimum_confidence: float = 0.97,
) -> list[dict[str, Any]]:
    """Find high-confidence PP-OCR values for empty multimodal fields."""

    text = "\n\n".join(str(page.get("text") or "") for page in pages)
    if not text.strip():
        return []
    try:
        candidates = adapt_profile_text(profile, text)[0]
    except Exception:  # noqa: BLE001 - supplementation must stay optional.
        return []
    supplements: list[dict[str, Any]] = []
    for field, value in fields.items():
        if compact(value):
            continue
        candidate = str(candidates.get(field) or "")
        direct_evidence: list[tuple[EvidenceMatch, int]] = []
        if profile.key == "driver_license" and field == "准驾车型":
            for image_index, page in enumerate(pages):
                evidence = _driver_class_evidence(page.get("tokens") or [])
                if evidence is not None:
                    direct_evidence.append((evidence, image_index))
            if direct_evidence:
                best, _image_index = max(
                    direct_evidence,
                    key=lambda item: float(item[0].token_confidence or 0),
                )
                candidate = best.value
        if not compact(candidate):
            continue
        evidence_items: list[tuple[EvidenceMatch, int, float | None, bool]] = [
            (evidence, image_index, 1.0, True)
            for evidence, image_index in direct_evidence
        ]
        for image_index, page in enumerate(pages):
            if direct_evidence:
                break
            tokens = page.get("tokens") or []
            if not isinstance(tokens, list):
                continue
            labeled = find_labeled_evidence(candidate, tokens, field)
            if labeled is None:
                continue
            evidence, label_confidence, unique = labeled
            if _equivalent(candidate, evidence.value):
                evidence_items.append(
                    (evidence, image_index, label_confidence, unique)
                )
        if not evidence_items:
            continue
        evidence, image_index, label_confidence, unique = max(
            evidence_items,
            key=lambda item: (
                item[3],
                float(item[0].token_confidence or 0),
                float(item[2] or 0),
            ),
        )
        confidence = evidence.token_confidence
        if not isinstance(confidence, (int, float)) or confidence < minimum_confidence:
            continue
        if not isinstance(label_confidence, (int, float)) or label_confidence < 0.95:
            continue
        if not unique:
            continue
        if field in _FAST_OVERRIDE_FIELDS and not valid_fast_field_value(field, candidate):
            continue
        supplements.append({
            "field": field,
            "value": candidate,
            "image_index": image_index,
            "ppocr_confidence": confidence,
            "label_confidence": label_confidence,
            "association": "label_spatial",
            "association_unique": unique,
            "bbox": list(evidence.bbox) if evidence.bbox else None,
            "token_indexes": list(evidence.token_indexes),
        })
    return supplements


def _driver_class_evidence(tokens: list[dict[str, Any]]) -> EvidenceMatch | None:
    """Read the closest class-shaped token following a licence class label."""

    label_indexes = [
        index
        for index, token in enumerate(tokens)
        if "准驾车型" in compact(token.get("text"))
        or compact(token.get("text")) == "CLASS"
    ]
    matches: list[tuple[int, float, int, str]] = []
    pattern = re.compile(r"(?:A[1-3]|B[12]|C[1-6])(?:[DEF])?|[DEFMNP]")
    for label_index in label_indexes:
        for index in range(label_index + 1, min(len(tokens), label_index + 7)):
            candidate = compact(tokens[index].get("text"))
            if not pattern.fullmatch(candidate):
                continue
            score = tokens[index].get("score")
            confidence = float(score) if isinstance(score, (int, float)) else 0.0
            matches.append((index - label_index, -confidence, index, candidate))
    if not matches:
        return None
    _distance, negative_confidence, index, candidate = min(matches)
    return EvidenceMatch(
        value=candidate,
        score=1.0,
        token_indexes=(index,),
        bbox=_bbox(tokens, (index,)),
        token_confidence=round(-negative_confidence, 6),
    )


def _single_character_evidence(
    value: str,
    tokens: list[dict[str, Any]],
    field: str,
) -> EvidenceMatch | None:
    """Accept an exact one-character token only next to its field label."""

    target = compact(value)
    label = compact(field)
    label_indexes = [
        index
        for index, token in enumerate(tokens)
        if label and label in compact(token.get("text"))
    ]
    candidates: list[tuple[float, int]] = []
    for index, token in enumerate(tokens):
        if compact(token.get("text")) != target:
            continue
        if not any(0 < index - label_index <= 3 for label_index in label_indexes):
            continue
        score = token.get("score")
        confidence = float(score) if isinstance(score, (int, float)) else 0.0
        candidates.append((confidence, index))
    if not candidates:
        return None
    confidence, index = max(candidates)
    return EvidenceMatch(
        value=value,
        score=1.0,
        token_indexes=(index,),
        bbox=_bbox(tokens, (index,)),
        token_confidence=round(confidence, 6) if confidence else None,
    )


def create_recheck_images(
    source: Path,
    bbox: list[float],
    output_dir: Path,
    field: str,
) -> list[Path]:
    """Save auditable original/color/grayscale field crops."""

    if len(bbox) != 4 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in bbox
    ):
        raise ValueError("复核字段坐标必须包含四个有限数值")
    left, top, right, bottom = map(float, bbox)
    if right <= left or bottom <= top:
        raise ValueError("复核字段坐标顺序无效")

    with Image.open(source) as image:
        # PP-OCR coordinates refer to the visually oriented image. JPEG files
        # can still store that orientation only as EXIF metadata, so crop from
        # the transposed pixel canvas rather than Pillow's raw matrix.
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = right - left, bottom - top
        pad_x = max(16, width * 0.08)
        pad_y = max(14, height * 0.45)
        box = (
            min(rgb.width, max(0, round(left - pad_x))),
            min(rgb.height, max(0, round(top - pad_y))),
            min(rgb.width, max(0, round(right + pad_x))),
            min(rgb.height, max(0, round(bottom + pad_y))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(
                "复核字段坐标未落在图片范围内："
                f"bbox={bbox}, image={rgb.width}x{rgb.height}"
            )
        crop = rgb.crop(box)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_field = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", field)
        original = output_dir / f"{safe_field}_01原始裁剪.png"
        crop.save(original)
        scale = min(8, max(3, round(360 / max(crop.height, 1))))
        enlarged = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        color = ImageOps.autocontrast(enlarged, cutoff=1)
        color = ImageEnhance.Sharpness(color).enhance(1.45)
        color_path = output_dir / f"{safe_field}_02彩色增强.png"
        color.save(color_path)
        gray = ImageOps.autocontrast(ImageOps.grayscale(enlarged), cutoff=1)
        gray = ImageEnhance.Sharpness(gray).enhance(1.8).convert("RGB")
        gray_path = output_dir / f"{safe_field}_03灰度增强.png"
        gray.save(gray_path)
    return [original, color_path, gray_path]


def recheck_conflict(
    client: OpenAICompatibleClient,
    field: str,
    images: list[Path],
) -> dict[str, Any]:
    if client.config.vision_response_adapter == "hunyuan_ocr":
        system = ""
        user = local_recheck_prompt()
    else:
        system = "你是严谨的证件局部文字复核引擎，只能依据图片可见内容逐字读取，不得猜测或补全。"
        user = (
            f"图片是证件字段“{field}”的同一区域及其增强版本。请综合辨认该字段完整文字，"
            f"输出合法JSON：{{\"{field}\":\"识别结果\"}}。看不清则输出空字符串。"
        )
    payload = client.build_general_vision_payload(images, system_prompt=system, user_prompt=user)
    raw = client._complete_payload(payload)
    return {"value": parse_recheck_value(raw, field), "raw_content": raw}


def parse_recheck_value(raw: str, field: str) -> str:
    """Accept JSON plus HunyuanOCR plain/spotting responses for a crop."""

    try:
        parsed = parse_json_object(raw)
        return str(parsed.get(field) or "").strip()
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        spotting = json.loads(raw)
    except json.JSONDecodeError:
        spotting = None
    if isinstance(spotting, list):
        raw = "\n".join(
            str(item.get("text") or "").strip()
            for item in spotting
            if isinstance(item, dict) and item.get("text")
        )
    references = [
        item.strip() for item in _SPOTTING_REF.findall(raw) if item.strip()
    ]
    if references:
        values = [item for item in references if compact(item) != compact(field)]
        if values:
            return "".join(values).strip()
    value = _SPOTTING_COORDINATE.sub("", raw)
    value = re.sub(r"<\|/?(?:ref|det)\|>", "", value)
    value = value.strip().strip("`\"' ：:，,。")
    table_match = re.search(
        rf"\|\s*{re.escape(field)}\s*\|\s*([^|\r\n]+)",
        value,
    )
    if table_match:
        value = table_match.group(1).strip()
    lines: list[str] = []
    for line in value.splitlines():
        line = re.sub(
            r"^\s*(?:图片中的(?:文本内容|文字)|识别结果)\s*(?:是|为)?\s*[:：]?\s*",
            "",
            line,
        )
        cleaned = re.sub(
            rf"^\s*{re.escape(field)}\s*[:：,，]?\s*",
            "",
            line,
        ).strip(" `\"'：:,，。")
        if cleaned and compact(cleaned) != compact(field):
            lines.append(cleaned)
    # HunyuanOCR may repeat the same crop text for original/color/gray inputs.
    value = "".join(dict.fromkeys(lines))
    candidate = _field_recheck_candidate(value, field)
    if candidate:
        return candidate
    # A local crop may be returned as one plain value. Reject explanations or
    # a whole-page spotting response so arbitration cannot accept garbage.
    if not value or len(value) > 160:
        raise ValueError("HunyuanOCR 局部复核未返回可用字段值")
    if any(marker in value for marker in ("无法识别", "识别结果如下", "JSON")):
        raise ValueError("HunyuanOCR 局部复核返回了解释文本")
    return value


def _field_recheck_candidate(value: str, field: str) -> str:
    """Extract a field-shaped candidate from otherwise verbose crop OCR."""

    if field == "证号":
        match = re.search(r"(?<![0-9A-Z*])[0-9*]{17}[0-9X*](?![0-9A-Z*])", value.upper())
        return match.group(0) if match else ""
    if field == "档案编号":
        match = re.search(r"(?<!\d)\d{12}(?!\d)", value)
        return match.group(0) if match else ""
    if field in {"出生日期", "初次领证日期"}:
        match = re.search(r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?:日)?", value)
        return match.group(0) if match else ""
    if field == "有效期限":
        match = re.search(
            r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?:日)?"
            r"\s*(?:至|-)\s*(?:长期|\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?:日)?)",
            value,
        )
        return match.group(0) if match else ""
    if field == "准驾车型":
        match = re.search(
            r"(?<![A-Z0-9])(?:A[1-3]|B[12]|C[1-6])(?:[DEF])?(?![A-Z0-9])|"
            r"(?<![A-Z0-9])[DEFMNP](?![A-Z0-9])",
            value.upper(),
        )
        return match.group(0) if match else ""
    return ""


def arbitrate(comparison: dict[str, Any], recheck_value: str) -> tuple[str, str, bool]:
    """Return final value, decision reason and whether the source was changed."""

    initial = str(comparison.get("multimodal_value") or "")
    ppocr = str(comparison.get("ppocr_value") or "")
    field = str(comparison.get("field") or "")
    confidence = comparison.get("ppocr_confidence")
    if (
        ppocr
        and _supports_candidate(recheck_value, ppocr, field)
        and not _equivalent(initial, ppocr, field)
        and isinstance(confidence, (int, float))
        and confidence >= 0.85
    ):
        return ppocr, "局部复测支持PP-OCRv6，且OCR证据置信度达到0.85", True
    if _equivalent(recheck_value, initial, field):
        return initial, "局部复测支持首次多模态结果，保留首次结果", False
    return initial, "局部复测未形成可靠一致证据，保留首次多模态结果并记录冲突", False
