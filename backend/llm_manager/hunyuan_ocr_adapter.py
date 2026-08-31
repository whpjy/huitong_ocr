"""HunyuanOCR-specific structured extraction helpers."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

from .profiles import ExtractionProfile
from .hunyuan_ocr_prompts import ID_CARD_FRONT_FIELDS


_ID_CARD_ALIASES = {
    "公民身份号码": "身份证号",
    "公民身份证号码": "身份证号",
    "身份证号码": "身份证号",
    "出生": "出生日期",
    "有效期限": "有效期",
}

_COORDINATE_PAIR_RE = re.compile(
    r"\(\s*\d{1,5}\s*,\s*\d{1,5}\s*\)"
)
_COORDINATE_BOX_RE = re.compile(
    r"(?:\(\s*\d{1,5}\s*,\s*\d{1,5}\s*\)\s*,?\s*){2}"
)
_DOCUMENT_ALIASES = {
    "driver_license": {
        "副页档案编号": "档案编号",
        "档案号码": "档案编号",
        "有效期区间": "有效期限",
        "有效期限区间": "有效期限",
    },
    "vehicle_license": {
        "证件标题": "证件类型",
        "类型": "证件类型",
        "VIN": "车辆识别代号",
        "车辆识别代码": "车辆识别代号",
        "档案编号": "档案号码",
    },
}
_FIELD_LABEL_VALUES = {
    "姓名", "性别", "国籍", "住址", "出生日期", "初次领证日期",
    "准驾车型", "证号", "档案编号", "档案号码", "有效期限", "类型",
}


def contains_spotting_coordinates(value: Any) -> bool:
    """Detect HunyuanOCR spotting output accidentally returned as a value."""

    parsed = _spotting_value(value)
    if isinstance(parsed, list) and any(
        isinstance(item, Mapping)
        and "text" in item
        and any(key in item for key in ("box", "bbox", "polygon"))
        for item in parsed
    ):
        return True
    text = json_like_text(value)
    return len(_COORDINATE_PAIR_RE.findall(text)) >= 2


def spotting_text(value: Any) -> str:
    """Convert HunyuanOCR spotting JSON/text into clean reading-order text."""

    parsed = _spotting_value(value)
    if isinstance(parsed, list):
        lines = [
            str(item.get("text") or "").strip()
            for item in parsed
            if isinstance(item, Mapping) and item.get("text")
        ]
        if lines:
            return "\n".join(lines)
    text = str(value or "")
    text = _COORDINATE_BOX_RE.sub("\n", text)
    text = _COORDINATE_PAIR_RE.sub("", text)
    return "\n".join(
        line.strip(" ,，")
        for line in text.splitlines()
        if line.strip(" ,，")
    )


def spotting_markdown(value: Any) -> str:
    """Convert HunyuanOCR box/text output to a compact Markdown table."""

    parsed = _spotting_value(value)
    if not isinstance(parsed, list):
        return _lines_markdown(spotting_text(value).splitlines())

    positioned: list[tuple[float, float, float, str]] = []
    unpositioned: list[str] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        box = item.get("box") or item.get("bbox")
        if (
            isinstance(box, (list, tuple))
            and len(box) >= 4
            and all(isinstance(number, (int, float)) for number in box[:4])
        ):
            x1, y1, _x2, y2 = (float(number) for number in box[:4])
            positioned.append(((y1 + y2) / 2, x1, max(1.0, y2 - y1), text))
        else:
            unpositioned.append(text)

    if not positioned:
        return _lines_markdown(unpositioned)

    positioned.sort(key=lambda item: (item[0], item[1]))
    rows: list[list[tuple[float, str]]] = []
    row_y: list[float] = []
    row_heights: list[float] = []
    for y_center, x1, height, text in positioned:
        if (
            not rows
            or abs(y_center - row_y[-1]) > max(12.0, row_heights[-1] * 0.7)
        ):
            rows.append([(x1, text)])
            row_y.append(y_center)
            row_heights.append(height)
        else:
            rows[-1].append((x1, text))
            count = len(rows[-1])
            row_y[-1] = ((row_y[-1] * (count - 1)) + y_center) / count
            row_heights[-1] = max(row_heights[-1], height)

    markdown_rows: list[tuple[str, str]] = []
    for row in rows:
        values = [text for _, text in sorted(row)]
        if len(values) == 1:
            markdown_rows.append(("文字", values[0]))
            continue
        for index in range(0, len(values), 2):
            field = values[index]
            content = (
                values[index + 1]
                if index + 1 < len(values)
                else "[无法识别]"
            )
            markdown_rows.append((field, content))
    markdown_rows.extend(("文字", text) for text in unpositioned)
    return _markdown_table(markdown_rows)


def _spotting_value(value: Any) -> Any:
    """Parse JSON or Python-literal spotting arrays without executing code."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(
            r"^```(?:json|python)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        )
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def _lines_markdown(lines: list[str]) -> str:
    return _markdown_table(
        ("文字", line.strip()) for line in lines if line.strip()
    )


def _markdown_table(rows: Any) -> str:
    rendered = ["| 字段 | 内容 |", "| --- | --- |"]
    for field, content in rows:
        safe_field = str(field).replace("|", "\\|").replace("\n", "<br>")
        safe_content = str(content).replace("|", "\\|").replace("\n", "<br>")
        rendered.append(f"| {safe_field} | {safe_content} |")
    return "\n".join(rendered)


def json_like_text(value: Any) -> str:
    """Flatten a JSON-compatible value without depending on JSON rendering."""

    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {json_like_text(item)}" for key, item in value.items()
        )
    if isinstance(value, list):
        return " ".join(json_like_text(item) for item in value)
    return str(value or "")


def normalize_document_page(
    profile: ExtractionProfile,
    value: Any,
) -> list[dict[str, str]]:
    """Validate one non-ID HunyuanOCR structured page response."""

    if contains_spotting_coordinates(value):
        raise ValueError("HunyuanOCR 误返回文字坐标")
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        raw_records: Any = value["records"]
    elif isinstance(value, Mapping):
        raw_records = [value]
    elif isinstance(value, list):
        raw_records = value
    else:
        raise ValueError("HunyuanOCR 证件输出不是 JSON 对象")

    aliases = _DOCUMENT_ALIASES.get(profile.key, {})
    fields = set(profile.field_names)
    candidates = [
        {
            aliases.get(str(key), str(key)): item_value
            for key, item_value in item.items()
        }
        for item in raw_records
        if isinstance(item, Mapping)
        and (fields | set(aliases)).intersection(str(key) for key in item)
    ]
    if not candidates:
        raise ValueError(
            "HunyuanOCR 未返回目标证件字段，可能误输出了文字坐标"
        )
    records = profile.normalize_records(candidates)
    if not any(any(record.values()) for record in records):
        raise ValueError("HunyuanOCR 结构化证件结果为空")
    return sanitize_document_records(profile, records)


def sanitize_document_records(
    profile: ExtractionProfile,
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Remove field labels and impossible identifiers from OCR-derived JSON."""

    sanitized = [dict(record) for record in records]
    for record in sanitized:
        for field, value in tuple(record.items()):
            compact_value = re.sub(r"\s+", "", str(value or ""))
            if compact_value in _FIELD_LABEL_VALUES:
                record[field] = ""
        if profile.key == "driver_license":
            identity = re.sub(r"\s+", "", record.get("证号", "")).upper()
            if not (
                15 <= len(identity) <= 20
                and sum(char.isdigit() for char in identity) >= 6
                and re.fullmatch(r"[0-9A-Z*]+", identity)
            ):
                identity = ""
            record["证号"] = identity
            nationality = record.get("国籍", "")
            if nationality and nationality != "中国":
                record["国籍"] = ""
            archive = re.sub(
                r"\s+", "", record.get("档案编号", "")
            ).upper()
            if archive and not re.fullmatch(r"[0-9*]{12}", archive):
                archive = ""
            record["档案编号"] = archive
        elif profile.key == "vehicle_license":
            vin = re.sub(
                r"\s+", "", record.get("车辆识别代号", "")
            ).upper()
            if vin and not re.fullmatch(r"[0-9A-HJ-NPR-Z*]{17}", vin):
                vin = ""
            record["车辆识别代号"] = vin
    return sanitized


def _merge_document_group(
    profile: ExtractionProfile,
    records: list[dict[str, str]],
) -> dict[str, str]:
    merged = {
        field: _preferred_value(
            field,
            [record.get(field, "") for record in records],
        )
        for field in profile.field_names
    }
    if profile.key == "vehicle_license":
        merged["证件类型"] = _vehicle_license_type_from_evidence(records)
    return merged


_VEHICLE_BACK_PAGE_FIELDS = (
    "档案号码",
    "核定载人数",
    "总质量",
    "整备质量",
    "外廓尺寸",
)


def _vehicle_license_type_from_evidence(
    records: list[dict[str, str]],
) -> str:
    """Classify merged vehicle pages from title and back-page field evidence."""

    type_values = [record.get("证件类型", "") for record in records]
    has_front = any(
        "正本" in value or ("行驶证" in value and "副页" not in value)
        for value in type_values
    )
    has_back = any("副页" in value for value in type_values) or any(
        record.get(field, "")
        for record in records
        for field in _VEHICLE_BACK_PAGE_FIELDS
    )
    if has_front and has_back:
        return "中国行驶证正本与副页"
    if has_front:
        return "中国行驶证正本"
    if has_back:
        return "中国行驶证副页"
    return ""


def _masked_driver_identity_match(
    left: str,
    right: str,
    left_records: list[dict[str, str]],
    right_records: list[dict[str, str]],
) -> bool:
    """Match a masked electronic ID only when another stable field agrees."""

    if len(left) != len(right) or "*" not in left + right:
        return False
    if any(a != b and a != "*" and b != "*" for a, b in zip(left, right)):
        return False
    left_names = {record.get("姓名", "") for record in left_records} - {""}
    right_names = {record.get("姓名", "") for record in right_records} - {""}
    left_archives = {record.get("档案编号", "") for record in left_records} - {""}
    right_archives = {record.get("档案编号", "") for record in right_records} - {""}
    return bool(
        left_names.intersection(right_names)
        or left_archives.intersection(right_archives)
    )


def _document_identity_groups(
    profile: ExtractionProfile,
    page_records: list[dict[str, str]],
    identity_field: str,
    identities: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Return ordered canonical identities and a source-to-canonical map."""

    clusters: list[list[str]] = []
    records_by_identity = {
        identity: [
            record
            for record in page_records
            if record.get(identity_field) == identity
        ]
        for identity in identities
    }
    for identity in identities:
        target: list[str] | None = None
        if profile.key == "driver_license":
            for cluster in clusters:
                if any(
                    _masked_driver_identity_match(
                        identity,
                        existing,
                        records_by_identity[identity],
                        records_by_identity[existing],
                    )
                    for existing in cluster
                ):
                    target = cluster
                    break
        if target is None:
            clusters.append([identity])
        else:
            target.append(identity)

    canonical_identities: list[str] = []
    aliases: dict[str, str] = {}
    for cluster in clusters:
        canonical = min(
            cluster,
            key=lambda value: (value.count("*"), identities.index(value)),
        )
        canonical_identities.append(canonical)
        aliases.update({identity: canonical for identity in cluster})
    return canonical_identities, aliases


def merge_document_pages(
    profile: ExtractionProfile,
    page_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge front/back page JSON while keeping distinct subjects separate."""

    if not page_records:
        raise ValueError("HunyuanOCR 没有返回可合并的证件记录")
    identity_field = (
        "证号" if profile.key == "driver_license"
        else "车辆识别代号" if profile.key == "vehicle_license"
        else ""
    )
    if not identity_field:
        return [_merge_document_group(profile, page_records)]
    identities = list(
        dict.fromkeys(
            record.get(identity_field, "")
            for record in page_records
            if record.get(identity_field)
        )
    )
    if len(identities) <= 1:
        return [_merge_document_group(profile, page_records)]

    identities, identity_aliases = _document_identity_groups(
        profile,
        page_records,
        identity_field,
        identities,
    )
    groups = {identity: [] for identity in identities}
    for record in page_records:
        source_identity = record.get(identity_field, "")
        if source_identity:
            groups[identity_aliases[source_identity]].append(record)
    # Identity-less pages can only be attached when another stable field makes
    # the owner unambiguous. Otherwise leave them unassigned instead of mixing
    # two people's/vehicles' back-page values.
    secondary_field = "姓名" if profile.key == "driver_license" else "号牌号码"
    secondary_values = {
        identity: {
            record.get(secondary_field, "")
            for record in records
            if record.get(secondary_field)
        }
        for identity, records in groups.items()
    }
    for record in page_records:
        if record.get(identity_field):
            continue
        value = record.get(secondary_field, "")
        matches = [
            identity
            for identity, values in secondary_values.items()
            if value and value in values
        ]
        if len(matches) == 1:
            groups[matches[0]].append(record)
    return [
        _merge_document_group(profile, groups[identity])
        for identity in identities
    ]


def id_card_side(image_path: Path) -> str:
    """Resolve the configured ID-card side from its DG12/DG13 folder."""

    parts = {part.upper() for part in image_path.parts}
    if "DG12" in parts:
        return "front"
    if "DG13" in parts:
        return "back"
    return ""


def normalize_id_card_page(
    profile: ExtractionProfile,
    value: Any,
    side: str = "",
) -> list[dict[str, str]]:
    """Validate and normalize one HunyuanOCR per-image JSON response."""

    raw_records: Any
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        raw_records = value["records"]
    elif isinstance(value, Mapping):
        raw_records = [value]
    elif isinstance(value, list):
        raw_records = value
    else:
        raise ValueError("HunyuanOCR 身份证输出不是 JSON 对象")

    field_names = set(profile.field_names)
    candidates = [
        {
            _ID_CARD_ALIASES.get(str(key), str(key)): item_value
            for key, item_value in item.items()
        }
        for item in raw_records
        if isinstance(item, Mapping)
        and (field_names | set(_ID_CARD_ALIASES)).intersection(item)
    ]
    if not candidates:
        raise ValueError(
            "HunyuanOCR 未返回身份证字段，可能误输出了文字坐标"
        )

    records = profile.normalize_records(candidates)
    for record in records:
        if side == "front":
            record["签发机关"] = ""
            record["有效期"] = ""
        elif side == "back":
            for field in ID_CARD_FRONT_FIELDS[1:]:
                record[field] = ""
        document_type = record.get("证件类型", "")
        if "身份证" in document_type or any(record.values()):
            record["证件类型"] = "第二代身份证"
        identity = record.get("身份证号", "")
        if identity:
            record["出生日期"] = identity[6:14]
            if not record.get("性别"):
                record["性别"] = "男" if int(identity[16]) % 2 else "女"
    if side == "back" and not any(
        record.get("签发机关") or record.get("有效期")
        for record in records
    ):
        raise ValueError("HunyuanOCR 身份证背面 JSON 未包含背面字段")
    if side == "front" and not any(
        any(record.get(field) for field in ID_CARD_FRONT_FIELDS[1:])
        for record in records
    ):
        raise ValueError("HunyuanOCR 身份证正面 JSON 未包含正面字段")
    return records


def _preferred_value(field: str, values: list[str]) -> str:
    nonempty = [value for value in values if value]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    first_position = {value: nonempty.index(value) for value in counts}
    # Repeated readings are the strongest signal. For equally frequent address
    # or authority candidates retain the more complete value, then input order.
    prefer_length = field in {"住址", "签发机关"}
    return max(
        counts,
        key=lambda value: (
            counts[value],
            -value.count("*"),
            len(value) if prefer_length else 0,
            -first_position[value],
        ),
    )


def _merge_group(
    profile: ExtractionProfile,
    records: list[dict[str, str]],
) -> dict[str, str]:
    merged = {
        field: _preferred_value(field, [record.get(field, "") for record in records])
        for field in profile.field_names
    }
    if any(merged.values()):
        merged["证件类型"] = "第二代身份证"
    identity = merged.get("身份证号", "")
    if identity:
        merged["出生日期"] = identity[6:14]
        if not merged.get("性别"):
            merged["性别"] = "男" if int(identity[16]) % 2 else "女"
    return merged


def merge_id_card_pages(
    profile: ExtractionProfile,
    page_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge duplicate/front/back page JSON without crossing card holders."""

    if not page_records:
        raise ValueError("HunyuanOCR 没有返回可合并的身份证记录")

    front_fields = ("姓名", "身份证号", "出生日期", "性别", "民族", "住址")
    fronts = [
        record for record in page_records if any(record.get(field) for field in front_fields)
    ]
    backs_by_index: dict[str, list[dict[str, str]]] = {}
    for record in page_records:
        if record.get("__hunyuan_side") != "back":
            continue
        backs_by_index.setdefault(
            record.get("__hunyuan_side_index", "-1"),
            [],
        ).append(record)
    identities = list(
        dict.fromkeys(record["身份证号"] for record in fronts if record.get("身份证号"))
    )
    if len(identities) == 1:
        return [_merge_group(profile, page_records)]

    if identities:
        groups: dict[str, list[dict[str, str]]] = {
            identity: [
                record
                for record in fronts
                if record.get("身份证号") == identity
            ]
            for identity in identities
        }
        names = {
            identity: {
                record.get("姓名", "")
                for record in records
                if record.get("姓名")
            }
            for identity, records in groups.items()
        }
        for record in fronts:
            if record.get("身份证号"):
                matches = [record["身份证号"]]
            else:
                matches = [
                    identity
                    for identity, group_names in names.items()
                    if record.get("姓名") in group_names
                ]
                if len(matches) == 1:
                    groups[matches[0]].append(record)
            if len(matches) == 1:
                groups[matches[0]].extend(
                    backs_by_index.get(
                        record.get("__hunyuan_side_index", "-1"),
                        [],
                    )
                )
        return [_merge_group(profile, groups[identity]) for identity in identities]

    names = list(
        dict.fromkeys(record["姓名"] for record in fronts if record.get("姓名"))
    )
    if len(names) <= 1:
        return [_merge_group(profile, page_records)]
    return [
        _merge_group(
            profile,
            [record for record in fronts if record.get("姓名") == name],
        )
        for name in names
    ]
