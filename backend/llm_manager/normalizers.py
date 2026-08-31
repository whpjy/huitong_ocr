"""Reusable field normalizers selected by extraction profile YAML."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any


Normalizer = Callable[[str], str]


# 身份证民族字段使用的 56 个民族规范简称；证件上实际还可能登记
# “穿青人”，该名称中的“人”不可作为后缀删除。
ID_CARD_ETHNICITIES = frozenset(
    {
        "汉",
        "蒙古",
        "回",
        "藏",
        "维吾尔",
        "苗",
        "彝",
        "壮",
        "布依",
        "朝鲜",
        "满",
        "侗",
        "瑶",
        "白",
        "土家",
        "哈尼",
        "哈萨克",
        "傣",
        "黎",
        "傈僳",
        "佤",
        "畲",
        "高山",
        "拉祜",
        "水",
        "东乡",
        "纳西",
        "景颇",
        "柯尔克孜",
        "土",
        "达斡尔",
        "仫佬",
        "羌",
        "布朗",
        "撒拉",
        "毛南",
        "仡佬",
        "锡伯",
        "阿昌",
        "普米",
        "塔吉克",
        "怒",
        "乌孜别克",
        "俄罗斯",
        "鄂温克",
        "德昂",
        "保安",
        "裕固",
        "京",
        "塔塔尔",
        "独龙",
        "鄂伦春",
        "赫哲",
        "门巴",
        "珞巴",
        "基诺",
        "穿青人",
    }
)

ETHNICITY_ALIASES = {
    "穿青": "穿青人",
    "穿青族": "穿青人",
}


def _compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _compact(value: str) -> str:
    return re.sub(r"[\s_]+", "", value)


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _date_match(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"\s*(\d{4})[年./-]?(\d{1,2})[月./-]?(\d{1,2})(?:日)?\s*",
        value,
    )
    return match.groups() if match else None


def _date_yyyymmdd(value: str) -> str:
    parts = _date_match(value)
    if not parts:
        return ""
    year, month, day = parts
    return f"{year}{int(month):02d}{int(day):02d}"


def _date_yyyy_mm_dd(value: str) -> str:
    parts = _date_match(value)
    if not parts:
        return ""
    year, month, day = parts
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _date_digits(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 8 else ""


def _date_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(
        r"(?<!\d)(\d{4})[年./-]?(\d{1,2})[月./-]?(\d{1,2})(?:日)?(?!\d)",
        value,
    ):
        year, month, day = match.groups()
        tokens.append(f"{year}{int(month):02d}{int(day):02d}")
    return tokens


def _date_range(value: str) -> str:
    dates = _date_tokens(value)
    if "长期" in value and dates:
        return f"{dates[0]}-长期"
    if len(dates) >= 2:
        return f"{dates[0]}-{dates[1]}"
    compact = re.fullmatch(r"(\d{8})\s*[-至]\s*(\d{8})", value)
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}"
    long_term = re.fullmatch(r"(\d{8})\s*[-至]\s*长期", value)
    if long_term:
        return f"{long_term.group(1)}-长期"
    return ""


def _id_card_valid_period(value: str) -> str:
    """Normalize complete ID validity ranges without erasing partial text."""

    normalized = _date_range(value)
    return normalized or _compact_whitespace(value)


def _duration(value: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*年\s*", value)
    if match:
        return f"{match.group(1)}年"
    return "长期" if value == "长期" else ""


def _gender(value: str) -> str:
    return value if value in {"男", "女"} else ""


def _ethnicity(value: str) -> str:
    compact = _compact_whitespace(value)
    aliased = ETHNICITY_ALIASES.get(compact, compact)
    candidate = aliased[:-1] if aliased.endswith("族") else aliased
    return candidate if candidate in ID_CARD_ETHNICITIES else ""


def _identity_number(value: str) -> str:
    compact = _compact_whitespace(value).upper()
    if not re.fullmatch(r"[1-9]\d{16}[\dX]", compact):
        return ""
    try:
        datetime.strptime(compact[6:14], "%Y%m%d")
    except ValueError:
        return ""
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    checksum = sum(
        int(digit) * weight
        for digit, weight in zip(compact[:17], weights)
    )
    return compact if compact[-1] == checks[checksum % 11] else ""


def _driving_class(value: str) -> str:
    return re.sub(r"[\s,，、]+", "", value).upper()


def _nationality(value: str) -> str:
    compact = _compact_whitespace(value)
    return (
        "中国"
        if compact.upper() in {"中国", "中国/CHN", "CHN", "中华人民共和国"}
        else compact
    )


def _driver_document_type(value: str) -> str:
    if "电子" in value and "驾驶证" in value:
        return "电子驾驶证"
    if "驾驶证" in value:
        return "中华人民共和国机动车驾驶证"
    return ""


def _inspection_date(value: str) -> str:
    full_date = _date_yyyymmdd(value)
    if full_date:
        return full_date
    match = re.fullmatch(r"\s*(\d{4})[年./-]?(\d{1,2})(?:月)?\s*", value)
    if not match:
        return ""
    year, month = match.groups()
    return f"{year}{int(month):02d}"


def _mass_kg(value: str) -> str:
    text = value.replace(",", "").replace("，", "").strip()
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?:kg|千克|公斤)?",
        text,
        flags=re.IGNORECASE,
    )
    return f"{match.group(1)}kg" if match else ""


def _people(value: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*(?:人)?\s*", value)
    return f"{match.group(1)}人" if match else ""


def _dimensions_mm(value: str) -> str:
    numbers = [
        _dimension_integer(match.group(0))
        for match in re.finditer(r"\d+(?:\.\d+)?", value)
    ]
    return f"{'×'.join(numbers)}mm" if len(numbers) == 3 else ""


def _dimension_integer(value: str) -> str:
    """Repair OCR decimal-point noise in integer millimetre dimensions."""

    if "." not in value:
        return value
    integer, fraction = value.split(".", 1)
    if set(fraction) <= {"0"}:
        return integer
    # Vehicle dimensions expressed in millimetres are integers. OCR commonly
    # reads 4762 as 47.62; join the fragments when the integer part is short.
    if len(integer) < 4 and 3 <= len(integer + fraction) <= 5:
        return integer + fraction
    return value


def _vehicle_license_type(value: str) -> str:
    if "正本" in value and "副页" in value:
        return "中国行驶证正本与副页"
    if "副页" in value:
        return "中国行驶证副页"
    if "正本" in value or "行驶证" in value:
        return "中国行驶证正本"
    return ""


def _number(value: str) -> str:
    text = value.strip().replace(",", "").replace("，", "")
    text = re.sub(
        r"\s*(?:ml|kw|mm|kg|cm3|cm³|cc|人|个|只|辆|片)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text if re.fullmatch(r"\d+(?:\.\d+)?", text) else ""


def _dimensions_slash(value: str) -> str:
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    return "/".join(numbers) if len(numbers) == 3 else ""


def _domestic_imported(value: str) -> str:
    if "进口" in value:
        return "进口"
    if "国产" in value:
        return "国产"
    return ""


def _money(value: str) -> str:
    text = value.strip().replace(",", "").replace("，", "")
    text = re.sub(
        r"^(?:小写|人民币|RMB|CNY|[¥￥])\s*[:：]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*(?:元|圆)\s*$", "", text)
    match = re.fullmatch(r"-?\d+(?:\.\d{1,2})?", text)
    return match.group(0) if match else ""


def _numeric(value: str) -> str:
    text = value.strip().replace(",", "").replace("，", "")
    text = re.sub(
        r"\s*(?:人|吨|t)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text if re.fullmatch(r"\d+(?:\.\d+)?", text) else ""


def _percent(value: str) -> str:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", value)
    return f"{match.group(1)}%" if match else ""


def _invoice_type(value: str) -> str:
    compact = _compact_whitespace(value)
    if "电子" in compact and "机动车" in compact and "发票" in compact:
        return "电子发票（机动车销售统一发票）"
    if "机动车" in compact and "发票" in compact:
        return "机动车销售统一发票"
    return compact


def _business_term(value: str) -> str:
    text = _compact_whitespace(value)
    if "长期" in text:
        return "长期"
    dates = re.findall(
        r"(\d{4})[年./-]?(\d{1,2})[月./-]?(\d{1,2})(?:日)?",
        text,
    )
    if len(dates) != 2:
        return ""
    normalized = [
        f"{year}{int(month):02d}{int(day):02d}"
        for year, month, day in dates
    ]
    return f"{normalized[0]}至{normalized[1]}"


def _premium_list(value: str) -> str:
    parts = re.split(r"[；;\r\n]", value)
    cleaned: list[str] = []
    for part in parts:
        if not part.strip():
            cleaned.append("")
            continue
        money = _money(part)
        if not money:
            return ""
        cleaned.append(money)
    return "；".join(cleaned)


def _stamp_yes_no(value: str) -> str:
    compact = _compact_whitespace(value)
    if compact in {"是", "有", "红章", "红色", "红色印章"}:
        return "是"
    if compact in {"否", "非红章", "不是红章", "非红色印章"}:
        return "否"
    return ""


def _measurement(value: str) -> str:
    text = value.strip().replace(",", "").replace("，", "")
    text = re.sub(
        r"\s*(?:mm|毫米|kg|千克|km/h|公里/小时)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else ""


NORMALIZERS: dict[str, Normalizer] = {
    "text": lambda value: value,
    "space": _space,
    "compact": _compact,
    "compact_whitespace": _compact_whitespace,
    "compact_upper": lambda value: _compact_whitespace(value).upper(),
    "date_yyyymmdd": _date_yyyymmdd,
    "date_yyyy_mm_dd": _date_yyyy_mm_dd,
    "date_digits": _date_digits,
    "date_range": _date_range,
    "id_card_valid_period": _id_card_valid_period,
    "duration": _duration,
    "gender": _gender,
    "ethnicity": _ethnicity,
    "identity_number": _identity_number,
    "driving_class": _driving_class,
    "nationality": _nationality,
    "driver_document_type": _driver_document_type,
    "inspection_date": _inspection_date,
    "mass_kg": _mass_kg,
    "people": _people,
    "dimensions_mm": _dimensions_mm,
    "vehicle_license_type": _vehicle_license_type,
    "number": _number,
    "dimensions_slash": _dimensions_slash,
    "domestic_imported": _domestic_imported,
    "money": _money,
    "numeric": _numeric,
    "percent": _percent,
    "invoice_type": _invoice_type,
    "business_term": _business_term,
    "premium_list": _premium_list,
    "stamp_yes_no": _stamp_yes_no,
    "measurement": _measurement,
}


def has_normalizer(name: str) -> bool:
    return name in NORMALIZERS


def normalize_value(
    normalizer: str,
    value: Any,
    *,
    empty_values: Iterable[str],
) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    configured = set(empty_values)
    lowered = {item.lower() for item in configured}
    if text in configured or text.lower() in lowered:
        return ""
    try:
        function = NORMALIZERS[normalizer]
    except KeyError as exc:
        raise ValueError(f"未知字段清洗器：{normalizer}") from exc
    return function(text)
