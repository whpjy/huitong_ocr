"""Convert a vision model's labelled document text into profile fields."""

from __future__ import annotations

import re
from typing import Any

from .driver_refiner import refine_driver_fields
from .id_card_refiner import refine_id_card_fields
from .profiles import ExtractionProfile
from .vehicle_refiner import refine_vehicle_fields


_DATE = r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?:日)?"


def _value_after_label(
    text: str,
    labels: tuple[str, ...],
    stop_labels: tuple[str, ...],
) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    stop_pattern = "|".join(re.escape(label) for label in stop_labels)
    match = re.search(
        rf"(?:{label_pattern})\s*[:：]?\s*(.+?)(?="
        rf"\s+(?:{stop_pattern})\s*[:：]?|\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" ：:，,。") if match else ""


def _generic_fields(profile: ExtractionProfile, text: str) -> dict[str, str]:
    fields = {name: "" for name in profile.field_names}
    stops = tuple(profile.field_names)
    for name in profile.field_names:
        fields[name] = _value_after_label(text, (name,), stops)
    return fields


def _driver_fields(profile: ExtractionProfile, text: str) -> dict[str, str]:
    fields = _generic_fields(profile, text)
    stops = tuple(profile.field_names) + (
        "Address",
        "Class",
        "Date of Birth",
        "Date of First Issue",
        "Valid Period",
        "记录",
    )
    aliases = {
        "住址": ("住址", "Address"),
        "出生日期": ("出生日期", "Date of Birth"),
        "准驾车型": ("准驾车型", "Class"),
        "国籍": ("国籍", "Nationality"),
        "证号": ("证号", "Certificate No", "Document No"),
        "初次领证日期": ("初次领证日期", "Date of First Issue"),
        "姓名": ("姓名", "Name"),
        "性别": ("性别", "Sex"),
        "档案编号": ("档案编号", "档案号码", "File Number"),
    }
    for field, labels in aliases.items():
        value = _value_after_label(text, labels, stops)
        if value:
            fields[field] = value
    valid_window = "\n".join(
        line for line in text.splitlines() if "有效期限" in line or "Valid Period" in line
    )
    dates = re.findall(_DATE, valid_window)
    if len(dates) >= 2:
        fields["有效期限"] = f"{dates[0]}-{dates[1]}"
    normalized = profile.normalize_records({"records": [fields]})[0]
    return refine_driver_fields(text, normalized)


def _id_card_fields(profile: ExtractionProfile, text: str) -> dict[str, str]:
    fields = _generic_fields(profile, text)
    stops = tuple(profile.field_names) + (
        "公民身份号码",
        "签发机关",
        "有效期限",
        "Address",
    )
    aliases = {
        "姓名": ("姓名", "Name"),
        "身份证号": ("公民身份号码", "公民身份证号码", "身份证号"),
        "出生日期": ("出生", "出生日期"),
        "性别": ("性别", "Sex"),
        "民族": ("民族",),
        "住址": ("住址", "Address"),
        # HunyuanOCR 偶尔把“签发机关”断成“签发机”与下一行的值。
        "签发机关": ("签发机关", "签发机"),
        "有效期": ("有效期限", "有效期"),
    }
    for field, labels in aliases.items():
        value = _value_after_label(text, labels, stops)
        if value:
            fields[field] = value
    if "居民身份证" in re.sub(r"\s+", "", text):
        fields["证件类型"] = "第二代身份证"
    normalized = profile.normalize_records({"records": [fields]})[0]
    return refine_id_card_fields(text, normalized)


def adapt_profile_text(
    profile: ExtractionProfile,
    text: str,
) -> list[dict[str, str]]:
    """Map labelled model text without invoking OCR or another LLM."""

    if profile.key == "driver_license":
        return [_driver_fields(profile, text)]
    if profile.key == "id_card":
        return [_id_card_fields(profile, text)]
    fields: dict[str, Any] = _generic_fields(profile, text)
    normalized = profile.normalize_records({"records": [fields]})[0]
    if profile.key == "vehicle_license":
        normalized = refine_vehicle_fields(text, normalized)
    return [normalized]
