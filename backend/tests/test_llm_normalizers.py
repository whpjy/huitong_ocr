from __future__ import annotations

from llm_manager.normalizers import ID_CARD_ETHNICITIES, normalize_value


EMPTY = ("", "null", "none", "unknown", "未知", "未识别", "无", "-", "--")


def normalized(name: str, value: object) -> str:
    return normalize_value(name, value, empty_values=EMPTY)


def test_common_document_normalizers() -> None:
    assert normalized("date_yyyymmdd", "2026年7月3日") == "20260703"
    assert normalized("date_yyyy_mm_dd", "2026/7/3") == "2026-07-03"
    assert normalized("date_range", "2026.7.3至2036.7.3") == (
        "20260703-20360703"
    )
    assert normalized(
        "id_card_valid_period",
        "2024.08.28-2044.08.28",
    ) == "20240828-20440828"
    assert normalized("id_card_valid_period", "2024年8月") == "2024年8月"
    assert normalized("identity_number", "110101199001011237") == (
        "110101199001011237"
    )
    assert normalized("mass_kg", "1,730 KG") == "1730kg"
    assert normalized("dimensions_mm", "4490×1860×1590") == (
        "4490×1860×1590mm"
    )
    assert normalized("money", "￥1,234.50元") == "1234.50"
    assert normalized("percent", "13") == "13%"


def test_invalid_or_empty_values_become_empty_strings() -> None:
    assert normalized("identity_number", "123") == ""
    assert normalized("identity_number", "511122200112164842") == ""
    assert normalized("identity_number", "511112200112164842") == (
        "511112200112164842"
    )
    assert normalized("date_yyyymmdd", "2026年7月") == ""
    assert normalized("stamp_yes_no", "有印章但颜色未知") == ""
    assert normalized("text", "unknown") == ""


def test_ethnicity_dictionary_contains_all_standard_values_and_chuanqing() -> None:
    assert len(ID_CARD_ETHNICITIES) == 57
    assert "汉" in ID_CARD_ETHNICITIES
    assert "基诺" in ID_CARD_ETHNICITIES
    assert "穿青人" in ID_CARD_ETHNICITIES


def test_ethnicity_normalizer_validates_dictionary_values() -> None:
    assert normalized("ethnicity", "维吾尔族") == "维吾尔"
    assert normalized("ethnicity", " 蒙古族 ") == "蒙古"
    assert normalized("ethnicity", "穿青") == "穿青人"
    assert normalized("ethnicity", "穿青人") == "穿青人"
    assert normalized("ethnicity", "窜青") == ""
    assert normalized("ethnicity", "未知民族") == ""
