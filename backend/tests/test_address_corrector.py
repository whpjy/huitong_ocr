import pytest

from llm_manager.address_corrector import (
    clean_address_number_ocr,
    correct_address_divisions,
)
from llm_manager.driver_refiner import refine_driver_fields
from llm_manager.id_card_refiner import refine_id_card_fields
from llm_manager.vehicle_refiner import refine_vehicle_fields


def test_corrects_unique_hierarchical_ocr_substitutions() -> None:
    assert (
        correct_address_divisions("广东省惠洲市惠城曲麦地路10号")
        == "广东省惠州市惠城区麦地路10号"
    )


def test_corrects_county_when_prefecture_is_omitted() -> None:
    assert (
        correct_address_divisions("湖南省武岗市湾头镇八合村16组")
        == "湖南省武冈市湾头镇八合村16组"
    )


def test_accepts_historical_division_without_modifying_it() -> None:
    assert (
        correct_address_divisions("北京市崇文区幸福街8号")
        == "北京市崇文区幸福街8号"
    )


def test_does_not_touch_detailed_address_and_preserves_masking() -> None:
    assert (
        correct_address_divisions("广东省惠州市惠城区麦堆路10号")
        == "广东省惠州市惠城区麦堆路10号"
    )
    assert (
        correct_address_divisions("广东省惠洲市惠城区麦地路**号")
        == "广东省惠州市惠城区麦地路**号"
    )
    assert (
        correct_address_divisions("广东省惠*市惠城区麦地路10号")
        == "广东省惠*市惠城区麦地路10号"
    )


def test_all_three_document_refiners_apply_address_correction() -> None:
    typo = "广东省惠洲市惠城区麦地路10号"

    identity = refine_id_card_fields("", {"住址": typo})
    driver = refine_driver_fields("", {"住址": typo})
    vehicle = refine_vehicle_fields("", {"住址": typo})

    expected = "广东省惠州市惠城区麦地路10号"
    assert identity["住址"] == expected
    assert driver["住址"] == expected
    assert vehicle["住址"] == expected


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ("广东省惠州市惠城区幸福村【3】号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村[3]号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村〔3〕号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村（3）号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村1-3-1号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村1／3／1号", "广东省惠州市惠城区幸福村131号"),
        ("广东省惠州市惠城区幸福村1 · 3 · 1号", "广东省惠州市惠城区幸福村131号"),
        ("幸福路l3I号", "幸福路131号"),
        ("幸福路2O1号12栋3O2室", "幸福路201号12栋302室"),
        ("幸福路S1号B栋", "幸福路51号B栋"),
        ("幸福路G6号", "幸福路66号"),
    ],
)
def test_cleans_high_confidence_address_number_ocr(
    observed: str,
    expected: str,
) -> None:
    assert clean_address_number_ocr(observed) == expected


def test_address_number_cleaning_is_limited_to_explicit_number_positions() -> None:
    assert clean_address_number_ocr("北京市SOHO现代城") == "北京市SOHO现代城"
    assert clean_address_number_ocr("幸福路1-23-1号") == "幸福路1-23-1号"
    assert clean_address_number_ocr("幸福路3一2一1号") == "幸福路3一2一1号"


@pytest.mark.parametrize(
    "refiner",
    [refine_id_card_fields, refine_driver_fields, refine_vehicle_fields],
)
def test_all_three_document_refiners_clean_house_number_ocr(refiner) -> None:
    refined = refiner("", {"住址": "广东省惠州市惠城区幸福村【3】号"})
    assert refined["住址"] == "广东省惠州市惠城区幸福村131号"
