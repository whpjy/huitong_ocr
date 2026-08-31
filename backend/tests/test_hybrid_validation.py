from pathlib import Path

import pytest
from PIL import Image

from llm_manager.hybrid_validation import (
    arbitrate,
    arbitrate_fast,
    compare_fields,
    compare_fields_fast,
    create_recheck_images,
    find_evidence,
    missing_field_supplements,
    parse_recheck_value,
    recheck_conflict,
    valid_fast_field_value,
)
from llm_manager.hunyuan_ocr_prompts import local_recheck_prompt
from llm_manager.hybrid_pipeline import fuse_hybrid_result
from llm_manager.models import ExtractionResult
from llm_manager.profiles import load_profiles


def test_compare_fields_finds_single_character_conflict() -> None:
    tokens = [
        {
            "text": "广西钦州市钦南区久隆镇青草村委那珠村1号",
            "score": 0.97,
            "bbox": [100, 200, 900, 260],
        }
    ]

    result = compare_fields(
        {"住址": "广西钦州市钦南区久隆镇青草村委那殊村1号"},
        tokens,
    )[0]

    assert result["status"] == "conflict"
    assert result["ppocr_value"] == "广西钦州市钦南区久隆镇青草村委那珠村1号"
    assert result["ppocr_confidence"] == 0.97
    assert result["bbox"] == [100.0, 200.0, 900.0, 260.0]


def test_compare_fields_ignores_english_label_inside_wrapped_address() -> None:
    tokens = [
        {"text": "广西钦州市钦南区久隆镇青草村委那珠村", "score": 0.98, "bbox": [100, 200, 850, 250]},
        {"text": "Address", "score": 0.99, "bbox": [20, 245, 90, 270]},
        {"text": "1号", "score": 0.99, "bbox": [100, 255, 160, 290]},
    ]

    result = compare_fields(
        {"住址": "广西钦州市钦南区久隆镇青草村委那殊村1号"},
        tokens,
    )[0]

    assert result["status"] == "conflict"
    assert result["ppocr_value"] == "广西钦州市钦南区久隆镇青草村委那珠村1号"


def test_compare_fields_normalizes_long_term_date_range_before_conflict() -> None:
    tokens = [
        {
            "text": "2025-04-03至长期",
            "score": 0.99,
            "bbox": [100, 200, 400, 250],
        }
    ]

    result = compare_fields({"有效期限": "20250403-长期"}, tokens)[0]

    assert result["status"] == "consistent"


def test_fast_compare_requires_label_spatial_evidence() -> None:
    unrelated = [
        {"text": "鲁A12345", "score": 0.999, "bbox": [500, 300, 650, 330]},
    ]
    labeled = [
        {"text": "号牌号码", "score": 0.99, "bbox": [10, 10, 90, 30]},
        {"text": "鲁A12345", "score": 0.99, "bbox": [110, 10, 220, 30]},
    ]

    assert compare_fields_fast({"车牌号码": "鲁A1234S"}, unrelated)[0][
        "status"
    ] == "ppocr_no_labeled_evidence"
    comparison = compare_fields_fast(
        {"车牌号码": "鲁A1234S"}, labeled
    )[0]
    assert comparison["status"] == "conflict"
    assert comparison["association"] == "label_spatial"
    assert comparison["ppocr_value"] == "鲁A12345"


def test_fast_arbitration_uses_only_valid_high_confidence_strict_fields() -> None:
    comparison = {
        "field": "车牌号码",
        "multimodal_value": "鲁A1234S",
        "ppocr_value": "鲁A12345",
        "association": "label_spatial",
        "association_unique": True,
        "label_confidence": 0.99,
        "ppocr_confidence": 0.99,
    }

    final, _reason, changed = arbitrate_fast(comparison, {})

    assert final == "鲁A12345"
    assert changed is True
    assert valid_fast_field_value("车牌号码", final)


def test_fast_arbitration_keeps_hunyuan_for_semantic_text_conflict() -> None:
    final, reason, changed = arbitrate_fast(
        {
            "field": "车辆类型",
            "multimodal_value": "小型普通客车",
            "ppocr_value": "小型轿车",
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 0.999,
            "ppocr_confidence": 0.999,
        },
        {},
    )

    assert final == "小型普通客车"
    assert changed is False
    assert "语义字段" in reason


@pytest.mark.parametrize(
    "address",
    [
        "云南省大理白族自治州洱源县三营镇士登村民委员会义常村13号",
        "广东省中山市古镇镇教昌外闸北三巷1号",
        "福建省晋江市英林镇龙西村东垵11号",
    ],
)
def test_fast_address_validation_accepts_three_certificate_addresses(
    address: str,
) -> None:
    assert valid_fast_field_value("住址", address)


def test_fast_arbitration_allows_high_confidence_address_override() -> None:
    initial = "云南省大理白族自治州洱源县三营镇土登村民委员会义常村13号"
    ppocr = "云南省大理白族自治州洱源县三营镇士登村民委员会义常村13号"

    final, _reason, changed = arbitrate_fast(
        {
            "field": "住址",
            "multimodal_value": initial,
            "ppocr_value": ppocr,
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 0.9997,
            "ppocr_confidence": 0.9998,
        },
        {"住址": initial},
    )

    assert final == ppocr
    assert changed is True


def test_fast_arbitration_keeps_address_below_existing_value_threshold() -> None:
    initial = "陕西省汉中市洋县渭水镇五丰村七组"
    ppocr = "陕西省汉中市洋县湑水镇五丰村七组"

    final, reason, changed = arbitrate_fast(
        {
            "field": "住址",
            "multimodal_value": initial,
            "ppocr_value": ppocr,
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 0.9993,
            "ppocr_confidence": 0.9666,
        },
        {"住址": initial},
    )

    assert final == initial
    assert changed is False
    assert "0.985" in reason


@pytest.mark.parametrize(
    "candidate",
    [
        "公民身份号码532930198310201756",
        "士登",
        "532930198310201756",
        "四川省眉山市洪雅县高庙镇中和村5组3实习",
        "河南省柘城县皇集乡易周村委二组091自",
    ],
)
def test_fast_address_validation_rejects_non_address_content(candidate: str) -> None:
    assert not valid_fast_field_value("住址", candidate)


def test_fast_arbitration_rejects_an_unrelated_high_confidence_address() -> None:
    initial = "云南省大理白族自治州洱源县三营镇士登村13号"
    unrelated = "广东省中山市古镇镇教昌外闸北三巷1号"

    final, reason, changed = arbitrate_fast(
        {
            "field": "住址",
            "multimodal_value": initial,
            "ppocr_value": unrelated,
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 0.999,
            "ppocr_confidence": 0.999,
        },
        {"住址": initial},
    )

    assert final == initial
    assert changed is False
    assert "差异过大" in reason


def test_fast_arbitration_keeps_hunyuan_when_address_division_changes() -> None:
    initial = "河南省荥阳市刘河镇庵上村外口053号"
    ppocr = "河南省荣阳市刘河镇庵上村外口053号"

    final, reason, changed = arbitrate_fast(
        {
            "field": "住址",
            "multimodal_value": initial,
            "ppocr_value": ppocr,
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 0.999,
            "ppocr_confidence": 0.9926,
        },
        {"住址": initial},
    )

    assert final == initial
    assert changed is False
    assert "行政区划前缀冲突" in reason
    assert "0.995" in reason


def test_fast_arbitration_allows_ultra_confident_address_division_change() -> None:
    initial = "安徽省宣城市孔城镇晴岚村马干6号"
    ppocr = "安徽省桐城市孔城镇晴岚村马干6号"

    final, _reason, changed = arbitrate_fast(
        {
            "field": "住址",
            "multimodal_value": initial,
            "ppocr_value": ppocr,
            "association": "label_spatial",
            "association_unique": True,
            "label_confidence": 1.0,
            "ppocr_confidence": 0.9952,
        },
        {"住址": initial},
    )

    assert final == ppocr
    assert changed is True


@pytest.mark.parametrize(
    "profile_key",
    ["id_card", "driver_license", "vehicle_license"],
)
def test_hybrid_pipeline_applies_safe_address_override_to_all_three_certificates(
    tmp_path: Path,
    profile_key: str,
) -> None:
    profile = load_profiles().get(profile_key)
    initial = "云南省大理白族自治州洱源县三营镇土登村民委员会义常村13号"
    ppocr = "云南省大理白族自治州洱源县三营镇士登村民委员会义常村13号"
    fields = {field: "" for field in profile.field_names}
    fields["住址"] = initial
    result = ExtractionResult(
        application_no="ADDRESS001",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    process = fuse_hybrid_result(
        application_no="ADDRESS001",
        profile=profile,
        client=object(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[
            {
                "input_path": str(tmp_path / f"{profile_key}.jpg"),
                "text": f"住址\n{ppocr}",
                "tokens": [
                    {"text": "住址", "score": 0.9997, "bbox": [10, 10, 70, 35]},
                    {"text": ppocr, "score": 0.9998, "bbox": [80, 10, 720, 35]},
                ],
            }
        ],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["住址"] == ppocr
    assert process["conflicts"][0]["changed"] is True
    assert process["conflicts"][0]["final_value"] == ppocr


def test_hybrid_fast_mode_never_calls_secondary_hunyuan_recheck(
    tmp_path: Path,
) -> None:
    profile = load_profiles().get("vehicle_license")
    fields = {field: "" for field in profile.field_names}
    fields["车牌号码"] = "鲁A1234S"
    fields["号牌号码"] = "鲁A1234S"
    result = ExtractionResult(
        application_no="A001",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    class NoSecondaryCallClient:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected secondary Hunyuan call: {name}")

    process = fuse_hybrid_result(
        application_no="A001",
        profile=profile,
        client=NoSecondaryCallClient(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[
            {
                "input_path": str(tmp_path / "vehicle.jpg"),
                "text": "号牌号码\n鲁A12345",
                "tokens": [
                    {"text": "号牌号码", "score": 0.99, "bbox": [10, 10, 90, 30]},
                    {"text": "鲁A12345", "score": 0.99, "bbox": [110, 10, 220, 30]},
                ],
            }
        ],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["车牌号码"] == "鲁A12345"
    assert result.fields["号牌号码"] == "鲁A12345"
    assert process["secondary_model_recheck"] is False
    assert process["conflicts"][0]["recheck"]["skipped"] is True
    assert not (tmp_path / "artifacts").exists()


def test_driver_type_detection_does_not_trigger_a_fallback_model_call(
    tmp_path: Path,
) -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    result = ExtractionResult(
        application_no="A002",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    class NoSecondaryCallClient:
        def __getattr__(self, name: str):
            raise AssertionError(f"unexpected secondary Hunyuan call: {name}")

    process = fuse_hybrid_result(
        application_no="A002",
        profile=profile,
        client=NoSecondaryCallClient(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[
            {
                "input_path": str(tmp_path / "driver.jpg"),
                "text": "无法判断页面类型",
                "tokens": [],
            }
        ],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["证件类型"] == ""
    assert process["secondary_model_recheck"] is False


@pytest.mark.parametrize("nationality", ["中国", "CHN", "中国/CHN"])
def test_driver_nationality_normalizes_chinese_aliases(
    tmp_path: Path,
    nationality: str,
) -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    fields["国籍"] = nationality
    result = ExtractionResult(
        application_no="N001",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    fuse_hybrid_result(
        application_no="N001",
        profile=profile,
        client=object(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["国籍"] == "中国"


@pytest.mark.parametrize(
    ("ocr_text", "address", "expected_source"),
    [
        ("国籍 中国", "", "model_text_china_evidence"),
        ("中华人民共和国机动车驾驶证", "", "model_text_china_evidence"),
        ("", "河南省叶县龙泉乡彭庄一组", "domestic_address_rule"),
        ("", "深圳市南山区粤海街道1号", "domestic_address_rule"),
    ],
)
def test_driver_empty_nationality_uses_china_evidence(
    tmp_path: Path,
    ocr_text: str,
    address: str,
    expected_source: str,
) -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    fields["住址"] = address
    result = ExtractionResult(
        application_no="N002",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    process = fuse_hybrid_result(
        application_no="N002",
        profile=profile,
        client=object(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[
            {
                "input_path": str(tmp_path / "driver.jpg"),
                "text": ocr_text,
                "tokens": [],
            }
        ],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["国籍"] == "中国"
    nationality_item = next(
        item for item in process["supplements"] if item["field"] == "国籍"
    )
    assert nationality_item["source"] == expected_source


def test_driver_nationality_rule_does_not_overwrite_a_foreign_value(
    tmp_path: Path,
) -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    fields["国籍"] = "美国"
    fields["住址"] = "北京市朝阳区建国路1号"
    result = ExtractionResult(
        application_no="N003",
        success=True,
        status="success",
        fields=dict(fields),
        records=[dict(fields)],
        elapsed=1.0,
    )

    fuse_hybrid_result(
        application_no="N003",
        profile=profile,
        client=object(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts",
        result=result,
        ocr_files=[
            {
                "input_path": str(tmp_path / "driver.jpg"),
                "text": "国籍 中国",
                "tokens": [],
            }
        ],
        ppocr_elapsed=0.5,
        parallel_elapsed=1.0,
    )

    assert result.fields["国籍"] == "美国"


def test_short_name_one_character_conflict_uses_whole_ocr_token() -> None:
    tokens = [
        {"text": "姓名", "score": 0.99, "bbox": [10, 10, 40, 30]},
        {"text": "陈小畔", "score": 0.98, "bbox": [50, 10, 110, 30]},
    ]

    evidence = find_evidence("陈小晖", tokens, "姓名")

    assert evidence is not None
    assert evidence.value == "陈小畔"
    assert evidence.token_indexes == (1,)


def test_two_character_conflict_requires_nearby_field_label() -> None:
    unrelated = [{"text": "李四", "score": 0.99, "bbox": [50, 10, 90, 30]}]
    labeled = [
        {"text": "姓名", "score": 0.99, "bbox": [10, 10, 40, 30]},
        {"text": "李四", "score": 0.99, "bbox": [50, 10, 90, 30]},
    ]

    assert find_evidence("李玉", unrelated, "姓名") is None
    assert find_evidence("李玉", labeled, "姓名").value == "李四"


def test_short_value_prefers_token_near_its_field_label() -> None:
    tokens = [
        {"text": "陈小畔", "score": 0.999, "bbox": [10, 80, 70, 100]},
        {"text": "姓名", "score": 0.99, "bbox": [10, 10, 40, 30]},
        {"text": "陈小盼", "score": 0.96, "bbox": [50, 10, 110, 30]},
    ]

    evidence = find_evidence("陈小晖", tokens, "姓名")

    assert evidence is not None
    assert evidence.value == "陈小盼"
    assert evidence.token_indexes == (2,)


def test_arbitrate_uses_ppocr_when_recheck_agrees() -> None:
    final, reason, changed = arbitrate(
        {
            "multimodal_value": "那殊村",
            "ppocr_value": "那珠村",
            "ppocr_confidence": 0.96,
        },
        "那珠村",
    )

    assert final == "那珠村"
    assert changed is True
    assert "支持PP-OCRv6" in reason


def test_arbitrate_accepts_ppocr_candidate_inside_verbose_crop_text() -> None:
    final, _reason, changed = arbitrate(
        {
            "field": "证号",
            "multimodal_value": "372929196912863317",
            "ppocr_value": "372929196912063317",
            "ppocr_confidence": 0.99,
        },
        "图片中的文本内容是号372929196912063317372900753",
    )

    assert final == "372929196912063317"
    assert changed is True


def test_arbitrate_keeps_initial_when_recheck_is_new_result() -> None:
    final, _reason, changed = arbitrate(
        {
            "multimodal_value": "那殊村",
            "ppocr_value": "那珠村",
            "ppocr_confidence": 0.96,
        },
        "那朱村",
    )

    assert final == "那殊村"
    assert changed is False


def test_missing_fields_are_supplemented_from_high_confidence_ppocr() -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    pages = [
        {
            "text": (
                "中华人民共和国机动车驾驶证\n"
                "姓名\n张三\n性别\n男\n准驾车型\nC1\n"
                "档案编号\n110100123456"
            ),
            "tokens": [
                {"text": "中华人民共和国机动车驾驶证", "score": 0.99, "bbox": [0, 0, 300, 30]},
                {"text": "姓名", "score": 0.99, "bbox": [0, 40, 40, 60]},
                {"text": "张三", "score": 0.98, "bbox": [50, 40, 100, 60]},
                {"text": "性别", "score": 0.99, "bbox": [0, 70, 40, 90]},
                {"text": "男", "score": 0.99, "bbox": [50, 70, 70, 90]},
                {"text": "准驾车型", "score": 0.99, "bbox": [0, 100, 80, 120]},
                {"text": "C1", "score": 0.99, "bbox": [90, 100, 120, 120]},
                {"text": "档案编号", "score": 0.99, "bbox": [0, 130, 80, 150]},
                {"text": "110100123456", "score": 0.99, "bbox": [90, 130, 240, 150]},
            ],
        }
    ]

    supplements = missing_field_supplements(profile, fields, pages)
    values = {item["field"]: item["value"] for item in supplements}

    assert values["姓名"] == "张三"
    assert values["性别"] == "男"
    assert values["准驾车型"] == "C1"
    assert values["档案编号"] == "110100123456"


def test_missing_fields_ignore_low_confidence_ppocr() -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    pages = [{
        "text": "姓名\n张三",
        "tokens": [
            {"text": "姓名", "score": 0.99, "bbox": [0, 0, 40, 20]},
            {"text": "张三", "score": 0.72, "bbox": [50, 0, 100, 20]},
        ],
    }]

    assert missing_field_supplements(profile, fields, pages) == []


def test_missing_driver_class_uses_nearest_labeled_token() -> None:
    profile = load_profiles().get("driver_license")
    fields = {field: "" for field in profile.field_names}
    pages = [{
        "text": "准驾车型\n交通警察支队\nClass\nC1D",
        "tokens": [
            {"text": "准驾车型", "score": 0.99, "bbox": [0, 0, 80, 20]},
            {"text": "交通警察支队", "score": 0.99, "bbox": [90, 0, 180, 20]},
            {"text": "Class", "score": 0.99, "bbox": [0, 30, 50, 50]},
            {"text": "C1D", "score": 0.98, "bbox": [60, 30, 100, 50]},
        ],
    }]

    supplements = missing_field_supplements(profile, fields, pages)

    assert {item["field"]: item["value"] for item in supplements}["准驾车型"] == "C1D"


def test_recheck_parser_accepts_hunyuan_plain_and_spotting_text() -> None:
    assert parse_recheck_value("张三", "姓名") == "张三"
    assert parse_recheck_value("姓名\n张三\n姓名\n张三", "姓名") == "张三"
    assert parse_recheck_value(
        '[{"box":[1,2,3,4],"text":"姓名"},'
        '{"box":[5,6,7,8],"text":"张三"}]',
        "姓名",
    ) == "张三"
    assert (
        parse_recheck_value(
            "姓名(10,20),(50,40)张三(55,20),(95,40)",
            "姓名",
        )
        == "张三"
    )
    assert parse_recheck_value("图片中的文本内容是：陈学贵", "姓名") == "陈学贵"
    assert (
        parse_recheck_value("图片中的文本内容是：422822740401", "档案编号")
        == "422822740401"
    )
    assert (
        parse_recheck_value(
            "|信息|详情|\n|---|---|\n|住址|云南省昭通市镇雄县穿洞村145号|",
            "住址",
        )
        == "云南省昭通市镇雄县穿洞村145号"
    )


def test_hunyuan_recheck_uses_local_ocr_prompt_only(tmp_path: Path) -> None:
    class Config:
        vision_response_adapter = "hunyuan_ocr"

    class Client:
        config = Config()
        request: dict[str, object] = {}

        def build_general_vision_payload(
            self, images, *, system_prompt: str, user_prompt: str
        ):
            self.request = {
                "images": images,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            return {"request": "local-recheck"}

        def _complete_payload(self, payload):
            assert payload == {"request": "local-recheck"}
            return "张三"

    images = [tmp_path / f"crop-{index}.png" for index in range(3)]
    client = Client()

    result = recheck_conflict(client, "姓名", images)

    assert result["value"] == "张三"
    assert client.request["images"] == images
    assert client.request["system_prompt"] == ""
    assert client.request["user_prompt"] == local_recheck_prompt()
    assert "姓名" not in client.request["user_prompt"]
    assert (
        parse_recheck_value(
            "<|ref|>姓名<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
            "<|ref|>张三<|/ref|><|det|>[[5,6,7,8]]<|/det|>",
            "姓名",
        )
        == "张三"
    )


def test_create_recheck_images_saves_three_auditable_variants(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1000, 600), "white").save(source)

    paths = create_recheck_images(
        source,
        [200, 200, 800, 280],
        tmp_path / "crops",
        "住址",
    )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_create_recheck_images_uses_exif_oriented_coordinate_canvas(tmp_path: Path) -> None:
    source = tmp_path / "exif_source.jpg"
    image = Image.new("RGB", (400, 300), "white")
    # Raw bottom-left becomes visual top-left after EXIF orientation 6.
    for x in range(0, 100):
        for y in range(200, 300):
            image.putpixel((x, y), (255, 0, 0))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)

    paths = create_recheck_images(
        source,
        [0, 0, 100, 100],
        tmp_path / "crops",
        "住址",
    )

    with Image.open(paths[0]) as crop:
        center = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
    assert center[0] > 200
    assert center[1] < 60
    assert center[2] < 60


@pytest.mark.parametrize(
    "bbox",
    [
        [800, 100, 200, 200],
        [100, 300, 200, 100],
        [float("nan"), 10, 50, 50],
        [1500, 100, 1800, 200],
        [-500, 100, -200, 200],
    ],
)
def test_create_recheck_images_rejects_invalid_or_outside_bbox(
    tmp_path: Path,
    bbox: list[float],
) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1000, 600), "white").save(source)
    output_dir = tmp_path / "crops"

    with pytest.raises(ValueError, match="复核字段坐标"):
        create_recheck_images(source, bbox, output_dir, "住址")

    assert not output_dir.exists()


def test_create_recheck_images_clamps_partly_outside_bbox(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1000, 600), "white").save(source)

    paths = create_recheck_images(
        source,
        [-20, 540, 200, 640],
        tmp_path / "crops",
        "住址",
    )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
