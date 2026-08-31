from llm_manager.driver_refiner import (
    detect_driver_page_type,
    refine_driver_fields,
)


def test_driver_type_prefers_electronic_screen_markers_over_rendered_title() -> None:
    assert detect_driver_page_type(
        "电子驾驶证\n中华人民共和国机动车驾驶证\n累积记分 0分\n主页\n下载"
    ) == "电子驾驶证"


def test_driver_refiner_uses_label_evidence_and_constraints() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "电子驾驶证",
                "出生日期 1994-03-16",
                "初次领证日期 2019-04-18",
                "准驾车型 C1",
                "有效期限 2025-04-18 至 2035-04-18",
                "档案编号 130502341096",
            ]
        ),
        {
            "类型": "",
            "出生日期": "",
            "初次领证日期": "",
            "准驾车型": "BIL",
            "有效期限": "10年",
            "档案编号": "无效档案编号",
            "性别": "未知",
        },
    )

    assert refined["类型"] == "电子驾驶证"
    assert refined["出生日期"] == "19940316"
    assert refined["初次领证日期"] == "20190418"
    assert refined["准驾车型"] == "C1"
    assert refined["有效期限"] == "20250418-20350418"
    assert refined["档案编号"] == "130502341096"
    assert refined["性别"] == ""


def test_driver_refiner_prefers_full_image_archive_over_wrong_local_recheck() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "[图片 DG14/physical.jpg]",
                "证号",
                "410721198604064551",
                "410700998596",
                "住址",
                "河南省新乡市红旗区关堤乡刘堤村404号",
                "姓名",
                "陈小畔",
                "档案编号",
                "[局部复识别 DG14/physical.jpg/档案编号]",
                "档案编号",
                "410999997596",
            ]
        ),
        {"档案编号": "410999997596"},
    )

    assert refined["档案编号"] == "410700998596"


def test_driver_refiner_clears_archive_when_full_images_conflict() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "[图片 DG14/one.jpg]",
                "档案编号 110100123456",
                "[图片 DG14/two.jpg]",
                "档案编号 220200654321",
                "[局部复识别 DG14/two.jpg/档案编号]",
                "档案编号 220200654321",
            ]
        ),
        {"档案编号": "220200654321"},
    )

    assert refined["档案编号"] == ""


def test_driver_refiner_does_not_use_duration_without_date_range() -> None:
    refined = refine_driver_fields(
        "中华人民共和国机动车驾驶证\n有效年限 10年",
        {"有效期限": ""},
    )

    assert refined["有效期限"] == ""


def test_driver_refiner_ignores_class_explanation_table() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "准驾车型",
                "准驾车型代号规定",
                "A1:大型客车和A3、B1、B2",
                "C1:小型汽车和C2、C3",
            ]
        ),
        {"准驾车型": "BIL"},
    )

    assert refined["准驾车型"] == ""


def test_driver_refiner_reads_each_image_before_its_own_class_table() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "[图片 DG14/back.jpg]",
                "准驾车型代号规定",
                "A1:大型客车和A3、B1、B2",
                "[图片 DG14/front.jpg]",
                "姓名 张三",
                "准驾车型",
                "C1D",
                "累积记分 0分",
            ]
        ),
        {"准驾车型": ""},
    )

    assert refined["准驾车型"] == "C1D"


def test_driver_refiner_repairs_common_digit_one_ocr_confusions() -> None:
    ci = refine_driver_fields(
        "准驾车型\nCI\n累积记分 0分",
        {"准驾车型": ""},
    )
    cj = refine_driver_fields(
        "[局部复识别 DG14/example.jpg/准驾车型]\n准驾车型 CJ\nClass",
        {"准驾车型": ""},
    )
    cid = refine_driver_fields(
        "准驾车型\nCID\n有效期限",
        {"准驾车型": ""},
    )

    assert ci["准驾车型"] == "C1"
    assert cj["准驾车型"] == "C1"
    assert cid["准驾车型"] == "C1D"


def test_driver_refiner_recovers_rotated_reading_order_class() -> None:
    before_label = refine_driver_fields(
        "初次领证日期\n2018-12-25\nC1\n准驾车型\n交通警察支队\nClass",
        {"准驾车型": ""},
    )
    delayed = refine_driver_fields(
        "准驾车型\n初次领证日期\n出生日期\nClass\n"
        "Date of First Issue\n证号\n2014-12-12\nSex\nCID\nNationality中国",
        {"准驾车型": ""},
    )

    assert before_label["准驾车型"] == "C1"
    assert delayed["准驾车型"] == "C1D"


def test_driver_refiner_does_not_choose_conflicting_labelled_classes() -> None:
    refined = refine_driver_fields(
        "[图片 DG14/one.jpg]\n准驾车型 C1\n累积记分\n"
        "[图片 DG14/two.jpg]\n准驾车型 C1D\n累积记分",
        {"准驾车型": ""},
    )

    assert refined["准驾车型"] == ""


def test_driver_refiner_detects_electronic_ui_without_title() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "姓名 张三",
                "准驾车型 C1",
                "累积记分 0分",
                "状态 正常",
                "生成日期为2026年07月28日",
                "主页 副页 刷新 换照片 下载",
            ]
        ),
        {"类型": "中华人民共和国机动车驾驶证"},
    )

    assert refined["类型"] == "电子驾驶证"


def test_driver_refiner_writes_new_document_type_field_from_ocr_rules() -> None:
    electronic = refine_driver_fields(
        "累积记分 0分\n生成日期 2026年07月28日\n下载",
        {"证件类型": ""},
    )
    physical = refine_driver_fields(
        "中华人民共和国机动车驾驶证",
        {"证件类型": ""},
    )

    assert electronic["证件类型"] == "电子驾驶证"
    assert physical["证件类型"] == "中华人民共和国机动车驾驶证"


def test_driver_refiner_preserves_model_type_for_mixed_materials() -> None:
    text = "\n".join(
        [
            "中华人民共和国机动车驾驶证",
            "电子驾驶证",
            "累积记分 0分",
            "生成日期为2026年07月28日",
        ]
    )

    physical = refine_driver_fields(
        text,
        {"类型": "中华人民共和国机动车驾驶证"},
    )
    electronic = refine_driver_fields(text, {"类型": "电子驾驶证"})

    assert physical["类型"] == "中华人民共和国机动车驾驶证"
    assert electronic["类型"] == "电子驾驶证"


def test_driver_refiner_prefers_unique_complete_identity_and_address() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "[图片 DG14/masked.jpg]",
                "住址",
                "北京市朝阳区幸福路**号",
                "证号",
                "110105********002X",
                "[图片 DG14/complete.jpg]",
                "住址",
                "北京市朝阳区幸福路88号",
                "出生日期",
                "证号",
                "11010519491231002X",
                "性别",
            ]
        ),
        {
            "住址": "北京市朝阳区幸福路**号",
            "证号": "110105********002X",
        },
    )

    assert refined["住址"] == "北京市朝阳区幸福路88号"
    assert refined["证号"] == "11010519491231002X"


def test_driver_refiner_keeps_masked_values_when_complete_candidates_conflict() -> None:
    refined = refine_driver_fields(
        "\n".join(
            [
                "[图片 DG14/one.jpg]",
                "住址",
                "北京市朝阳区幸福路88号",
                "证号 11010519491231002X",
                "[图片 DG14/two.jpg]",
                "住址",
                "上海市浦东新区世纪大道100号",
                "证号 340121199012291631",
            ]
        ),
        {
            "住址": "北京市朝阳区幸福路**号",
            "证号": "110105********002X",
        },
    )

    assert refined["住址"] == "北京市朝阳区幸福路**号"
    assert refined["证号"] == "110105********002X"


def test_driver_refiner_never_reconstructs_when_only_masked_value_exists() -> None:
    refined = refine_driver_fields(
        "住址\n安徽省淮南市谢家集**********\n证号\n340121********1631",
        {
            "住址": "安徽省淮南市谢家集**********",
            "证号": "340121********1631",
        },
    )

    assert refined["住址"] == "安徽省淮南市谢家集**********"
    assert refined["证号"] == "340121********1631"


def test_driver_refiner_repairs_address_only_from_unique_normalized_evidence() -> None:
    refined = refine_driver_fields(
        "[图片 DG14/a.jpg]\n住址\n黑龙江省齐齐哈尔市建华区杨家窑社区5\n"
        "Address\n7组\n[驾驶证版式归一化/住址]\n"
        "黑龙江省齐齐哈尔市建华区杨家窑社区57组",
        {"住址": "黑龙江省齐齐哈尔市建华区杨家窑社区5组"},
    )

    assert refined["住址"] == "黑龙江省齐齐哈尔市建华区杨家窑社区57组"


def test_driver_refiner_repairs_theta_as_leading_zero_in_doorplate() -> None:
    refined = refine_driver_fields(
        "[图片 DG14/a.jpg]\n住址山东省鄄城县旧城镇毛洼行政村毛洼村θ\n"
        "Address\n59号\n[驾驶证版式归一化/住址]\n"
        "山东省鄄城县旧城镇毛洼行政村毛洼村θ59号",
        {"住址": "山东省鄄城县旧城镇毛洼行政村毛洼村θ59号"},
    )

    assert refined["住址"] == "山东省鄄城县旧城镇毛洼行政村毛洼村059号"


def test_driver_refiner_does_not_replace_theta_outside_doorplate_digits() -> None:
    refined = refine_driver_fields(
        "住址\n北京市海淀区Θ科技园",
        {"住址": "北京市海淀区Θ科技园"},
    )

    assert refined["住址"] == "北京市海淀区Θ科技园"


def test_driver_refiner_trims_authority_location_after_complete_group() -> None:
    refined = refine_driver_fields(
        "住址\n湖南省武冈县湾头镇八合村16组\nAddress\n广东省惠州\n"
        "Date of Birth\n出生日期1970-04-21\n市公安局交",
        {"住址": "湖南省武冈县湾头镇八合村16组广东省惠州"},
    )

    assert refined["住址"] == "湖南省武冈县湾头镇八合村16组"


def test_driver_refiner_preserves_valid_address_after_group() -> None:
    with_doorplate = refine_driver_fields(
        "住址\n湖南省武冈县湾头镇八合村16组3号",
        {"住址": "湖南省武冈县湾头镇八合村16组3号"},
    )
    with_building = refine_driver_fields(
        "住址\n湖南省武冈县湾头镇八合村16组2栋501室",
        {"住址": "湖南省武冈县湾头镇八合村16组2栋501室"},
    )

    assert with_doorplate["住址"] == "湖南省武冈县湾头镇八合村16组3号"
    assert with_building["住址"] == "湖南省武冈县湾头镇八合村16组2栋501室"


def test_driver_refiner_keeps_conflicting_or_ambiguous_address() -> None:
    conflicting = refine_driver_fields(
        "[驾驶证版式归一化/住址]\n北京市朝阳区幸福路88号",
        {"住址": "上海市浦东新区世纪大道100号"},
    )
    ambiguous = refine_driver_fields(
        "[图片 a.jpg]\n[驾驶证版式归一化/住址]\n北京市朝阳区幸福路88号\n"
        "[图片 b.jpg]\n[驾驶证版式归一化/住址]\n北京市朝阳区幸福路89号",
        {"住址": "北京市朝阳区幸福路8号"},
    )

    assert conflicting["住址"] == "上海市浦东新区世纪大道100号"
    assert ambiguous["住址"] == "北京市朝阳区幸福路8号"
