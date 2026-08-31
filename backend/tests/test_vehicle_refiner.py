from llm_manager.vehicle_refiner import refine_vehicle_fields


def test_refiner_repairs_label_adjacent_vehicle_fields() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "检验记录",
                "汽油",
                "总质量 1800kg",
                "整备质量 1350kg",
                "外廓尺寸 4490×1860×1590mm",
            ]
        ),
        {
            "检验记录": "",
            "总质量": "1350kg",
            "整备质量": "",
            "外廓尺寸": "",
        },
    )

    assert refined["检验记录"] == "汽油"
    assert refined["总质量"] == "1800kg"
    assert refined["整备质量"] == "1350kg"
    assert refined["外廓尺寸"] == "4490×1860×1590mm"


def test_refiner_repairs_decimal_point_noise_in_dimensions() -> None:
    refined = refine_vehicle_fields(
        "外廓尺寸 47.62×1926×1632mm",
        {"外廓尺寸": "47.62×1926×1632mm"},
    )

    assert refined["外廓尺寸"] == "4762×1926×1632mm"


def test_refiner_pairs_registration_and_issue_dates() -> None:
    refined = refine_vehicle_fields(
        "注册日期\n发证日期\n2022-09-19\n2024-12-31",
        {"注册日期": "", "发证日期": ""},
    )

    assert refined["注册日期"] == "20220919"
    assert refined["发证日期"] == "20241231"


def test_refiner_recovers_month_only_inspection_expiry_on_next_line() -> None:
    refined = refine_vehicle_fields(
        "检验有效期至\n2024年11月",
        {"检验有效期": ""},
    )

    assert refined["检验有效期"] == "202411"


def test_refiner_maps_values_before_secondary_page_mass_labels() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 Z002/example.jpg]",
                "1778kg",
                "核定载人数5人",
                "总质量",
                "1318kg",
                "核定载质量",
                "整备质量",
                "外廓尺寸4670×1806×1474mm",
            ]
        ),
        {"总质量": "1318kg", "整备质量": ""},
    )

    assert refined["总质量"] == "1778kg"
    assert refined["整备质量"] == "1318kg"


def test_refiner_never_uses_dimension_as_curb_mass() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 Z002/example.jpg]",
                "1725kg",
                "总质量",
                "1265kg",
                "核定载质量",
                "整备质量",
                "4670×1806×1474mm",
                "外廓尺寸",
            ]
        ),
        {"总质量": "", "整备质量": "4670kg"},
    )

    assert refined["总质量"] == "1725kg"
    assert refined["整备质量"] == "1265kg"


def test_refiner_pairs_dates_when_issue_label_comes_first() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 Z002/example.jpg]",
                "发证日期",
                "管理局",
                "注册日期",
                "2024-08-14",
                "2022-06-23",
                "Issue Date",
                "RegisterDate",
            ]
        ),
        {"注册日期": "20240814", "发证日期": "20220623"},
    )

    assert refined["注册日期"] == "20220623"
    assert refined["发证日期"] == "20240814"


def test_refiner_ignores_invalid_numeric_date_noise() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 Z002/example.jpg]",
                "注册日期",
                "发证日期",
                "51145100",
                "62113601",
            ]
        ),
        {"注册日期": "20190606", "发证日期": "20260109"},
    )

    assert refined["注册日期"] == "20190606"
    assert refined["发证日期"] == "20260109"


def test_refiner_recovers_total_and_curb_mass_from_two_values() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 Z002/example.jpg]",
                "核定载人数5人",
                "总质量",
                "1778kg",
                "整备质量",
                "1318kg",
                "核定载质量",
            ]
        ),
        {"总质量": "1778kg", "整备质量": "1778kg"},
    )

    assert refined["总质量"] == "1778kg"
    assert refined["整备质量"] == "1318kg"


def test_refiner_constrains_use_nature_to_ocr_dictionary_value() -> None:
    refined = refine_vehicle_fields(
        "车辆类型 小型轿车\n使用性质\n非营运\n品牌型号 示例牌ABC123",
        {"使用性质": "示例牌ABC123"},
    )

    assert refined["使用性质"] == "非营运"


def test_refiner_recovers_complete_plate_with_province_prefix() -> None:
    refined = refine_vehicle_fields(
        "号牌号码\n粤 B12345\nPlate No.",
        {"车牌号码": "B12345", "号牌号码": "B12345"},
    )

    assert refined["车牌号码"] == "粤B12345"
    assert refined["号牌号码"] == "粤B12345"


def test_refiner_does_not_invent_missing_plate_province() -> None:
    refined = refine_vehicle_fields(
        "号牌号码\nB12345\nPlate No.",
        {"车牌号码": "B12345", "号牌号码": "B12345"},
    )

    assert refined["车牌号码"] == ""
    assert refined["号牌号码"] == ""


def test_refiner_preserves_full_new_energy_inspection_text() -> None:
    electric = refine_vehicle_fields(
        "检验记录\n新能源/电",
        {"检验记录": "新能源"},
    )
    hybrid = refine_vehicle_fields(
        "检验记录\n新能源/混",
        {"检验记录": "混合动力"},
    )

    assert electric["检验记录"] == "新能源/电"
    assert hybrid["检验记录"] == "新能源/混"


def test_refiner_recovers_archive_number_across_noise_lines() -> None:
    refined = refine_vehicle_fields(
        "\n".join([
            "档案编号",
            "二维码请妥善保管",
            "公安交通管理局",
            "440300123456",
            "检验记录",
        ]),
        {"档案号码": ""},
    )

    assert refined["档案号码"] == "440300123456"


def test_refiner_recovers_people_only_from_label_neighbourhood() -> None:
    recovered = refine_vehicle_fields(
        "1778kg\n5人\n总质量\n核定载人数\n1318kg\n整备质量",
        {"核定载人数": ""},
    )
    inferred_small_car = refine_vehicle_fields(
        "车辆类型 小型轿车\n品牌型号 示例牌ABC123",
        {"核定载人数": ""},
    )
    not_guessed = refine_vehicle_fields(
        "车辆类型 小型普通客车\n品牌型号 示例牌ABC123",
        {"核定载人数": ""},
    )

    assert recovered["核定载人数"] == "5人"
    assert inferred_small_car["核定载人数"] == "5人"
    assert not_guessed["核定载人数"] == ""


def test_refiner_prefers_explicit_people_over_small_car_default() -> None:
    refined = refine_vehicle_fields(
        "车辆类型 小型轿车\n核定载人数 4人",
        {"核定载人数": ""},
    )

    assert refined["核定载人数"] == "4人"


def test_vehicle_refiner_prefers_unique_complete_vin_and_address() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 X001/masked.jpg]",
                "住址",
                "山东省滕州市荆庄街**号",
                "车辆识别代号",
                "LSV******LN010878",
                "[图片 X001/complete.jpg]",
                "住址",
                "山东省滕州市荆庄街18号",
                "使用性质",
                "车辆识别代号",
                "LSVUD6B22LN010878",
                "发动机号码",
            ]
        ),
        {
            "住址": "山东省滕州市荆庄街**号",
            "车辆识别代号": "LSV******LN010878",
        },
    )

    assert refined["住址"] == "山东省滕州市荆庄街18号"
    assert refined["车辆识别代号"] == "LSVUD6B22LN010878"


def test_vehicle_refiner_keeps_masked_values_when_complete_candidates_conflict() -> None:
    refined = refine_vehicle_fields(
        "\n".join(
            [
                "[图片 X001/one.jpg]",
                "住址\n山东省滕州市荆庄街18号\n使用性质",
                "车辆识别代号\nLSVUD6B22LN010878\n发动机号码",
                "[图片 X001/two.jpg]",
                "住址\n上海市浦东新区世纪大道100号\n使用性质",
                "车辆识别代号\nLSVAL60C2N2048648\n发动机号码",
            ]
        ),
        {
            "住址": "山东省滕州市荆庄街**号",
            "车辆识别代号": "LSV******LN010878",
        },
    )

    assert refined["住址"] == "山东省滕州市荆庄街**号"
    assert refined["车辆识别代号"] == "LSV******LN010878"


def test_vehicle_refiner_never_reconstructs_when_only_masked_value_exists() -> None:
    refined = refine_vehicle_fields(
        "住址\n山东省滕州市荆庄街**号\n车辆识别代号\nLSV******LN010878",
        {
            "住址": "山东省滕州市荆庄街**号",
            "车辆识别代号": "LSV******LN010878",
        },
    )

    assert refined["住址"] == "山东省滕州市荆庄街**号"
    assert refined["车辆识别代号"] == "LSV******LN010878"
