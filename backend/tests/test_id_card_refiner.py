from llm_manager.id_card_refiner import refine_id_card_fields


def test_id_card_refiner_uses_repeated_exact_address_not_llm_rewrite() -> None:
    text = (
        "[图片 DG12/one.jpg]\n住址\n贵州省印江土家族苗族自\n"
        "治县天堂镇金城村老虎塘\n组\n公民身份号码\n"
        "522226198802181216\n"
        "[图片 DG12/two.jpg]\n住址\n贵州省印江土家族苗族自\n"
        "治县天堂镇金城村老虎塘\n组\n公民身份号码\n"
        "522226198802181216"
    )

    refined = refine_id_card_fields(
        text,
        {
            "住址": "贵州省印江土家族苗族自治州印江土家族苗族自冶县"
            "天堂镇金城村老虎塘组"
        },
    )

    assert refined["住址"] == "贵州省印江土家族苗族自治县天堂镇金城村老虎塘组"


def test_id_card_refiner_prefers_clean_address_without_inserted_noise() -> None:
    refined = refine_id_card_fields(
        "[图片 DG12/noisy.jpg]\n住址河南省叶县龙泉乡雷岗村\n"
        "全安店念发一\n雷北组\n公民身份号码\n41042219930717541X\n"
        "[图片 DG12/clean.jpg]\n住址\n河南省叶县龙泉乡雷岗村\n"
        "雷北组\n公民身份号码\n41042219930717541X",
        {"住址": "河南省叶县龙泉乡雷岗村全安店念发一雷北组"},
    )

    assert refined["住址"] == "河南省叶县龙泉乡雷岗村雷北组"


def test_id_card_refiner_drops_standalone_number_after_address() -> None:
    refined = refine_id_card_fields(
        "住址\n内蒙古包头市固阳县下湿\n壕镇后脑包脑包壕村\n"
        "221\n公民身份号码150222197808253267",
        {"住址": "内蒙古包头市固阳县下湿壕镇后脑包脑包壕村221"},
    )

    assert refined["住址"] == "内蒙古包头市固阳县下湿壕镇后脑包脑包壕村"


def test_id_card_refiner_does_not_replace_complete_address_with_truncated_copy() -> None:
    refined = refine_id_card_fields(
        "[图片 DG12/truncated.jpg]\n住址\n重庆市丰都县三合街道乌\n"
        "公民身份号码500230199001010019\n"
        "[图片 DG12/complete.jpg]\n住址\n重庆市丰都县三合街道乌龙7组60号\n"
        "公民身份号码500230199001010019",
        {"住址": "重庆市丰都县三合街道乌龙7组60号"},
    )

    assert refined["住址"] == "重庆市丰都县三合街道乌龙7组60号"


def test_id_card_refiner_recovers_birth_from_unique_valid_identity() -> None:
    first = refine_id_card_fields(
        "出生1979年53月19日\n公民身份号码15263019790519037X",
        {"出生日期": ""},
    )
    second = refine_id_card_fields(
        "出生1980年63月5日\n公民身份号码152822198006050811",
        {"出生日期": ""},
    )

    assert first["出生日期"] == "19790519"
    assert second["出生日期"] == "19800605"


def test_id_card_refiner_does_not_choose_birth_when_identities_conflict() -> None:
    refined = refine_id_card_fields(
        "公民身份号码532523200408190420\n"
        "公民身份号码532528197505251528",
        {"出生日期": ""},
    )

    assert refined["出生日期"] == ""
