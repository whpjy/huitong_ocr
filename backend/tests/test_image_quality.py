from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from api.image_quality import (
    _driver_license_field_assignments,
    _driver_large_area_glare_fallback_metrics,
    _driver_license_semantic_glare_metrics,
    _driver_partial_valid_period_projection,
    _driver_physical_glare_fallback_metrics,
    _driver_weak_physical_evidence,
    _is_severe_glare,
    _refine_field_glare_metrics,
    _vehicle_license_semantic_glare_metrics,
    analyze_image_quality,
)


def _encode(image: np.ndarray) -> bytes:
    success, content = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert success
    return content.tobytes()


def _encode_with_exif_orientation(image: np.ndarray, orientation: int) -> bytes:
    output = BytesIO()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(rgb).save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


def _document_photo() -> np.ndarray:
    image = np.full((800, 1200, 3), 45, dtype=np.uint8)
    cv2.rectangle(image, (90, 90), (1110, 710), (220, 205, 175), -1)
    cv2.rectangle(image, (90, 90), (1110, 710), (245, 235, 215), 8)
    for y in range(180, 620, 70):
        cv2.line(image, (180, y), (780, y), (55, 65, 75), 9)
    cv2.rectangle(image, (850, 180), (1030, 440), (90, 110, 130), -1)
    return image


def _ppocr(*tokens: tuple[str, float, list[float]]) -> dict:
    return {
        "tokens": [
            {"text": text, "score": score, "bbox": bbox}
            for text, score, bbox in tokens
        ]
    }


def test_clear_document_photo_is_accepted() -> None:
    report = analyze_image_quality(_encode(_document_photo()), "id_card")

    assert report.accepted
    assert report.metrics["blur"]["roi_source"] == "full_image"


def test_exif_orientation_matches_ppocr_visual_coordinates() -> None:
    # EXIF orientation 6 displays a 240x120 pixel matrix as 120x240. PP-OCR
    # normalizes that orientation before returning boxes, so the quality gate
    # must use the same displayed dimensions when clipping those boxes.
    image = np.full((120, 240, 3), 90, dtype=np.uint8)
    content = _encode_with_exif_orientation(image, 6)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [10, 175, 110, 220]),
    )

    report = analyze_image_quality(content, "driver_license", ppocr)

    assert report.metrics["width"] == 120
    assert report.metrics["height"] == 240
    assert report.metrics["driver_license_scope"]["original_token_count"] == 1


def test_glare_metrics_reject_only_obviously_large_areas() -> None:
    # Small or scattered reflections are tolerated.
    assert not _is_severe_glare(0.0271, 0.0177, 0.0013)
    assert not _is_severe_glare(0.049, 0.032, 0.024)
    assert not _is_severe_glare(0.079, 0.049, 0.039)
    assert not _is_severe_glare(0.119, 0.079, 0.059)

    # Long bright details alone are not enough when their occupied area is low.
    assert not _is_severe_glare(0.0176, 0.0096, 0.0033)
    # A bright printed area alone is not treated as specular glare.
    assert not _is_severe_glare(0.0485, 0.0222, 0.0)
    # A large connected patch or widespread streaks still fail the gate.
    assert _is_severe_glare(0.125, 0.085, 0.0)
    assert _is_severe_glare(0.125, 0.016, 0.065)
    assert _is_severe_glare(0.21, 0.02, 0.0)


def test_obviously_blurred_photo_is_rejected() -> None:
    blurred = cv2.GaussianBlur(_document_photo(), (61, 61), 18)

    report = analyze_image_quality(_encode(blurred), "driver_license")

    assert "TOO_BLURRY" in {issue.code for issue in report.issues}


def test_moderately_defocused_document_text_is_allowed() -> None:
    blurred = cv2.GaussianBlur(_document_photo(), (13, 13), 4)

    report = analyze_image_quality(_encode(blurred), "driver_license")

    assert "TOO_BLURRY" not in {issue.code for issue in report.issues}, report.metrics[
        "blur"
    ]


def test_directional_motion_blur_is_allowed() -> None:
    kernel = np.zeros((25, 25), dtype=np.float32)
    kernel[12, :] = 1.0 / 25.0
    blurred = cv2.filter2D(_document_photo(), -1, kernel)

    report = analyze_image_quality(_encode(blurred), "driver_license")

    assert "TOO_BLURRY" not in {issue.code for issue in report.issues}, report.metrics[
        "blur"
    ]


def test_noisy_motion_blur_is_allowed() -> None:
    kernel = np.zeros((31, 31), dtype=np.float32)
    kernel[15, :] = 1.0 / 31.0
    blurred = cv2.filter2D(_document_photo(), -1, kernel).astype(np.int16)
    noise = np.random.default_rng(7).normal(0, 12, blurred.shape)
    noisy_blurred = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    report = analyze_image_quality(_encode(noisy_blurred), "driver_license")

    assert "TOO_BLURRY" not in {issue.code for issue in report.issues}, report.metrics[
        "blur"
    ]


def test_sensor_noise_does_not_reject_clear_document() -> None:
    clear = _document_photo().astype(np.int16)
    noise = np.random.default_rng(11).normal(0, 12, clear.shape)
    noisy_clear = np.clip(clear + noise, 0, 255).astype(np.uint8)

    report = analyze_image_quality(_encode(noisy_clear), "driver_license")

    assert "TOO_BLURRY" not in {issue.code for issue in report.issues}, report.metrics[
        "blur"
    ]


def test_large_glare_region_is_rejected() -> None:
    image = _document_photo()
    cv2.rectangle(image, (380, 220), (850, 500), (255, 255, 255), -1)

    report = analyze_image_quality(_encode(image), "vehicle_license")

    assert "SEVERE_GLARE" in {issue.code for issue in report.issues}


def test_moderate_fragmented_glare_streaks_are_allowed() -> None:
    image = _document_photo()
    for offset in (0, 85, 170, 255):
        cv2.line(
            image,
            (220, 190 + offset),
            (900, 360 + offset),
            (255, 255, 250),
            11,
        )

    report = analyze_image_quality(_encode(image), "driver_license")

    codes = {issue.code for issue in report.issues}
    assert "SEVERE_GLARE" not in codes
    assert report.metrics["elongated_glare_ratio"] > 0.012


def test_small_license_wallet_edge_glare_is_allowed() -> None:
    image = np.full((620, 1500, 3), 24, dtype=np.uint8)
    pages = (
        ((35, 45), (735, 575)),
        ((790, 55), (1465, 565)),
    )
    for (left, top), (right, bottom) in pages:
        cv2.rectangle(image, (left, top), (right, bottom), (195, 225, 205), -1)
        for y in range(top + 80, top + 300, 50):
            cv2.line(
                image,
                (left + 70, y),
                (right - 90, y),
                (45, 65, 55),
                5,
            )
        # Transparent wallet glare stays near the page edges and must not be
        # treated as text-obscuring glare.
        cv2.line(
            image,
            (left + 15, top + 8),
            (right - 20, top + 12),
            (255, 255, 250),
            18,
        )
        cv2.line(
            image,
            (left + 20, bottom - 12),
            (right - 25, bottom - 8),
            (255, 255, 250),
            18,
        )

    report = analyze_image_quality(_encode(image), "driver_license")

    codes = {issue.code for issue in report.issues}
    assert "SEVERE_GLARE" not in codes, report.metrics


def test_multiple_separate_documents_are_not_a_quality_condition() -> None:
    image = np.full((900, 1200, 3), 50, dtype=np.uint8)
    cv2.rectangle(image, (70, 100), (650, 450), (220, 205, 175), -1)
    cv2.rectangle(image, (70, 100), (650, 450), (245, 235, 215), 7)
    cv2.rectangle(image, (580, 500), (1130, 835), (210, 200, 175), -1)
    cv2.rectangle(image, (580, 500), (1130, 835), (245, 235, 215), 7)
    for start_x, start_y in ((110, 170), (620, 570)):
        for offset in range(0, 180, 45):
            cv2.line(
                image,
                (start_x, start_y + offset),
                (start_x + 330, start_y + offset),
                (55, 65, 75),
                6,
            )

    report = analyze_image_quality(_encode(image), "driver_license")

    assert "MULTIPLE_DOCUMENTS" not in {issue.code for issue in report.issues}


def test_slanted_licences_are_not_rejected_for_crop() -> None:
    image = np.full((1600, 1200, 3), (120, 155, 185), dtype=np.uint8)
    upper = np.array(
        [[170, 130], [910, 20], [1060, 920], [270, 1060]],
        dtype=np.int32,
    )
    lower = np.array(
        [[470, 1040], [1110, 980], [1199, 1540], [560, 1599]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, upper, (195, 225, 195))
    cv2.polylines(image, [upper], True, (35, 45, 50), 20)
    cv2.fillConvexPoly(image, lower, (190, 220, 190))
    cv2.polylines(image, [lower], True, (35, 45, 50), 18)
    for offset in (0, 150, 300, 450):
        cv2.line(
            image,
            (300, 260 + offset),
            (920, 390 + offset),
            (255, 255, 250),
            14,
        )

    report = analyze_image_quality(_encode(image), "driver_license")

    codes = {issue.code for issue in report.issues}
    assert "SEVERE_GLARE" not in codes
    assert "DOCUMENT_CROPPED" not in codes


def test_small_document_region_is_not_a_quality_condition() -> None:
    image = np.full((800, 1200, 3), 55, dtype=np.uint8)
    for x in range(0, 1200, 60):
        cv2.line(image, (x, 0), (x, 799), (65, 65, 65), 2)
    cv2.rectangle(image, (430, 300), (770, 510), (220, 205, 175), -1)
    cv2.rectangle(image, (430, 300), (770, 510), (245, 235, 215), 6)
    for y in range(345, 480, 35):
        cv2.line(image, (455, y), (650, y), (55, 65, 75), 5)

    report = analyze_image_quality(_encode(image), "id_card")

    assert "DOCUMENT_TOO_SMALL" not in {issue.code for issue in report.issues}


def test_non_blur_non_glare_conditions_never_appear() -> None:
    image = cv2.resize(_document_photo(), (420, 280))
    report = analyze_image_quality(_encode(image), "driver_license")

    disabled_codes = {
        "RESOLUTION_TOO_LOW",
        "TOO_DARK",
        "OVEREXPOSED",
        "LOW_CONTRAST",
        "MULTIPLE_DOCUMENTS",
        "DOCUMENT_TOO_SMALL",
        "DOCUMENT_CROPPED",
    }
    assert disabled_codes.isdisjoint(issue.code for issue in report.issues)


def test_unreadable_image_is_rejected() -> None:
    report = analyze_image_quality(b"not-an-image", "id_card")

    assert not report.accepted
    assert report.issues[0].code == "IMAGE_DECODE_FAILED"


def test_ppocr_high_confidence_text_ignores_glare_outside_text_boxes() -> None:
    image = _document_photo()
    cv2.rectangle(image, (820, 470), (1110, 700), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("姓名张三", 0.98, [180, 170, 420, 205]),
        ("公民身份号码110101199001011234", 0.97, [180, 235, 760, 275]),
        ("住址北京市朝阳区", 0.96, [180, 305, 620, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert report.accepted, report.metrics
    assert report.metrics["ocr"]["score_mean"] > 0.95


def test_id_card_back_scope_excludes_bright_background_ocr_tokens() -> None:
    image = _document_photo()
    cv2.rectangle(image, (120, 25), (760, 105), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("背景表格文字", 0.91, [150, 40, 730, 90]),
        ("中华人民共和国", 0.99, [220, 170, 680, 215]),
        ("居民身份证", 0.99, [260, 235, 620, 280]),
        ("签发机关北京市公安局", 0.99, [250, 350, 700, 390]),
        ("有效期限2018.08.13-2028.08.13", 0.99, [250, 430, 850, 475]),
        ("AMD", 0.96, [900, 730, 1080, 775]),
    )
    ppocr["quality_side"] = "DG13"

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert report.accepted, report.metrics
    scope = report.metrics["id_card_scope"]
    assert scope["strategy"] == "semantic_band"
    assert scope["original_token_count"] == 6
    assert scope["selected_token_count"] == 4


def test_id_card_high_confidence_moderate_bright_background_is_not_glare() -> None:
    image = _document_photo()
    cv2.rectangle(image, (250, 340), (430, 400), (255, 255, 255), -1)
    cv2.rectangle(image, (250, 420), (490, 485), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("中华人民共和国", 0.99, [220, 170, 680, 215]),
        ("居民身份证", 0.99, [260, 235, 620, 280]),
        ("签发机关北京市公安局", 0.99, [250, 350, 700, 390]),
        ("有效期限2018.08.13-2028.08.13", 0.99, [250, 430, 850, 475]),
    )
    ppocr["quality_side"] = "DG13"

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert report.metrics["ocr"]["weighted_glare_overlap"] >= 0.12
    assert 0.24 <= report.metrics["ocr"]["max_glare_overlap"] < 0.40
    assert report.accepted, report.metrics


def test_id_card_missing_both_required_anchors_reports_generic_key_field_issue() -> None:
    ppocr = _ppocr(
        ("无关背景文字", 0.99, [180, 170, 420, 205]),
        ("其他内容", 0.98, [180, 235, 760, 275]),
    )
    ppocr["quality_side"] = "DG12"

    report = analyze_image_quality(_encode(_document_photo()), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == [
        "ID_CARD_KEY_FIELDS_NOT_DETECTED"
    ]
    assert report.metrics["id_card_scope"]["anchors_missing"] is True


def test_ppocr_high_confidence_does_not_hide_strong_text_glare() -> None:
    image = _document_photo()
    cv2.rectangle(image, (260, 150), (390, 370), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("姓名张三", 0.99, [180, 170, 420, 205]),
        ("公民身份号码110101199001011234", 0.98, [180, 235, 760, 275]),
        ("住址北京市朝阳区", 0.97, [180, 305, 620, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]
    assert report.metrics["ocr"]["score_mean"] > 0.95


def test_ppocr_small_text_envelope_reports_document_too_small() -> None:
    image = np.full((1600, 1200, 3), 70, dtype=np.uint8)
    cv2.rectangle(image, (430, 650), (770, 860), (220, 205, 175), -1)
    ppocr = _ppocr(
        *(
            (
                f"字段{index}",
                0.98,
                [470, 680 + index * 24, 650, 688 + index * 24],
            )
            for index in range(6)
        )
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["DOCUMENT_TOO_SMALL"]


def test_ppocr_low_confidence_with_glare_overlap_reports_glare_reason() -> None:
    image = _document_photo()
    cv2.rectangle(image, (160, 150), (790, 370), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("姓名", 0.42, [180, 170, 420, 205]),
        ("公民身份号码", 0.38, [180, 235, 760, 275]),
        ("住址", 0.51, [180, 305, 620, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]
    assert report.metrics["ocr"]["max_glare_overlap"] >= 0.28


def test_ppocr_low_confidence_with_flat_text_regions_reports_blur() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 140), (800, 380), (160, 160, 160), -1)
    ppocr = _ppocr(
        ("姓名", 0.42, [180, 170, 420, 205]),
        ("证件号码", 0.39, [180, 235, 760, 275]),
        ("住址", 0.50, [180, 305, 620, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["TEXT_REGION_BLURRY"]


def test_ppocr_low_confidence_tiny_text_reports_resolution_reason() -> None:
    image = _document_photo()
    ppocr = _ppocr(
        ("姓名", 0.42, [180, 170, 420, 176]),
        ("证件号码", 0.39, [180, 235, 760, 242]),
        ("住址", 0.50, [180, 305, 620, 313]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["TEXT_RESOLUTION_TOO_LOW"]


def test_ppocr_low_confidence_dark_text_regions_reports_lighting_reason() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 140), (800, 380), (20, 20, 20), -1)
    ppocr = _ppocr(
        ("姓名", 0.42, [180, 170, 420, 205]),
        ("证件号码", 0.39, [180, 235, 760, 275]),
        ("住址", 0.50, [180, 305, 620, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["TEXT_REGION_TOO_DARK"]


def test_ppocr_unexplained_severe_low_confidence_uses_generic_reason() -> None:
    image = _document_photo()
    ppocr = _ppocr(
        ("姓名", 0.35, [180, 165, 500, 205]),
        ("证件号码", 0.40, [180, 235, 760, 275]),
        ("住址", 0.45, [180, 305, 700, 345]),
    )

    report = analyze_image_quality(_encode(image), "id_card", ppocr)

    assert [issue.code for issue in report.issues] == ["QUALITY_NOT_ACCEPTABLE"]
    assert report.issues[0].message == "图片质量不通过，请重新拍摄后再试"


def test_driver_with_no_detected_text_is_not_a_quality_rejection() -> None:
    report = analyze_image_quality(
        _encode(_document_photo()),
        "driver_license",
        {"tokens": []},
    )

    disposition = report.metrics["driver_license_quality_disposition"]
    assert report.accepted, report.metrics
    assert disposition["exclude_from_quality_rejected"] is True
    assert disposition["reason"] == "driver_license_not_recognized"


def test_driver_license_main_page_is_detected_without_secondary_page() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [180, 130, 760, 180]),
        ("姓名张三", 0.99, [180, 220, 420, 260]),
        ("准驾车型C1", 0.98, [180, 380, 430, 420]),
        ("有效期限2022-01-01至2032-01-01", 0.98, [180, 470, 780, 515]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "driver_license", ppocr)

    scope = report.metrics["driver_license_scope"]
    assert scope["side"] == "main"
    assert [band["page"] for band in scope["bands"]] == ["main"]


def test_driver_small_text_rejects_wide_two_page_envelope() -> None:
    image = np.full((2500, 1200, 3), 150, dtype=np.uint8)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [100, 1000, 700, 1020]),
        ("证号110101199001011234", 0.99, [150, 1040, 650, 1060]),
        ("姓名张三", 0.99, [150, 1080, 350, 1100]),
        ("出生日期1990-01-01", 0.99, [150, 1120, 550, 1140]),
        ("初次领证日期2020-01-01", 0.99, [150, 1160, 650, 1180]),
        ("准驾车型C1", 0.99, [150, 1200, 400, 1220]),
        ("有效期限2020-01-01至2030-01-01", 0.99, [150, 1240, 1100, 1260]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    scale = report.metrics["driver_license_document_scale"]
    assert report.metrics["ocr"]["text_envelope_area_ratio"] > 0.05
    assert scale["median_text_height_ratio"] < scale["minimum_text_height_ratio"]
    assert scale["document_too_small"] is True
    assert [issue.code for issue in report.issues] == ["DOCUMENT_TOO_SMALL"]

    electronic_ppocr = {
        **ppocr,
        "text": f"{ppocr.get('text', '')}\n电子驾驶证主页",
    }
    electronic_report = analyze_image_quality(
        _encode(image),
        "driver_license",
        electronic_ppocr,
    )
    assert electronic_report.metrics["driver_license_document_scale"][
        "document_too_small"
    ] is True
    assert electronic_report.metrics["driver_license_quality_disposition"][
        "reason"
    ] == "electronic_driver_license"
    assert electronic_report.accepted, electronic_report.metrics


def test_driver_license_secondary_page_is_detected_without_main_page() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证副页", 0.99, [180, 130, 800, 180]),
        ("姓名张三", 0.99, [180, 220, 420, 260]),
        ("档案编号110101123456", 0.99, [180, 330, 620, 375]),
        ("记录", 0.99, [180, 440, 300, 480]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "driver_license", ppocr)

    scope = report.metrics["driver_license_scope"]
    assert scope["side"] == "secondary"
    assert [band["page"] for band in scope["bands"]] == ["secondary"]


def test_driver_license_two_pages_are_detected_as_separate_semantic_bands() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [130, 100, 620, 145]),
        ("准驾车型C1", 0.99, [130, 300, 380, 340]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [130, 390, 650, 435]),
        ("中华人民共和国机动车驾驶证副页", 0.99, [130, 500, 700, 545]),
        ("档案编号110101123456", 0.99, [130, 620, 560, 665]),
        ("记录", 0.99, [130, 700, 250, 740]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "driver_license", ppocr)

    scope = report.metrics["driver_license_scope"]
    assert scope["side"] == "both"
    assert {band["page"] for band in scope["bands"]} == {"main", "secondary"}


def test_driver_license_high_confidence_bright_secondary_page_is_allowed() -> None:
    image = _document_photo()
    # A uniformly pale licence substrate is not a mirror-like highlight.
    cv2.rectangle(image, (120, 465), (1000, 720), (195, 225, 205), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("准驾车型C1", 0.99, [160, 300, 410, 340]),
        ("有效期限2022-01-01至2032-01-01", 0.98, [160, 390, 720, 435]),
        ("中华人民共和国机动车驾驶证副页", 0.99, [160, 500, 760, 545]),
        ("姓名张三", 0.99, [160, 570, 380, 610]),
        ("档案编号110101123456", 0.99, [160, 635, 600, 680]),
        ("记录", 0.99, [160, 690, 280, 720]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    assert report.metrics["driver_license_scope"]["side"] == "both"
    assert report.accepted, report.metrics


def test_driver_license_generic_ocr_glare_requires_semantic_field_evidence() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.42, [180, 140, 720, 185]),
        ("姓名", 0.38, [180, 230, 340, 270]),
        ("准驾车型", 0.45, [180, 350, 430, 390]),
        ("有效期限", 0.40, [180, 450, 430, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    assert report.accepted, report.metrics
    assert report.metrics["driver_license_semantic_glare"]["triggered_count"] == 0
    assert report.metrics["driver_license_electronic"]["detected"] is False


def test_driver_license_low_confidence_field_over_glare_threshold_is_rejected() -> None:
    image = _document_photo()
    for y in (230, 246, 262):
        cv2.line(image, (190, y), (670, y + 8), (255, 255, 255), 8)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("证号110101199001011234", 0.72, [180, 225, 680, 270]),
        ("准驾车型C1", 0.99, [180, 350, 430, 390]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [180, 450, 760, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    field_glare = report.metrics["driver_license_field_glare"]
    assert field_glare["threshold"] == 0.08
    assert field_glare["confidence_threshold"] == 0.80
    assert field_glare["max_overlap"] >= 0.08
    assert field_glare["triggered_fields"][0]["text"] == "证号110101199001011234"
    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]


def test_driver_license_high_confidence_key_field_glare_is_rejected() -> None:
    image = _document_photo()
    for y in (230, 246, 262):
        cv2.line(image, (190, y), (670, y + 8), (255, 255, 255), 8)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("证号110101199001011234", 0.99, [180, 225, 680, 270]),
        ("准驾车型C1", 0.99, [180, 350, 430, 390]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [180, 450, 760, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    field_glare = report.metrics["driver_license_field_glare"]
    assert field_glare["candidate_count"] >= 1
    assert field_glare["max_overlap"] < field_glare["severe_overlap_threshold"]
    assert field_glare["triggered_count"] == 0
    assert field_glare["readable_glare_count"] >= 1
    key_field_glare = report.metrics["driver_license_semantic_glare"]
    assert key_field_glare["decision_basis"] == (
        "specular_highlight_or_texture_washout"
    )
    assert key_field_glare["triggered_fields"][0]["field"] == "certificate_number"
    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]


def test_driver_license_bright_connected_glare_is_rejected_when_ocr_is_readable() -> None:
    image = _document_photo()
    cv2.rectangle(image, (170, 215), (690, 280), (150, 150, 150), -1)
    cv2.rectangle(image, (180, 220), (420, 270), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("证号110101199001011234", 0.99, [180, 225, 680, 270]),
        ("准驾车型C1", 0.99, [180, 350, 430, 390]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [180, 450, 760, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    semantic = report.metrics["driver_license_semantic_glare"]
    certificate = semantic["triggered_fields"][0]
    assert certificate["field"] == "certificate_number"
    assert certificate["box_glare_metrics"][0]["specular_component_supported"] is True
    assert certificate["box_glare_metrics"][0][
        "specular_component_brightness_delta"
    ] >= semantic["specular_min_brightness_delta"]
    assert semantic["triggered_count"] == 1
    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]


def test_driver_license_background_ocr_glare_does_not_trigger_field_rule() -> None:
    image = _document_photo()
    cv2.rectangle(image, (820, 20), (1110, 90), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("屏幕背景文字", 0.32, [830, 30, 1080, 80]),
        ("中华人民共和国机动车驾驶证", 0.99, [160, 150, 720, 195]),
        ("证号110101199001011234", 0.99, [180, 240, 680, 285]),
        ("准驾车型C1", 0.99, [180, 360, 430, 400]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [180, 460, 760, 505]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    assert report.metrics["driver_license_field_glare"]["triggered_count"] == 0
    assert report.accepted, report.metrics


def test_driver_address_with_low_glare_is_allowed_without_text_validation() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (300, 190), (330, 250), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南城市高", 0.99, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    assert metrics["threshold"] == 0.08
    assert metrics["decision_basis"] == (
        "specular_highlight_or_texture_washout"
    )
    assert metrics["contiguous_glare_threshold"] == 0.03
    assert metrics["texture_loss_threshold"] == 0.18
    assert metrics["contiguous_texture_loss_threshold"] == 0.08
    assert metrics["padding_follows_text_axis"] is True
    assert metrics["horizontal_padding_ratio"] == 0.20
    assert metrics["minimum_horizontal_padding_heights"] == 1.50
    assert metrics["vertical_padding_ratio"] == 0.10
    assert metrics["triggered_count"] == 0
    assert "valid" not in metrics["fields"][0]
    assert "validation_reasons" not in metrics["fields"][0]


def test_driver_address_with_high_glare_is_rejected_without_text_validation() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (300, 190), (450, 250), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南省永城市高庄镇王庄村", 0.99, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    assert metrics["triggered_count"] == 1
    assert metrics["triggered_fields"][0]["field"] == "address"
    assert metrics["triggered_fields"][0][
        "max_contiguous_glare_overlap"
    ] >= metrics["contiguous_glare_threshold"]
    assert metrics["triggered_fields"][0]["texts"] == [
        "河南省永城市高庄镇王庄村"
    ]


def test_driver_address_texture_washout_is_rejected() -> None:
    specular_mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(specular_mask, (220, 205), (230, 215), 1, -1)
    texture_loss_mask = np.zeros_like(specular_mask)
    cv2.rectangle(texture_loss_mask, (220, 195), (520, 245), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南省永城市高庄镇王庄村", 0.97, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
    )

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["max_glare_overlap"] < metrics["threshold"]
    assert address["max_glare_overlap"] >= metrics[
        "texture_loss_min_specular_evidence"
    ]
    assert address["max_texture_loss_overlap"] >= metrics[
        "texture_loss_threshold"
    ]
    assert address["max_contiguous_texture_loss_overlap"] >= metrics[
        "contiguous_texture_loss_threshold"
    ]
    assert metrics["triggered_count"] == 1


def test_driver_readable_complete_address_suppresses_texture_washout() -> None:
    specular_mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(specular_mask, (220, 205), (230, 215), 1, -1)
    texture_loss_mask = np.zeros_like(specular_mask)
    cv2.rectangle(texture_loss_mask, (220, 195), (520, 245), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南省永城市高庄镇王庄村", 0.99, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
    )

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["texture_washout_candidate"] is True
    assert address["reliably_readable_value"] is True
    assert address["texture_washout_suppressed_by_readable_value"] is True
    assert metrics["triggered_count"] == 0


def test_partial_driver_address_projects_roi_along_text_direction() -> None:
    specular_mask = np.zeros((300, 600), dtype=np.uint8)
    texture_loss_mask = np.zeros_like(specular_mask)
    grayscale = np.full_like(specular_mask, 100, dtype=np.uint8)
    cv2.rectangle(texture_loss_mask, (210, 100), (310, 130), 1, -1)
    cv2.rectangle(grayscale, (210, 100), (310, 130), 230, -1)
    cv2.rectangle(specular_mask, (225, 105), (238, 118), 1, -1)
    tokens = _ppocr(
        ("\u4f4f\u5740", 0.99, [30, 100, 80, 130]),
        ("\u6cb3\u5357\u7701", 0.99, [100, 100, 160, 130]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
        grayscale,
    )

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["box_count"] == 2
    assert any(
        box["texture_specular_spatially_linked"]
        for box in address["box_glare_metrics"]
    )
    assert metrics["triggered_count"] == 1


def test_driver_address_is_inferred_when_glare_erases_its_label() -> None:
    mask = np.zeros((300, 600), dtype=np.uint8)
    tokens = _ppocr(
        (
            "\u5357\u7701\u65b0\u91ce\u53bf\u6b6a\u5b50\u9547\u9a6c\u6e56\u6751\u6f58\u84258\u7ec4",
            0.93,
            [100, 100, 430, 135],
        ),
        (
            "\u81ea2021\u5e7408\u670816\u65e5\u81f3\u6709\u6548\u8d77\u59cb\u65e5\u671f\u6709\u6548",
            0.99,
            [100, 180, 500, 215],
        ),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["texts"] == [
        "\u5357\u7701\u65b0\u91ce\u53bf\u6b6a\u5b50\u9547\u9a6c\u6e56\u6751\u6f58\u84258\u7ec4"
    ]


def test_label_backed_driver_number_discards_unlabeled_portrait_duplicate() -> None:
    tokens = _ppocr(
        ("证号", 0.99, [30, 100, 80, 130]),
        ("320925196910095837", 0.99, [90, 100, 350, 130]),
        ("320925196910095837", 1.0, [400, 220, 560, 240]),
    )["tokens"]

    assignment = _driver_license_field_assignments(tokens)["certificate_number"]

    assert len(assignment["values"]) == 1
    assert assignment["values"][0]["bbox"] == [90, 100, 350, 130]
    assert assignment["values"][0].get("inferred") is not True


def test_partial_driver_valid_period_projects_toward_missing_start_date() -> None:
    projection = _driver_partial_valid_period_projection(
        {
            "text": "0304至20320304",
            "field_value": "0304至20320304",
            "bbox": [300, 100, 500, 130],
        }
    )

    assert projection is not None
    assert projection["projection_direction"] == "before"
    assert projection["bbox"] == [60.0, 100.0, 300.0, 130.0]


def test_partial_driver_valid_period_projects_missing_end_date_region() -> None:
    specular_mask = np.zeros((300, 700), dtype=np.uint8)
    texture_loss_mask = np.zeros_like(specular_mask)
    grayscale = np.full_like(specular_mask, 100, dtype=np.uint8)
    cv2.rectangle(texture_loss_mask, (330, 100), (450, 130), 1, -1)
    cv2.rectangle(grayscale, (330, 100), (450, 130), 230, -1)
    cv2.rectangle(specular_mask, (350, 105), (365, 118), 1, -1)
    tokens = _ppocr(
        ("\u6709\u6548\u671f\u9650", 0.99, [30, 100, 120, 130]),
        ("2021-08-193", 0.98, [140, 100, 300, 130]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
        grayscale,
    )

    valid_period = next(
        field for field in metrics["fields"] if field["field"] == "valid_period"
    )
    assert valid_period["actionable"] is True
    assert valid_period["texts"] == ["202108193"]
    assert valid_period["box_count"] == 2
    assert metrics["triggered_count"] == 1


def test_partial_driver_field_box_uses_character_height_padding() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    # The highlight is outside the partial OCR value box [200, 200, 240, 240]
    # but inside the 1.5-character-height horizontal expansion.
    cv2.rectangle(mask, (150, 205), (185, 235), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南省", 0.99, [200, 200, 240, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["max_glare_overlap"] >= metrics["threshold"]
    assert address["max_contiguous_glare_overlap"] >= metrics[
        "contiguous_glare_threshold"
    ]
    assert metrics["triggered_count"] == 1


def test_rotated_driver_field_box_expands_along_text_axis() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    # The highlight sits below a vertical OCR box. Axis-aware padding includes
    # it; horizontal-only padding would stop near y=530.
    cv2.rectangle(mask, (200, 540), (240, 605), 1, -1)
    tokens = _ppocr(
        ("住址", 0.99, [200, 100, 240, 180]),
        ("河南省永城市高庄镇", 0.99, [200, 200, 240, 500]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["max_glare_overlap"] > 0.03
    assert address["max_contiguous_glare_overlap"] > 0.03


def test_fragmented_driver_security_pattern_is_not_treated_as_glare() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    # Dense but disconnected bright dots model a pale security pattern. Their
    # total area exceeds 8%, while no individual component approaches 3%.
    for y in range(198, 243, 7):
        for x in range(105, 796, 7):
            mask[y : y + 3, x : x + 3] = 1
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南省永城市高庄镇王庄村", 0.99, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    address = next(field for field in metrics["fields"] if field["field"] == "address")
    assert address["max_glare_overlap"] >= metrics["threshold"]
    assert address["max_contiguous_glare_overlap"] < metrics[
        "contiguous_glare_threshold"
    ]
    assert metrics["triggered_count"] == 0


def test_multiple_elongated_highlights_crossing_driver_text_are_rejected() -> None:
    image = _document_photo()
    for x in (230, 290, 350, 410):
        cv2.line(image, (x, 205), (x + 8, 330), (255, 255, 255), 5)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("证号110101199001011234", 0.99, [180, 225, 680, 270]),
        ("姓名张三", 0.99, [180, 285, 430, 325]),
        ("准驾车型C1", 0.99, [180, 350, 430, 390]),
        ("有效期限2022-01-01至2032-01-01", 0.99, [180, 450, 760, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    elongated = report.metrics["driver_license_elongated_text_glare"]
    assert elongated["component_count"] >= elongated["minimum_component_count"]
    assert elongated["total_area_ratio"] >= elongated[
        "minimum_total_area_ratio"
    ]
    assert elongated["detected"] is True
    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]


def test_driver_physical_glare_fallback_rejects_two_text_crossing_streaks() -> None:
    metrics = _driver_physical_glare_fallback_metrics(
        glare_ratio=0.0203,
        elongated_glare_count=2,
        elongated_glare_ratio=0.0034,
        ocr_metrics={
            "max_glare_overlap": 0.1049,
            "glare_affected_token_count": 0,
        },
        field_glare={"candidate_count": 2, "max_overlap": 0.1049},
    )

    assert metrics["detected"] is True
    assert metrics["multiple_streaks"] is True
    assert metrics["decision_basis"] == "multiple_streaks_crossing_text"


def test_driver_physical_glare_fallback_ignores_generic_strong_text_overlap() -> None:
    metrics = _driver_physical_glare_fallback_metrics(
        glare_ratio=0.0254,
        elongated_glare_count=0,
        elongated_glare_ratio=0.0,
        ocr_metrics={
            "max_glare_overlap": 0.6689,
            "glare_affected_token_count": 5,
        },
        field_glare={"candidate_count": 0, "max_overlap": 0.0408},
    )

    assert metrics["detected"] is False
    assert metrics["strong_text_glare"] is False
    assert metrics["decision_basis"] is None


def test_driver_physical_glare_fallback_ignores_isolated_highlight() -> None:
    metrics = _driver_physical_glare_fallback_metrics(
        glare_ratio=0.021,
        elongated_glare_count=1,
        elongated_glare_ratio=0.0025,
        ocr_metrics={
            "max_glare_overlap": 0.48,
            "glare_affected_token_count": 1,
        },
        field_glare={"candidate_count": 1, "max_overlap": 0.09},
    )

    assert metrics["detected"] is False


def test_driver_large_area_glare_fallback_rejects_broad_text_glare() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (180, 190), (420, 280), 1, -1)
    tokens = _ppocr(
        ("姓名张三", 0.99, [170, 180, 310, 230]),
        ("住址河南省永城市", 0.99, [300, 225, 570, 275]),
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.026,
        largest_glare_ratio=0.015,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 5},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
    )

    assert metrics["detected"] is True
    assert metrics["recognized_document_glare"] is True
    assert metrics["decision_basis"] == "large_specular_region_crossing_text"


def test_driver_large_area_glare_fallback_recovers_degraded_driver() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (360, 240), (500, 350), 1, -1)

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        _ppocr(("D", 0.36, [60, 60, 90, 100]))["tokens"],
        1.0,
        glare_ratio=0.0195,
        largest_glare_ratio=0.0129,
        ocr_metrics={"token_count": 8, "glare_affected_token_count": 0},
        driver_scope={"side": "unknown"},
        weak_physical_evidence={"vehicle_classes": ["D"]},
    )

    assert metrics["detected"] is True
    assert metrics["degraded_document_glare"] is True
    assert metrics["decision_basis"] == (
        "large_specular_region_with_degraded_driver_evidence"
    )


def test_driver_large_area_glare_fallback_ignores_small_highlight() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (180, 190), (195, 205), 1, -1)

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        _ppocr(("姓名张三", 0.99, [170, 180, 310, 230]))["tokens"],
        1.0,
        glare_ratio=0.03,
        largest_glare_ratio=0.02,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 8},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
    )

    assert metrics["detected"] is False


def test_driver_large_area_glare_fallback_ignores_broad_pale_background() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (180, 190), (420, 280), 1, -1)
    tokens = _ppocr(
        ("姓名张三", 0.99, [170, 180, 310, 230]),
        ("住址河南省永城市", 0.99, [300, 225, 570, 275]),
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.051,
        largest_glare_ratio=0.02,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 8},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
    )

    assert metrics["detected"] is False


def test_driver_large_area_glare_fallback_rejects_low_global_strict_streaks() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (160, 190), (430, 220), 1, -1)
    cv2.rectangle(mask, (270, 240), (580, 270), 1, -1)
    tokens = _ppocr(
        ("姓名张三", 0.99, [170, 180, 310, 230]),
        ("住址河南省永城市", 0.99, [300, 225, 570, 275]),
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.012,
        largest_glare_ratio=0.005,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 0},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
        elongated_glare_count=3,
        elongated_glare_ratio=0.011,
    )

    assert metrics["detected"] is True
    assert metrics["low_global_strict_streak_glare"] is True
    assert metrics["decision_basis"] == "low_global_strict_streaks_crossing_text"


def test_driver_large_area_glare_fallback_ignores_low_global_non_streak_highlight() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (160, 190), (430, 220), 1, -1)
    cv2.rectangle(mask, (270, 240), (580, 270), 1, -1)
    tokens = _ppocr(
        ("姓名张三", 0.99, [170, 180, 310, 230]),
        ("住址河南省永城市", 0.99, [300, 225, 570, 275]),
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.0082,
        largest_glare_ratio=0.0025,
        ocr_metrics={"token_count": 42, "glare_affected_token_count": 2},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
        elongated_glare_count=0,
        elongated_glare_ratio=0.0,
    )

    assert metrics["detected"] is False
    assert metrics["low_global_strict_streak_glare"] is False


def test_driver_large_area_glare_fallback_rejects_multiple_strict_streaks() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (140, 180), (680, 260), 1, -1)
    tokens = _ppocr(
        *(
            (f"字段{index}", 0.99, [160 + index * 90, 190, 240 + index * 90, 240])
            for index in range(5)
        )
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.020,
        largest_glare_ratio=0.003,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 0},
        driver_scope={"side": "secondary"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
        elongated_glare_count=4,
        elongated_glare_ratio=0.0067,
    )

    assert metrics["detected"] is True
    assert metrics["multiple_strict_streak_glare"] is True
    assert metrics["decision_basis"] == "multiple_strict_streaks_crossing_text"


def test_driver_large_area_glare_fallback_rejects_overwhelming_text_glare() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (140, 180), (680, 260), 1, -1)
    tokens = _ppocr(
        *(
            (f"字段{index}", 0.99, [160 + index * 90, 190, 240 + index * 90, 240])
            for index in range(5)
        )
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.032,
        largest_glare_ratio=0.012,
        ocr_metrics={"token_count": 20, "glare_affected_token_count": 1},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": ["C1"]},
    )

    assert metrics["detected"] is True
    assert metrics["overwhelming_strict_text_glare"] is True
    assert metrics["decision_basis"] == "overwhelming_strict_glare_crossing_text"


def test_driver_large_area_glare_fallback_rejects_degraded_secondary_page() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (100, 160), (650, 250), 1, -1)
    tokens = _ppocr(
        *(
            (f"残存文字{index}", 0.90, [130 + index * 90, 175, 220 + index * 90, 235])
            for index in range(5)
        )
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.028,
        largest_glare_ratio=0.016,
        ocr_metrics={"token_count": 11, "glare_affected_token_count": 0},
        driver_scope={"side": "secondary"},
        weak_physical_evidence={"vehicle_classes": []},
        semantic_glare_metrics={"fields": []},
    )

    assert metrics["detected"] is True
    assert metrics["degraded_secondary_page_glare"] is True
    assert metrics["decision_basis"] == (
        "degraded_secondary_page_under_broad_glare"
    )


def test_driver_large_area_glare_fallback_rejects_degraded_main_page() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (260, 150), 1, -1)
    tokens = _ppocr(
        ("中华人民共和国机动车驾驶证副页", 0.99, [650, 600, 1050, 650]),
        ("姓名张三", 0.99, [680, 660, 820, 700]),
    )["tokens"]
    fields = [
        {"field": "name", "label_count": 1, "texts": ["张三"]},
        {
            "field": "archive_number",
            "label_count": 1,
            "texts": ["123456789012"],
        },
    ]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.0172,
        largest_glare_ratio=0.0132,
        ocr_metrics={"token_count": 18, "glare_affected_token_count": 0},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": []},
        semantic_glare_metrics={"fields": fields},
    )

    assert metrics["detected"] is True
    assert metrics["degraded_main_page_glare"] is True
    assert metrics["decision_basis"] == (
        "degraded_main_page_beside_secondary_page"
    )


def test_driver_large_area_glare_fallback_ignores_reverse_page_without_title() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (260, 150), 1, -1)
    tokens = _ppocr(
        ("准驾车型代号规定", 0.99, [650, 600, 1050, 650]),
        ("记录", 0.99, [680, 660, 820, 700]),
    )["tokens"]

    metrics = _driver_large_area_glare_fallback_metrics(
        mask,
        tokens,
        1.0,
        glare_ratio=0.0172,
        largest_glare_ratio=0.0132,
        ocr_metrics={"token_count": 18, "glare_affected_token_count": 0},
        driver_scope={"side": "both"},
        weak_physical_evidence={"vehicle_classes": []},
        semantic_glare_metrics={
            "fields": [
                {
                    "field": "archive_number",
                    "label_count": 1,
                    "texts": ["123456789012"],
                }
            ]
        },
    )

    assert metrics["detected"] is False
    assert metrics["degraded_main_page_glare"] is False


def test_driver_weak_physical_evidence_requires_three_independent_signals() -> None:
    detected = _driver_weak_physical_evidence(
        _ppocr(
            ("C1", 0.76, [20, 20, 50, 30]),
            ("中国/CHN", 0.84, [60, 20, 110, 30]),
            ("1997-11-09", 0.63, [120, 20, 190, 30]),
            ("2022-08-25", 0.74, [200, 20, 270, 30]),
        )["tokens"]
    )
    unrelated = _driver_weak_physical_evidence(
        _ppocr(
            ("营业执照", 0.99, [20, 20, 80, 30]),
            ("2022-08-25", 0.99, [90, 20, 160, 30]),
            ("2032-08-25", 0.99, [170, 20, 240, 30]),
        )["tokens"]
    )
    degraded_without_nationality = _driver_weak_physical_evidence(
        _ppocr(
            ("C1", 0.76, [20, 20, 50, 30]),
            ("1997-11-09", 0.63, [60, 20, 130, 30]),
            ("2016-08-25", 0.74, [140, 20, 210, 30]),
            ("2022-08-25", 0.71, [220, 20, 290, 30]),
        )["tokens"]
    )
    vehicle_class_table = _driver_weak_physical_evidence(
        _ppocr(
            ("A1", 0.98, [20, 20, 50, 30]),
            ("C1", 0.98, [60, 20, 90, 30]),
            ("C2", 0.98, [100, 20, 130, 30]),
            ("1997-11-09", 0.98, [140, 20, 210, 30]),
            ("2016-08-25", 0.98, [220, 20, 290, 30]),
            ("2022-08-25", 0.98, [300, 20, 370, 30]),
        )["tokens"]
    )

    assert detected["detected"] is True
    assert degraded_without_nationality["detected"] is True
    assert unrelated["detected"] is False
    assert vehicle_class_table["detected"] is False


def test_degraded_driver_evidence_keeps_document_too_small_rejection() -> None:
    image = _document_photo()
    ppocr = _ppocr(
        ("C1", 0.76, [100, 300, 130, 306]),
        ("中国/CHN", 0.84, [160, 300, 230, 306]),
        ("1997-11-09", 0.63, [260, 300, 340, 306]),
        ("2016-08-25", 0.74, [370, 300, 450, 306]),
        ("2022-08-25", 0.71, [480, 300, 560, 306]),
        ("41148119", 0.70, [590, 300, 660, 306]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    disposition = report.metrics["driver_license_quality_disposition"]
    assert disposition["recognizable_physical_license"] is True
    assert disposition["weak_physical_evidence"]["detected"] is True
    assert [issue.code for issue in report.issues] == ["DOCUMENT_TOO_SMALL"]


def test_invalid_driver_nationality_with_glare_is_diagnostic_only() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (300, 190), (360, 250), 1, -1)
    tokens = _ppocr(
        ("国籍", 0.99, [100, 200, 180, 240]),
        ("中CHN", 0.94, [200, 200, 500, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    nationality = next(
        field for field in metrics["fields"] if field["field"] == "nationality"
    )
    assert nationality["actionable"] is False
    assert nationality["max_glare_overlap"] >= metrics["threshold"]
    assert "valid" not in nationality
    assert "validation_reasons" not in nationality
    assert metrics["triggered_count"] == 0


def test_invalid_driver_field_without_glare_is_not_attributed_to_glare() -> None:
    mask = np.zeros((800, 1200), dtype=np.uint8)
    tokens = _ppocr(
        ("住址", 0.99, [100, 200, 180, 240]),
        ("河南城市高", 0.99, [200, 200, 700, 240]),
    )["tokens"]

    metrics = _driver_license_semantic_glare_metrics(mask, tokens, 1.0)

    assert metrics["fields"][0]["max_glare_overlap"] == 0.0
    assert metrics["triggered_count"] == 0


def test_electronic_driver_license_keywords_bypass_quality_rejection() -> None:
    keywords = (
        "\u7535\u5b50",
        "\u626b\u7801",
        "\u751f\u6210",
        "\u72b6\u6001",
        "\u4ee3\u53f7",
        "\u89c4\u5b9a",
        "\u4e3b\u9875",
        "\u5237\u65b0",
    )

    for keyword in keywords:
        image = _document_photo()
        cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
        ppocr = _ppocr(
            (keyword, 0.31, [180, 140, 420, 185]),
            ("\u65e0\u6cd5\u8fa8\u8ba4", 0.28, [180, 240, 520, 285]),
        )

        report = analyze_image_quality(_encode(image), "driver_license", ppocr)

        electronic = report.metrics["driver_license_electronic"]
        disposition = report.metrics["driver_license_quality_disposition"]
        assert report.accepted, (keyword, report.metrics)
        assert electronic["detected"] is True
        assert keyword in electronic["matched_keywords"]
        assert disposition["exclude_from_quality_rejected"] is True
        assert disposition["reason"] == "electronic_driver_license"


def test_electronic_driver_license_can_be_detected_from_ocr_full_text() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(("\u65e0\u6cd5\u8fa8\u8ba4", 0.28, [180, 140, 520, 185]))
    ppocr["text"] = "\u8bf7\u5237\u65b0\u7535\u5b50\u9a7e\u9a76\u8bc1\u4e3b\u9875"

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    electronic = report.metrics["driver_license_electronic"]
    assert report.accepted, report.metrics
    assert electronic["matched_keywords"] == ["\u7535\u5b50", "\u4e3b\u9875", "\u5237\u65b0"]


def test_unrecognized_non_driver_document_is_excluded_from_quality_rejections() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("营业执照", 0.99, [180, 140, 420, 185]),
        ("统一社会信用代码91410100MA12345678", 0.98, [180, 240, 760, 285]),
        ("110101199001011234", 0.99, [180, 350, 560, 395]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    disposition = report.metrics["driver_license_quality_disposition"]
    assert report.metrics["driver_license_scope"]["side"] == "unknown"
    assert disposition["recognizable_physical_license"] is False
    assert disposition["label_backed_value_fields"] == []
    assert disposition["exclude_from_quality_rejected"] is True
    assert disposition["reason"] == "driver_license_not_recognized"
    assert report.accepted, report.metrics


def test_generic_label_backed_fields_do_not_recognize_driver_license() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("姓名", 0.99, [180, 140, 260, 180]),
        ("张三", 0.99, [300, 140, 380, 180]),
        ("性别", 0.99, [180, 230, 260, 270]),
        ("男", 0.99, [300, 230, 340, 270]),
        ("国籍", 0.99, [180, 320, 260, 360]),
        ("中国", 0.99, [300, 320, 380, 360]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    disposition = report.metrics["driver_license_quality_disposition"]
    assert disposition["recognizable_physical_license"] is False
    assert disposition["driver_specific_label_fields"] == []
    assert disposition["reason"] == "driver_license_not_recognized"
    assert report.accepted, report.metrics


def test_physical_driver_license_generic_words_do_not_create_semantic_glare() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证副页", 0.40, [160, 130, 760, 175]),
        ("准驾车型代号规定", 0.42, [180, 225, 620, 270]),
        ("档案编号110101123456", 0.38, [180, 350, 620, 395]),
        ("记录", 0.41, [180, 450, 320, 495]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    electronic = report.metrics["driver_license_electronic"]
    assert electronic["detected"] is False
    assert electronic["candidate_keywords"] == ["代号", "规定"]
    assert electronic["physical_page_side"] == "both"
    assert report.accepted, report.metrics
    assert report.metrics["driver_license_semantic_glare"]["triggered_count"] == 0


def test_physical_page_anchors_with_electronic_marker_are_electronic() -> None:
    image = _document_photo()
    ppocr = _ppocr(
        ("电子", 0.99, [80, 40, 180, 80]),
        ("中华人民共和国机动车驾驶证", 0.99, [160, 130, 720, 175]),
        ("证号110101199001011234", 0.99, [180, 225, 680, 270]),
        ("准驾车型C1", 0.99, [180, 350, 430, 390]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    electronic = report.metrics["driver_license_electronic"]
    disposition = report.metrics["driver_license_quality_disposition"]
    assert electronic["physical_page_side"] == "main"
    assert electronic["detected"] is True
    assert "电子" in electronic["matched_keywords"]
    assert disposition["reason"] == "electronic_driver_license"
    assert report.accepted, report.metrics


def test_physical_page_with_multiple_screen_ui_keywords_can_be_electronic() -> None:
    image = _document_photo()
    cv2.rectangle(image, (150, 120), (800, 530), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车驾驶证", 0.35, [160, 130, 720, 175]),
        ("请刷新状态", 0.34, [180, 225, 520, 270]),
        ("无法辨认", 0.30, [180, 350, 520, 395]),
    )

    report = analyze_image_quality(_encode(image), "driver_license", ppocr)

    electronic = report.metrics["driver_license_electronic"]
    assert electronic["detected"] is True
    assert electronic["matched_keywords"] == ["状态", "刷新"]
    assert report.accepted, report.metrics


def test_readable_diffuse_bright_background_is_not_treated_as_field_glare() -> None:
    field_metrics = {
        "field_token_count": 40,
        "triggered_count": 38,
        "triggered_fields": [{"text": "status", "overlap": 0.62}],
    }
    ocr_metrics = {
        "scored_token_count": 40,
        "score_mean": 0.9844,
        "score_p25": 0.9861,
        "low_score_ratio": 0.0,
        "low_score_char_ratio": 0.0,
        "glare_affected_low_score_count": 0,
    }

    refined = _refine_field_glare_metrics(field_metrics, ocr_metrics)

    assert refined["candidate_count"] == 38
    assert refined["candidate_ratio"] == 0.95
    assert refined["suppressed_as_diffuse_bright_background"] is True
    assert refined["triggered_count"] == 0


def test_local_effective_field_glare_is_not_suppressed_by_page_metrics() -> None:
    field_metrics = {
        "field_token_count": 8,
        "triggered_count": 1,
        "triggered_fields": [{"text": "licence-number", "overlap": 0.12}],
    }
    ocr_metrics = {
        "scored_token_count": 8,
        "score_mean": 0.99,
        "score_p25": 0.98,
        "low_score_ratio": 0.0,
        "low_score_char_ratio": 0.0,
        "glare_affected_low_score_count": 0,
    }

    refined = _refine_field_glare_metrics(field_metrics, ocr_metrics)

    assert refined["suppressed_as_diffuse_bright_background"] is False
    assert refined["triggered_count"] == 1


def test_widespread_glare_with_unreliable_ocr_is_not_suppressed() -> None:
    field_metrics = {
        "field_token_count": 10,
        "triggered_count": 8,
        "triggered_fields": [{"text": "unreadable", "overlap": 0.72}],
    }
    ocr_metrics = {
        "scored_token_count": 10,
        "score_mean": 0.58,
        "score_p25": 0.44,
        "low_score_ratio": 0.5,
        "low_score_char_ratio": 0.55,
        "glare_affected_low_score_count": 4,
    }

    refined = _refine_field_glare_metrics(field_metrics, ocr_metrics)

    assert refined["suppressed_as_diffuse_bright_background"] is False
    assert refined["triggered_count"] == 8


def test_vehicle_license_main_page_is_detected_without_secondary_page() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车行驶证", 0.99, [150, 120, 720, 165]),
        ("号牌号码粤B12345", 0.99, [170, 210, 520, 250]),
        ("所有人张三", 0.99, [170, 285, 440, 325]),
        ("车辆识别代号LSVUD6B22LN010878", 0.99, [170, 390, 760, 435]),
        ("注册日期2022-06-23", 0.99, [170, 470, 560, 515]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "vehicle_license", ppocr)

    scope = report.metrics["vehicle_license_scope"]
    assert scope["side"] == "main"
    assert [band["page"] for band in scope["bands"]] == ["main"]


def test_vehicle_license_secondary_page_is_detected_without_main_page() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车行驶证副页", 0.99, [150, 120, 760, 165]),
        ("档案编号110101123456", 0.99, [170, 220, 560, 260]),
        ("总质量1778kg", 0.99, [170, 330, 460, 370]),
        ("核定载人数5人", 0.99, [170, 420, 480, 460]),
        ("外廓尺寸4670×1806×1474mm", 0.99, [170, 510, 730, 555]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "vehicle_license", ppocr)

    scope = report.metrics["vehicle_license_scope"]
    assert scope["side"] == "secondary"
    assert [band["page"] for band in scope["bands"]] == ["secondary"]


def test_vehicle_license_two_pages_are_detected_as_separate_semantic_bands() -> None:
    ppocr = _ppocr(
        ("中华人民共和国机动车行驶证", 0.99, [130, 90, 650, 130]),
        ("所有人张三", 0.99, [130, 190, 380, 230]),
        ("车辆识别代号LSVUD6B22LN010878", 0.99, [130, 300, 700, 345]),
        ("中华人民共和国机动车行驶证副页", 0.99, [130, 470, 720, 515]),
        ("档案编号110101123456", 0.99, [130, 570, 540, 615]),
        ("总质量1778kg", 0.99, [130, 670, 420, 710]),
    )

    report = analyze_image_quality(_encode(_document_photo()), "vehicle_license", ppocr)

    scope = report.metrics["vehicle_license_scope"]
    assert scope["side"] == "both"
    assert {band["page"] for band in scope["bands"]} == {"main", "secondary"}


def test_vehicle_license_pale_stock_is_not_treated_as_field_glare() -> None:
    image = _document_photo()
    cv2.rectangle(image, (120, 455), (1000, 720), (195, 225, 205), -1)
    ppocr = _ppocr(
        ("中华人民共和国机动车行驶证", 0.99, [150, 100, 700, 145]),
        ("车辆识别代号LSVUD6B22LN010878", 0.99, [160, 280, 760, 325]),
        ("中华人民共和国机动车行驶证副页", 0.99, [150, 485, 760, 530]),
        ("档案编号110101123456", 0.99, [160, 585, 600, 630]),
        ("总质量1778kg", 0.99, [160, 670, 450, 710]),
    )

    report = analyze_image_quality(_encode(image), "vehicle_license", ppocr)

    assert report.metrics["vehicle_license_scope"]["side"] == "both"
    assert report.metrics["vehicle_license_field_glare"]["triggered_count"] == 0
    assert report.accepted, report.metrics


def test_vehicle_license_key_field_over_glare_threshold_is_rejected_regardless_of_confidence() -> None:
    image = _document_photo()
    for y in (230, 246, 262):
        cv2.line(image, (180, y), (740, y + 8), (255, 255, 255), 8)
    ppocr = _ppocr(
        ("中华人民共和国机动车行驶证", 0.99, [150, 120, 720, 165]),
        ("车辆识别代号LSVUD6B22LN010878", 0.99, [170, 225, 750, 270]),
        ("发动机号码EA88812345", 0.99, [170, 350, 620, 395]),
        ("注册日期2022-06-23", 0.99, [170, 460, 560, 505]),
    )

    report = analyze_image_quality(_encode(image), "vehicle_license", ppocr)

    field_glare = report.metrics["vehicle_license_field_glare"]
    semantic_glare = report.metrics["vehicle_license_semantic_glare"]
    assert field_glare["threshold"] == 0.08
    assert field_glare["max_overlap"] >= 0.08
    assert field_glare["triggered_count"] == 0
    assert semantic_glare["uses_ocr_confidence"] is False
    assert semantic_glare["triggered_fields"][0]["field"] == "vin"
    assert [issue.code for issue in report.issues] == ["GLARE_OCCLUDES_TEXT"]


def test_vehicle_license_background_ocr_glare_does_not_trigger_field_rule() -> None:
    image = _document_photo()
    cv2.rectangle(image, (820, 20), (1110, 90), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("屏幕背景文字", 0.31, [830, 30, 1080, 80]),
        ("中华人民共和国机动车行驶证", 0.99, [150, 140, 720, 185]),
        ("号牌号码粤B12345", 0.99, [170, 240, 520, 280]),
        ("车辆识别代号LSVUD6B22LN010878", 0.99, [170, 350, 750, 395]),
        ("注册日期2022-06-23", 0.99, [170, 470, 560, 515]),
    )

    report = analyze_image_quality(_encode(image), "vehicle_license", ppocr)

    assert report.metrics["vehicle_license_field_glare"]["triggered_count"] == 0
    assert report.metrics["vehicle_license_semantic_glare"]["triggered_count"] == 0
    assert report.accepted, report.metrics


def test_unrecognized_vehicle_license_is_not_reported_as_field_glare() -> None:
    image = _document_photo()
    cv2.rectangle(image, (170, 210), (760, 285), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("无法辨认的零散文字", 0.31, [180, 220, 750, 275]),
        ("97", 0.52, [900, 350, 980, 410]),
    )

    report = analyze_image_quality(_encode(image), "vehicle_license", ppocr)

    assert report.metrics["vehicle_license_scope"]["side"] == "unknown"
    assert report.metrics["vehicle_license_semantic_glare"]["field_count"] == 0
    assert [issue.code for issue in report.issues] == [
        "VEHICLE_LICENSE_NOT_RECOGNIZED"
    ]
    assert "未识别到行驶证有效信息" in report.issues[0].message


def test_vehicle_license_partial_value_is_located_without_format_validation() -> None:
    mask = np.zeros((260, 520), dtype=np.uint8)
    cv2.rectangle(mask, (175, 94), (285, 136), 1, -1)
    tokens = [
        {"text": "车辆识别代号", "score": 0.99, "bbox": [40, 100, 145, 130]},
        # A glare-damaged partial VIN is intentionally not a valid VIN.
        {"text": "LSVU6", "score": 0.99, "bbox": [170, 100, 290, 130]},
    ]

    metrics = _vehicle_license_semantic_glare_metrics(mask, tokens, 1.0)

    assert metrics["uses_value_validation"] is False
    assert metrics["triggered_count"] == 1
    assert metrics["triggered_fields"][0]["field"] == "vin"
    assert metrics["triggered_fields"][0]["texts"] == ["LSVU6"]


def test_vehicle_license_texture_washout_channel_detects_address_glare() -> None:
    specular_mask = np.zeros((260, 520), dtype=np.uint8)
    texture_loss_mask = np.zeros_like(specular_mask)
    cv2.rectangle(specular_mask, (195, 104), (225, 116), 1, -1)
    cv2.rectangle(texture_loss_mask, (150, 94), (345, 136), 1, -1)
    tokens = [
        {
            "text": "住址某路",
            "score": 0.99,
            "bbox": [130, 100, 330, 130],
        }
    ]

    metrics = _vehicle_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
    )

    assert metrics["triggered_count"] == 1
    field = metrics["triggered_fields"][0]
    assert field["field"] == "address"
    assert field["max_glare_overlap"] < metrics["threshold"]
    assert field["max_texture_loss_overlap"] >= metrics["texture_loss_threshold"]


def test_vehicle_license_bilingual_label_is_not_mistaken_for_field_value() -> None:
    mask = np.zeros((260, 520), dtype=np.uint8)
    cv2.rectangle(mask, (205, 94), (325, 136), 1, -1)
    tokens = [
        {"text": "所有人OWNER", "score": 0.99, "bbox": [40, 100, 145, 130]},
        {"text": "张三", "score": 0.99, "bbox": [200, 100, 330, 130]},
    ]

    metrics = _vehicle_license_semantic_glare_metrics(mask, tokens, 1.0)

    assert metrics["triggered_count"] == 1
    field = metrics["triggered_fields"][0]
    assert field["field"] == "owner"
    assert field["texts"] == ["张三"]


def test_vehicle_license_split_address_label_is_not_used_as_value() -> None:
    mask = np.zeros((240, 480), dtype=np.uint8)
    cv2.rectangle(mask, (20, 45), (90, 108), 1, -1)
    tokens = [
        {"text": "Address", "score": 0.99, "bbox": [30, 82, 110, 102]},
        {"text": "住", "score": 0.99, "bbox": [30, 50, 50, 80]},
        {"text": "址", "score": 0.99, "bbox": [55, 50, 75, 80]},
        {"text": "某市某区某路", "score": 0.99, "bbox": [140, 50, 330, 82]},
    ]

    metrics = _vehicle_license_semantic_glare_metrics(mask, tokens, 1.0)

    address = next(row for row in metrics["fields"] if row["field"] == "address")
    assert address["texts"] == ["某市某区某路"]
    assert address["box_count"] == 1
    assert metrics["triggered_count"] == 0


def test_vehicle_license_any_field_label_is_not_used_as_another_field_value() -> None:
    mask = np.zeros((240, 520), dtype=np.uint8)
    field_labels = (
        "\u4f7f\u7528\u6027\u8d28",
        "\u54c1\u724c\u578b\u53f7",
        "\u6ce8\u518c\u65e5\u671f",
        "\u6863\u6848\u7f16\u53f7",
        "\u6838\u5b9a\u8f7d\u4eba\u6570",
        "\u68c0\u9a8c\u8bb0\u5f55",
        "Vehicle Type",
        "Use Character",
        "File No.",
        "Inspection Record",
    )
    for field_label in field_labels:
        tokens = [
            {"text": "\u4f4f\u5740", "score": 0.99, "bbox": [30, 60, 90, 90]},
            # This is the nearest box, but it is another printed field name.
            {"text": field_label, "score": 0.99, "bbox": [105, 60, 205, 90]},
            {"text": "\u67d0\u5e02\u67d0\u533a\u67d0\u8def", "score": 0.99, "bbox": [220, 60, 390, 90]},
        ]

        metrics = _vehicle_license_semantic_glare_metrics(mask, tokens, 1.0)

        address = next(
            row for row in metrics["fields"] if row["field"] == "address"
        )
        assert address["texts"] == ["\u67d0\u5e02\u67d0\u533a\u67d0\u8def"]
        assert address["box_count"] == 1


def test_vehicle_license_uniform_pale_region_is_not_direct_specular_glare() -> None:
    specular_mask = np.zeros((240, 520), dtype=np.uint8)
    grayscale = np.full_like(specular_mask, 225, dtype=np.uint8)
    cv2.rectangle(specular_mask, (130, 54), (330, 96), 1, -1)
    tokens = [
        {
            "text": "\u4f4f\u5740\u67d0\u5e02\u67d0\u533a\u67d0\u8def",
            "score": 0.99,
            "bbox": [100, 60, 350, 90],
        }
    ]

    metrics = _vehicle_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        grayscale_image=grayscale,
    )

    assert metrics["triggered_count"] == 0
    box = metrics["fields"][0]["box_glare_metrics"][0]
    assert box["glare_overlap"] >= metrics["threshold"]
    assert box["specular_component_fill_ratio"] >= 0.35
    assert box["specular_component_brightness_delta"] < 15.0
    assert box["specular_component_supported"] is False


def test_vehicle_license_texture_and_highlight_must_be_spatially_linked() -> None:
    specular_mask = np.zeros((240, 480), dtype=np.uint8)
    texture_loss_mask = np.zeros_like(specular_mask)
    cv2.rectangle(specular_mask, (45, 102), (70, 138), 1, -1)
    cv2.rectangle(texture_loss_mask, (195, 100), (285, 140), 1, -1)
    tokens = [
        {"text": "住址某市某路", "score": 0.99, "bbox": [100, 100, 300, 140]}
    ]

    metrics = _vehicle_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
    )

    assert metrics["triggered_count"] == 0
    box = metrics["fields"][0]["box_glare_metrics"][0]
    assert box["texture_specular_spatially_linked"] is False


def test_vehicle_license_linked_washout_must_be_brighter_than_its_roi() -> None:
    specular_mask = np.zeros((240, 480), dtype=np.uint8)
    texture_loss_mask = np.zeros_like(specular_mask)
    grayscale = np.full_like(specular_mask, 150, dtype=np.uint8)
    cv2.rectangle(texture_loss_mask, (195, 100), (285, 140), 1, -1)
    cv2.rectangle(specular_mask, (220, 108), (250, 125), 1, -1)
    cv2.rectangle(grayscale, (195, 100), (285, 140), 230, -1)
    tokens = [
        {"text": "住址某市某路", "score": 0.99, "bbox": [100, 100, 300, 140]}
    ]

    metrics = _vehicle_license_semantic_glare_metrics(
        specular_mask,
        tokens,
        1.0,
        texture_loss_mask,
        grayscale,
    )

    assert metrics["triggered_count"] == 1
    box = metrics["triggered_fields"][0]["box_glare_metrics"][0]
    assert box["texture_specular_spatially_linked"] is True
    assert box["texture_component_brightness_delta"] >= 15.0


def test_vehicle_license_plate_labels_are_excluded_from_two_page_value_boxes() -> None:
    tokens = [
        {"text": "号牌号码", "score": 0.99, "bbox": [30, 50, 130, 80]},
        {"text": "PlaNo.", "score": 0.99, "bbox": [30, 82, 115, 102]},
        {"text": "湘AQ16S8", "score": 0.99, "bbox": [160, 50, 310, 82]},
        {"text": "号牌号码", "score": 0.99, "bbox": [30, 160, 130, 190]},
        {"text": "湘AQ16S8", "score": 0.99, "bbox": [160, 160, 310, 192]},
    ]
    label_only_glare = np.zeros((260, 520), dtype=np.uint8)
    cv2.rectangle(label_only_glare, (25, 78), (125, 108), 1, -1)

    label_metrics = _vehicle_license_semantic_glare_metrics(
        label_only_glare,
        tokens,
        1.0,
    )

    plate = next(
        row for row in label_metrics["fields"] if row["field"] == "plate_number"
    )
    assert plate["texts"] == ["湘AQ16S8", "湘AQ16S8"]
    assert plate["box_count"] == 2
    assert all(item["text"] != "PLANO" for item in plate["box_glare_metrics"])
    assert label_metrics["triggered_count"] == 0

    lower_plate_glare = np.zeros((260, 520), dtype=np.uint8)
    cv2.rectangle(lower_plate_glare, (155, 154), (315, 198), 1, -1)
    value_metrics = _vehicle_license_semantic_glare_metrics(
        lower_plate_glare,
        tokens,
        1.0,
    )

    assert value_metrics["triggered_count"] == 1
    assert value_metrics["triggered_fields"][0]["field"] == "plate_number"


def test_electronic_vehicle_license_bypasses_physical_card_glare_rule() -> None:
    image = _document_photo()
    cv2.rectangle(image, (160, 215), (760, 275), (255, 255, 255), -1)
    ppocr = _ppocr(
        ("电子行驶证", 0.99, [150, 100, 500, 150]),
        ("号牌号码鲁DLS918", 0.99, [170, 220, 520, 270]),
        ("车辆识别代号LSV******N010878", 0.99, [170, 350, 750, 395]),
        ("状态正常", 0.99, [170, 470, 400, 515]),
    )

    report = analyze_image_quality(_encode(image), "vehicle_license", ppocr)

    assert report.metrics["vehicle_license_electronic"]["detected"] is True
    assert report.accepted, report.metrics
