from llm_manager.driver_archive_visual_refiner import _correct_ambiguous_eights


def test_archive_visual_refiner_corrects_only_open_ambiguous_eights() -> None:
    value = "422822748481"
    scores = [0.0] * 12
    scores[3] = 0.061  # stable, genuine 8 in the same printed row

    corrected, indexes = _correct_ambiguous_eights(value, scores)

    assert corrected == "422822740401"
    assert indexes == (8, 10)


def test_archive_visual_refiner_requires_a_reference_eight() -> None:
    value = "422822748401"

    corrected, indexes = _correct_ambiguous_eights(value, [0.0] * 12)

    assert corrected == value
    assert indexes == ()


def test_archive_visual_refiner_does_not_change_a_single_eight() -> None:
    value = "422822740401"
    scores = [0.0] * 12
    scores[3] = 0.061

    corrected, indexes = _correct_ambiguous_eights(value, scores)

    assert corrected == value
    assert indexes == ()


def test_archive_visual_refiner_rejects_broad_multi_digit_rewrites() -> None:
    value = "888822748481"
    scores = [0.0] * 12
    scores[3] = 0.061

    corrected, indexes = _correct_ambiguous_eights(value, scores)

    assert corrected == value
    assert indexes == ()
