import pytest

from paraphrasel.match import (
    compare,
    compare_multiple,
    get_above_cutoff,
    get_best_match,
)

THRESHOLD = 0.7


def count_decimals(number):
    # Convert the float to a string and split it at the decimal point
    str_number = str(number).split(".")

    # Return the length of the digits after the decimal point (if exists)
    return len(str_number[1]) if len(str_number) > 1 else 0


@pytest.mark.parametrize(
    (
        "target_wordpair",
        "comparison_wordpair",
        "language_code",
        "decimals",
        "outcome_above_threshold",
    ),
    [
        ("하다 to do", "해요 do", "all", 4, True),
        ("하다 to do", "집 house", "all", 3, False),
        ("하다 to do", "늘다 to play", "all", None, False),
        ("가다 to go", "갔어요 went", "all", None, True),
    ],
)
def test_compare(
    target_wordpair: str,
    comparison_wordpair: str,
    language_code: str,
    decimals: int,
    outcome_above_threshold: bool,
):
    """Test if paraphrasel works for 4 cases"""
    outcome = compare(target_wordpair, comparison_wordpair, language_code, decimals)

    if decimals:
        assert count_decimals(outcome) <= decimals

    assert (outcome >= THRESHOLD) == outcome_above_threshold


@pytest.mark.parametrize(
    (
        "target_wordpair",
        "comparison_wordpairs",
        "language_code",
        "decimals",
        "outcome_above_thresholds",
    ),
    [
        (
            "하다 to do",
            ["해요 to do", "집 house", "늘다 to play"],
            "all",
            None,
            [True, False, False],
        ),
    ],
)
def test_compare_multiple(
    target_wordpair: str,
    comparison_wordpairs: list[str],
    language_code: str,
    decimals: int,
    outcome_above_thresholds: list[bool],
):
    """Test if paraphrasel works for given case"""
    outcomes = compare_multiple(
        target_wordpair, comparison_wordpairs, language_code, decimals
    )

    for outcome_key, above_threshold in zip(outcomes.keys(), outcome_above_thresholds):
        assert (outcomes[outcome_key] > THRESHOLD) == above_threshold


@pytest.mark.parametrize(
    (
        "target_wordpair",
        "comparison_wordpairs",
        "language_code",
        "decimals",
        "expected_outcomes",
    ),
    [
        (
            "하다 to do",
            ["해요 to do", "집 house", "늘다 to play"],
            "all",
            None,
            ["해요 to do"],
        ),
    ],
)
def test_compare_above_cutoff(
    target_wordpair: str,
    comparison_wordpairs: list[str],
    language_code: str,
    decimals: int,
    expected_outcomes: list[str],
):
    """Test if paraphrasel cutoff works for given case"""
    outcomes = get_above_cutoff(
        target_wordpair, comparison_wordpairs, language_code, decimals, THRESHOLD
    )

    assert list(outcomes.keys()) == expected_outcomes


@pytest.mark.parametrize(
    (
        "target_wordpair",
        "comparison_wordpairs",
        "language_code",
        "decimals",
        "expected_outcome",
    ),
    [
        (
            "사랑 love",
            ["애정 affection", "좋아하는 loving", "살다 to live"],
            "all",
            4,
            "애정 affection",
        ),
        (
            "하다 to do",
            ["해요 to do", "집 house", "늘다 to play"],
            "all",
            None,
            "해요 to do",
        ),
        (
            "가다 to go",
            ["가요 to go", "갔어요 went", "오다 to come"],
            "all",
            None,
            "가요 to go",
        ),
    ],
)
def test_best_match(
    target_wordpair: str,
    comparison_wordpairs: list[str],
    language_code: str,
    decimals: int,
    expected_outcome: str,
):
    """Test if paraphrasel works for given case"""
    outcomes = get_best_match(
        target_wordpair, comparison_wordpairs, language_code, decimals, THRESHOLD
    )

    assert list(outcomes.keys())[0] == expected_outcome
