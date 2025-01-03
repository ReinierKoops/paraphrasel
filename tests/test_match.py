import pytest

from semanticmatch.match import (
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
        "target_word",
        "comparison_word",
        "language_code",
        "decimals",
        "outcome_above_threshold",
    ),
    [
        ("하다", "해요", "kor", 4, True),
        ("하다", "집", "kor", 3, False),
        ("하다", "늘다", "kor", None, True),
        ("to go", "went", "eng", None, True),
    ],
)
def test_compare(
    target_word: str,
    comparison_word: str,
    language_code: str,
    decimals: int,
    outcome_above_threshold: bool,
):
    """Test if Semanticmatch works for 4 cases"""
    outcome = compare(target_word, comparison_word, language_code, decimals)

    if decimals:
        assert count_decimals(outcome) <= decimals

    assert (outcome > THRESHOLD) == outcome_above_threshold


@pytest.mark.parametrize(
    (
        "target_word",
        "comparison_words",
        "language_code",
        "decimals",
        "outcome_above_thresholds",
    ),
    [
        ("하다", ["해요", "집", "늘다"], "kor", None, [True, False, True]),
    ],
)
def test_compare_multiple(
    target_word: str,
    comparison_words: list[str],
    language_code: str,
    decimals: int,
    outcome_above_thresholds: list[bool],
):
    """Test if Semanticmatch works for given case"""
    outcomes = compare_multiple(target_word, comparison_words, language_code, decimals)

    for outcome_key, above_threshold in zip(outcomes.keys(), outcome_above_thresholds):
        assert (outcomes[outcome_key] > THRESHOLD) == above_threshold


@pytest.mark.parametrize(
    (
        "target_word",
        "comparison_words",
        "language_code",
        "decimals",
        "expected_outcomes",
    ),
    [
        ("하다", ["해요", "집", "늘다"], "kor", None, ["해요", "늘다"]),
    ],
)
def test_compare_above_cutoff(
    target_word: str,
    comparison_words: list[str],
    language_code: str,
    decimals: int,
    expected_outcomes: list[str],
):
    """Test if Semanticmatch cutoff works for given case"""
    outcomes = get_above_cutoff(
        target_word, comparison_words, language_code, decimals, THRESHOLD
    )

    assert list(outcomes.keys()) == expected_outcomes


@pytest.mark.parametrize(
    (
        "target_word",
        "comparison_words",
        "language_code",
        "decimals",
        "expected_outcome",
    ),
    [
        ("love", ["affection", "loving", "live"], "eng", 4, "loving"),
    ],
)
def test_best_match(
    target_word: str,
    comparison_words: list[str],
    language_code: str,
    decimals: int,
    expected_outcome: str,
):
    """Test if Semanticmatch works for given case"""
    outcomes = get_best_match(
        target_word, comparison_words, language_code, decimals, THRESHOLD
    )

    assert list(outcomes.keys())[0] == expected_outcome
