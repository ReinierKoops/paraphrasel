from typing import Optional
from paraphrasel.semantic_similarity import SemanticSimilarity


def compare(
    target_word: str,
    comparison_word: str,
    language_code: str = "all",
    decimals: Optional[int] = None,
) -> float:
    # Initialize the language
    similarity = SemanticSimilarity(language_code)

    # Compute the similarity score
    similarity_score = similarity.one_to_one_similarity(target_word, comparison_word)

    # Round the score (or not)
    if not decimals:
        return similarity_score
    return round(similarity_score, decimals)


def compare_multiple(
    target_word: str,
    comparison_words: list[str],
    language_code: str = "all",
    decimals: Optional[int] = None,
) -> dict[str]:
    # Initialize the language
    similarity = SemanticSimilarity(language_code)

    # Compute the similarity score
    similarity_scores = similarity.one_to_many_similarity(target_word, comparison_words)

    comparison_dict = {}
    for word, score in zip(comparison_words, similarity_scores):
        # Round the score (or not)
        if not decimals:
            comparison_dict[word] = score
        else:
            comparison_dict[word] = round(score, decimals)
    return comparison_dict


def get_above_cutoff(
    target_word: str,
    comparison_words: list[str],
    language_code: str = "all",
    decimals: Optional[int] = None,
    cutoff: Optional[float] = None,
) -> Optional[dict[str]]:
    # Compare all
    comparisons = compare_multiple(
        target_word, comparison_words, language_code, decimals
    )

    # Drop all which are below cutoff
    if cutoff:
        cutoff_comparisons = {}
        for key in comparisons.keys():
            if comparisons[key] >= cutoff:
                cutoff_comparisons[key] = comparisons[key]
        return cutoff_comparisons
    return comparisons


def get_best_match(
    target_word: str,
    comparison_words: list[str],
    language_code: str = "all",
    decimals: Optional[int] = None,
    cutoff: Optional[float] = None,
) -> Optional[dict[str]]:
    comparisons = get_above_cutoff(
        target_word, comparison_words, language_code, decimals, cutoff
    )

    # Get the dict entry with highest float value.
    best_value = max(comparisons, key=comparisons.get)
    return {best_value: comparisons[best_value]}
