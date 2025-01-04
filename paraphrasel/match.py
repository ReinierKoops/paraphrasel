from typing import Optional
from paraphrasel.semantic_similarity import SemanticSimilarity


def compare(
    target_word: str,
    comparison_word: str,
    language_code: str = "all",
    decimals: Optional[int] = None,
) -> float:
    """Compare target word to comparison word and get back their similarity score.

    Args:
        target_word (str): target
        comparison_word (str): comparison
        language_code (str, optional): language used/supported. Defaults to "all".
        decimals (Optional[int], optional): decimals used for score. Defaults to None.

    Returns:
        float: similarity score
    """
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
    """Compare target word to comparison words and get back their similarity scores.

    Args:
        target_word (str): target
        comparison_word (list[str]): comparisons
        language_code (str, optional): language used/supported. Defaults to "all".
        decimals (Optional[int], optional): decimals used for score. Defaults to None.

    Returns:
        list[float]: similarity scores
    """
    # Initialize the language
    similarity = SemanticSimilarity(language_code)

    # Compute the similarity score
    similarity_scores = similarity.one_to_many_similarity(target_word, comparison_words)

    # Put into a dict for better traceability
    comparison_dict = {}
    for word, score in zip(comparison_words, similarity_scores):
        # Round the score (or not)
        comparison_dict[word] = score if not decimals else round(score, decimals)

    return comparison_dict


def get_above_cutoff(
    target_word: str,
    comparison_words: list[str],
    language_code: str = "all",
    decimals: Optional[int] = None,
    cutoff: Optional[float] = None,
) -> Optional[dict[str]]:
    """Compare target word to comparison words and get back their similarity scores above cutoff.

    Args:
        target_word (str): target
        comparison_word (list[str]): comparisons
        language_code (str, optional): language used/supported. Defaults to "all".
        decimals (Optional[int], optional): decimals used for score. Defaults to None.
        cutoff (Optional[float], optional): _description_. Defaults to None.

    Returns:
        Optional[dict[str]]: similarity scores above cutoff
    """
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
    """Compare target word to comparison words and get back the word with the highest similarity score.

    Args:
        target_word (str): target
        comparison_word (list[str]): comparisons
        language_code (str, optional): language used/supported. Defaults to "all".
        decimals (Optional[int], optional): decimals used for score. Defaults to None.
        cutoff (Optional[float], optional): _description_. Defaults to None.

    Returns:
        Optional[dict[str]]: word of the highest simimlarity score
    """
    comparisons = get_above_cutoff(
        target_word, comparison_words, language_code, decimals, cutoff
    )

    # Get the dict entry with highest float value.
    best_value = max(comparisons, key=comparisons.get)
    return {best_value: comparisons[best_value]}
