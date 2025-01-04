import click
import json
from typing import List, Optional

from paraphrasel.match import (
    compare,
    compare_multiple,
    get_above_cutoff,
    get_best_match,
)


@click.group()
def cli():
    """Semantic Similarity CLI"""
    pass


@cli.command()
@click.argument("target_word")
@click.argument("comparison_word")
@click.option("--language", default="all", help="Language code (default: all)")
@click.option(
    "--decimals",
    type=int,
    default=None,
    help="Number of decimal places to round the similarity score",
)
def single(
    target_word: str, comparison_word: str, language: str, decimals: Optional[int]
):
    """Compare two words and return their similarity score."""
    score = compare(target_word, comparison_word, language, decimals)
    click.echo(score)


@cli.command()
@click.argument("target_word")
@click.argument("comparison_words", nargs=-1)
@click.option("--language", default="all", help="Language code (default: all)")
@click.option(
    "--decimals",
    type=int,
    default=None,
    help="Number of decimal places to round the similarity scores",
)
def multiple(
    target_word: str,
    comparison_words: List[str],
    language: str,
    decimals: Optional[int],
):
    """Compare a target word against multiple comparison words."""
    scores = compare_multiple(target_word, list(comparison_words), language, decimals)
    click.echo(json.dumps(scores, indent=2))


@cli.command()
@click.argument("target_word")
@click.argument("comparison_words", nargs=-1)
@click.option("--language", default="all", help="Language code (default: all)")
@click.option(
    "--decimals",
    type=int,
    default=None,
    help="Number of decimal places to round the similarity scores",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Cutoff value to filter similarity scores",
)
def above_cutoff(
    target_word: str,
    comparison_words: List[str],
    language: str,
    decimals: Optional[int],
    cutoff: Optional[float],
):
    """Get similarity scores above a specified cutoff."""
    results = get_above_cutoff(
        target_word, list(comparison_words), language, decimals, cutoff
    )
    if results:
        click.echo(json.dumps(results, indent=2))


@cli.command()
@click.argument("target_word")
@click.argument("comparison_words", nargs=-1)
@click.option("--language", default="all", help="Language code (default: all)")
@click.option(
    "--decimals",
    type=int,
    default=None,
    help="Number of decimal places to round the similarity scores",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Cutoff value to filter similarity scores before selecting the best match",
)
def best_match(
    target_word: str,
    comparison_words: List[str],
    language: str,
    decimals: Optional[int],
    cutoff: Optional[float],
):
    """Get the best matching word based on similarity score."""
    result = get_best_match(
        target_word, list(comparison_words), language, decimals, cutoff
    )
    if result:
        click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
