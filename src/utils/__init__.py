"""Utils package for data and metrics."""

from .data import (
    load_sample_articles,
    get_article_by_title,
    get_articles_by_category,
    get_available_categories,
    get_all_titles,
    clean_text,
    validate_text,
    get_text_statistics
)

from .metrics import (
    calculate_jaccard_similarity,
    calculate_compression_ratio,
    count_words,
    count_sentences,
    calculate_average_similarity,
    calculate_length_stats,
    calculate_consistency_metrics
)

__all__ = [
    "load_sample_articles",
    "get_article_by_title",
    "get_articles_by_category",
    "get_available_categories",
    "get_all_titles",
    "clean_text",
    "validate_text",
    "get_text_statistics",
    "calculate_jaccard_similarity",
    "calculate_compression_ratio",
    "count_words",
    "count_sentences",
    "calculate_average_similarity",
    "calculate_length_stats",
    "calculate_consistency_metrics"
]
