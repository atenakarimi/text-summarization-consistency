"""
Metrics Module

Provides evaluation metrics for text summarization and consistency analysis.
"""

import numpy as np
from typing import List, Set
import re


def tokenize_text(text: str) -> Set[str]:
    """
    Simple tokenization by splitting on whitespace and punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Set of lowercase tokens
    """
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(tokens)


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity coefficient between two texts.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    where A and B are sets of words in each text.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Two empty texts are identical
    if not text1 and not text2:
        return 1.0
    
    # One empty and one non-empty are completely different
    if not text1 or not text2:
        return 0.0
    
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    # Both tokenize to empty sets (e.g., only punctuation)
    if not tokens1 and not tokens2:
        return 1.0
    
    # One has tokens, one doesn't
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0.0


def calculate_compression_ratio(original: str, summary: str) -> float:
    """
    Calculate compression ratio (percentage of original text retained).
    
    Args:
        original: Original text
        summary: Summarized text
        
    Returns:
        Compression ratio as percentage (0-100)
    """
    if not original:
        return 0.0
    
    if not summary:
        return 0.0
    
    original_length = len(original)
    summary_length = len(summary)
    
    ratio = (summary_length / original_length) * 100
    return round(ratio, 1)


def count_words(text: str) -> int:
    """
    Count number of words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    if not text:
        return 0
    
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def count_sentences(text: str) -> int:
    """
    Count number of sentences in text.
    
    Simple heuristic based on sentence-ending punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Sentence count
    """
    if not text:
        return 0
    
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return len(sentences)


def calculate_average_similarity(summaries: List[str]) -> float:
    """
    Calculate average pairwise similarity among all summaries.
    
    Args:
        summaries: List of summary texts
        
    Returns:
        Average similarity score (0-100)
    """
    if len(summaries) < 2:
        return 100.0
    
    similarities = []
    n = len(summaries)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = calculate_jaccard_similarity(summaries[i], summaries[j])
            similarities.append(sim)
    
    avg_sim = np.mean(similarities) if similarities else 0.0
    return round(avg_sim * 100, 1)


def calculate_length_stats(summaries: List[str]) -> dict:
    """
    Calculate length statistics for a list of summaries.
    
    Args:
        summaries: List of summary texts
        
    Returns:
        Dictionary with length statistics
    """
    if not summaries:
        return {
            "min_length": 0,
            "max_length": 0,
            "avg_length": 0,
            "std_length": 0
        }
    
    lengths = [len(s) for s in summaries]
    
    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": round(np.mean(lengths), 1),
        "std_length": round(np.std(lengths), 2)
    }


def calculate_consistency_metrics(summaries: List[str]) -> dict:
    """
    Calculate comprehensive consistency metrics for summaries.
    
    Args:
        summaries: List of summary texts
        
    Returns:
        Dictionary of consistency metrics
    """
    if not summaries:
        return {
            "total_runs": 0,
            "unique_summaries": 0,
            "consistency_score": 0,
            "avg_similarity": 0,
            "is_deterministic": False
        }
    
    unique_count = len(set(summaries))
    total_runs = len(summaries)
    
    # Consistency score: 100% if all identical, 0% if all different
    consistency = (1 - (unique_count - 1) / max(total_runs - 1, 1)) * 100
    
    # Average similarity
    avg_sim = calculate_average_similarity(summaries)
    
    # Is deterministic?
    is_deterministic = (unique_count == 1)
    
    return {
        "total_runs": total_runs,
        "unique_summaries": unique_count,
        "consistency_score": round(consistency, 1),
        "avg_similarity": avg_sim,
        "is_deterministic": is_deterministic
    }
