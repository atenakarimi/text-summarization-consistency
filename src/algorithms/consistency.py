"""
Consistency Analysis Module

This module provides tools for measuring and analyzing the consistency
of summarization algorithms across multiple runs.
"""

from typing import List, Dict, Tuple
import numpy as np
from .extractive import get_summarizer


def run_consistency_experiment(
    text: str,
    algorithm: str,
    num_runs: int = 10,
    num_sentences: int = 3,
    language: str = "english"
) -> Dict:
    """
    Run a consistency experiment by applying the same algorithm multiple times.
    
    This function measures how consistent a summarization algorithm is by
    running it multiple times on the same input text and analyzing the
    variation in outputs.
    
    Args:
        text: Input text to summarize
        algorithm: Algorithm name ('textrank', 'lexrank', or 'luhn')
        num_runs: Number of times to run the algorithm
        num_sentences: Number of sentences to extract in each summary
        language: Language for text processing
        
    Returns:
        Dictionary containing:
            - summaries: List of all generated summaries
            - lengths: List of summary lengths (in characters)
            - avg_length: Average length across all summaries
            - length_variance: Variance in summary lengths
            - unique_summaries: Number of unique summaries
            - consistency_score: Overall consistency metric (0-100)
    """
    if not text or not text.strip():
        return {
            "summaries": [],
            "lengths": [],
            "avg_length": 0,
            "length_variance": 0,
            "unique_summaries": 0,
            "consistency_score": 0
        }
    
    # Get summarizer
    summarizer = get_summarizer(algorithm, language)
    
    # Run algorithm multiple times with different seeds
    summaries = []
    for i in range(num_runs):
        summary = summarizer.summarize(text, num_sentences, seed=i)
        summaries.append(summary)
    
    # Calculate lengths
    lengths = [len(s) for s in summaries]
    avg_length = np.mean(lengths) if lengths else 0
    length_variance = np.var(lengths) if len(lengths) > 1 else 0
    
    # Count unique summaries
    unique_summaries = len(set(summaries))
    
    # Calculate consistency score
    # 100% = all summaries identical, 0% = all summaries different
    consistency_score = (1 - (unique_summaries - 1) / max(num_runs - 1, 1)) * 100
    
    return {
        "summaries": summaries,
        "lengths": lengths,
        "avg_length": round(avg_length, 1),
        "length_variance": round(length_variance, 2),
        "unique_summaries": unique_summaries,
        "consistency_score": round(consistency_score, 1)
    }


def compare_algorithms(
    text: str,
    algorithms: List[str],
    num_runs: int = 10,
    num_sentences: int = 3,
    language: str = "english"
) -> Dict[str, Dict]:
    """
    Compare consistency across multiple algorithms.
    
    Args:
        text: Input text to summarize
        algorithms: List of algorithm names to compare
        num_runs: Number of runs per algorithm
        num_sentences: Number of sentences to extract
        language: Language for processing
        
    Returns:
        Dictionary mapping algorithm names to their experiment results
    """
    results = {}
    
    for algorithm in algorithms:
        results[algorithm] = run_consistency_experiment(
            text=text,
            algorithm=algorithm,
            num_runs=num_runs,
            num_sentences=num_sentences,
            language=language
        )
    
    return results


def calculate_pairwise_similarity(summaries: List[str]) -> List[Tuple[int, int, float]]:
    """
    Calculate pairwise similarity between all summaries.
    
    Uses Jaccard similarity coefficient based on word overlap.
    
    Args:
        summaries: List of summary texts
        
    Returns:
        List of tuples (index1, index2, similarity_score)
    """
    from utils.metrics import calculate_jaccard_similarity
    
    similarities = []
    n = len(summaries)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = calculate_jaccard_similarity(summaries[i], summaries[j])
            similarities.append((i, j, similarity))
    
    return similarities


def get_consistency_statistics(experiment_results: Dict) -> Dict:
    """
    Extract key statistics from experiment results.
    
    Args:
        experiment_results: Results from run_consistency_experiment
        
    Returns:
        Dictionary of statistics
    """
    summaries = experiment_results.get("summaries", [])
    
    if not summaries:
        return {
            "total_runs": 0,
            "unique_outputs": 0,
            "deterministic": False,
            "consistency_percentage": 0
        }
    
    unique_count = len(set(summaries))
    total_runs = len(summaries)
    is_deterministic = (unique_count == 1)
    
    return {
        "total_runs": total_runs,
        "unique_outputs": unique_count,
        "deterministic": is_deterministic,
        "consistency_percentage": experiment_results.get("consistency_score", 0)
    }
