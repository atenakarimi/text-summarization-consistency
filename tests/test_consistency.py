"""
Tests for consistency analysis functionality
"""

import pytest
from algorithms.consistency import (
    run_consistency_experiment,
    compare_algorithms,
    calculate_pairwise_similarity,
    get_consistency_statistics
)


class TestConsistencyExperiment:
    """Tests for consistency experiment runner."""
    
    def test_run_experiment_returns_dict(self, sample_text):
        """Test that experiment returns a dictionary."""
        result = run_consistency_experiment(
            text=sample_text,
            algorithm="luhn",
            num_runs=5,
            num_sentences=2
        )
        assert isinstance(result, dict)
    
    def test_experiment_result_structure(self, sample_text):
        """Test that result has expected keys."""
        result = run_consistency_experiment(
            text=sample_text,
            algorithm="luhn",
            num_runs=5,
            num_sentences=2
        )
        
        expected_keys = [
            "summaries",
            "lengths",
            "avg_length",
            "length_variance",
            "unique_summaries",
            "consistency_score"
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_experiment_correct_run_count(self, sample_text):
        """Test that correct number of summaries are generated."""
        num_runs = 7
        result = run_consistency_experiment(
            text=sample_text,
            algorithm="luhn",
            num_runs=num_runs,
            num_sentences=2
        )
        
        assert len(result["summaries"]) == num_runs
        assert len(result["lengths"]) == num_runs
    
    def test_luhn_is_fully_consistent(self, sample_text):
        """Test that Luhn algorithm produces identical results."""
        result = run_consistency_experiment(
            text=sample_text,
            algorithm="luhn",
            num_runs=10,
            num_sentences=3
        )
        
        # Luhn should be deterministic
        assert result["unique_summaries"] == 1
        assert result["consistency_score"] == 100.0
    
    def test_experiment_with_empty_text(self, empty_text):
        """Test experiment behavior with empty text."""
        result = run_consistency_experiment(
            text=empty_text,
            algorithm="luhn",
            num_runs=5,
            num_sentences=2
        )
        
        assert result["summaries"] == []
        assert result["consistency_score"] == 0
    
    def test_consistency_score_range(self, sample_text):
        """Test that consistency score is between 0 and 100."""
        result = run_consistency_experiment(
            text=sample_text,
            algorithm="textrank",
            num_runs=5,
            num_sentences=2
        )
        
        assert 0 <= result["consistency_score"] <= 100


class TestCompareAlgorithms:
    """Tests for algorithm comparison."""
    
    def test_compare_multiple_algorithms(self, sample_text):
        """Test comparing multiple algorithms."""
        algorithms = ["textrank", "lexrank", "luhn"]
        results = compare_algorithms(
            text=sample_text,
            algorithms=algorithms,
            num_runs=5,
            num_sentences=2
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert all(alg in results for alg in algorithms)
    
    def test_comparison_result_structure(self, sample_text):
        """Test that comparison results have correct structure."""
        algorithms = ["luhn"]
        results = compare_algorithms(
            text=sample_text,
            algorithms=algorithms,
            num_runs=3,
            num_sentences=2
        )
        
        assert "luhn" in results
        assert "summaries" in results["luhn"]
        assert "consistency_score" in results["luhn"]


class TestPairwiseSimilarity:
    """Tests for pairwise similarity calculation."""
    
    def test_pairwise_similarity_count(self, sample_summaries):
        """Test that correct number of pairs are calculated."""
        # 5 summaries should have 10 pairs (n*(n-1)/2)
        similarities = calculate_pairwise_similarity(sample_summaries)
        assert len(similarities) == 10
    
    def test_pairwise_similarity_structure(self, sample_summaries):
        """Test that similarity tuples have correct structure."""
        similarities = calculate_pairwise_similarity(sample_summaries)
        
        for sim in similarities:
            assert len(sim) == 3  # (index1, index2, similarity)
            assert isinstance(sim[0], int)
            assert isinstance(sim[1], int)
            assert isinstance(sim[2], float)
            assert 0 <= sim[2] <= 1
    
    def test_empty_summaries_list(self):
        """Test behavior with empty summaries list."""
        similarities = calculate_pairwise_similarity([])
        assert similarities == []


class TestConsistencyStatistics:
    """Tests for consistency statistics extraction."""
    
    def test_statistics_structure(self):
        """Test that statistics have expected keys."""
        experiment_result = {
            "summaries": ["summary1", "summary2", "summary1"],
            "consistency_score": 75.0
        }
        
        stats = get_consistency_statistics(experiment_result)
        
        assert "total_runs" in stats
        assert "unique_outputs" in stats
        assert "deterministic" in stats
        assert "consistency_percentage" in stats
    
    def test_deterministic_detection(self):
        """Test detection of deterministic behavior."""
        # All identical summaries
        experiment_result = {
            "summaries": ["same"] * 5,
            "consistency_score": 100.0
        }
        
        stats = get_consistency_statistics(experiment_result)
        assert stats["deterministic"] is True
        assert stats["unique_outputs"] == 1
    
    def test_non_deterministic_detection(self):
        """Test detection of non-deterministic behavior."""
        # All different summaries
        experiment_result = {
            "summaries": ["s1", "s2", "s3", "s4", "s5"],
            "consistency_score": 0.0
        }
        
        stats = get_consistency_statistics(experiment_result)
        assert stats["deterministic"] is False
        assert stats["unique_outputs"] == 5
    
    def test_empty_results(self):
        """Test behavior with empty experiment results."""
        experiment_result = {
            "summaries": [],
            "consistency_score": 0
        }
        
        stats = get_consistency_statistics(experiment_result)
        assert stats["total_runs"] == 0
        assert stats["deterministic"] is False
