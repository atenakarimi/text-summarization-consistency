"""
Tests for metrics and evaluation functions
"""

import pytest
from utils.metrics import (
    calculate_jaccard_similarity,
    calculate_compression_ratio,
    count_words,
    count_sentences,
    calculate_average_similarity,
    calculate_length_stats,
    calculate_consistency_metrics
)


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""
    
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        text = "This is a sample text"
        similarity = calculate_jaccard_similarity(text, text)
        assert similarity == 1.0
    
    def test_completely_different_texts(self):
        """Test similarity of completely different texts."""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"
        similarity = calculate_jaccard_similarity(text1, text2)
        assert similarity == 0.0
    
    def test_partial_overlap(self):
        """Test similarity with partial overlap."""
        text1 = "the quick brown fox"
        text2 = "the lazy brown dog"
        similarity = calculate_jaccard_similarity(text1, text2)
        assert 0 < similarity < 1
        # "the" and "brown" are common = 2/6 unique words
        assert similarity == pytest.approx(0.333, abs=0.01)
    
    def test_empty_texts(self):
        """Test behavior with empty texts."""
        assert calculate_jaccard_similarity("", "") == 1.0
        assert calculate_jaccard_similarity("text", "") == 0.0
        assert calculate_jaccard_similarity("", "text") == 0.0
    
    def test_case_insensitive(self):
        """Test that similarity is case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"
        similarity = calculate_jaccard_similarity(text1, text2)
        assert similarity == 1.0


class TestCompressionRatio:
    """Tests for compression ratio calculation."""
    
    def test_50_percent_compression(self):
        """Test 50% compression calculation."""
        original = "a" * 100
        summary = "a" * 50
        ratio = calculate_compression_ratio(original, summary)
        assert ratio == 50.0
    
    def test_no_compression(self):
        """Test when summary equals original."""
        text = "This is the text"
        ratio = calculate_compression_ratio(text, text)
        assert ratio == 100.0
    
    def test_empty_summary(self):
        """Test with empty summary."""
        original = "Original text here"
        ratio = calculate_compression_ratio(original, "")
        assert ratio == 0.0
    
    def test_empty_original(self):
        """Test with empty original."""
        ratio = calculate_compression_ratio("", "summary")
        assert ratio == 0.0


class TestWordCount:
    """Tests for word counting."""
    
    def test_simple_sentence(self):
        """Test counting words in simple sentence."""
        text = "This is a test"
        assert count_words(text) == 4
    
    def test_punctuation_handling(self):
        """Test that punctuation doesn't affect count."""
        text = "Hello, world! How are you?"
        assert count_words(text) == 5
    
    def test_empty_text(self):
        """Test counting words in empty text."""
        assert count_words("") == 0
    
    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        text = "word1    word2     word3"
        assert count_words(text) == 3


class TestSentenceCount:
    """Tests for sentence counting."""
    
    def test_single_sentence(self):
        """Test counting single sentence."""
        text = "This is one sentence."
        assert count_sentences(text) == 1
    
    def test_multiple_sentences(self):
        """Test counting multiple sentences."""
        text = "First sentence. Second sentence! Third sentence?"
        assert count_sentences(text) == 3
    
    def test_empty_text(self):
        """Test counting sentences in empty text."""
        assert count_sentences("") == 0
    
    def test_no_punctuation(self):
        """Test text without sentence-ending punctuation."""
        text = "This is a sentence without ending punctuation"
        assert count_sentences(text) == 1


class TestAverageSimilarity:
    """Tests for average similarity calculation."""
    
    def test_identical_summaries(self):
        """Test average similarity of identical summaries."""
        summaries = ["same text"] * 5
        avg_sim = calculate_average_similarity(summaries)
        assert avg_sim == 100.0
    
    def test_single_summary(self):
        """Test behavior with single summary."""
        summaries = ["only one"]
        avg_sim = calculate_average_similarity(summaries)
        assert avg_sim == 100.0
    
    def test_varied_summaries(self):
        """Test average similarity with varied summaries."""
        summaries = [
            "the quick brown fox",
            "the lazy brown dog",
            "a fast red cat"
        ]
        avg_sim = calculate_average_similarity(summaries)
        assert 0 <= avg_sim <= 100


class TestLengthStats:
    """Tests for length statistics calculation."""
    
    def test_length_stats_structure(self, sample_summaries):
        """Test that length stats have correct structure."""
        stats = calculate_length_stats(sample_summaries)
        
        assert "min_length" in stats
        assert "max_length" in stats
        assert "avg_length" in stats
        assert "std_length" in stats
    
    def test_empty_summaries(self):
        """Test behavior with empty summaries list."""
        stats = calculate_length_stats([])
        
        assert stats["min_length"] == 0
        assert stats["max_length"] == 0
        assert stats["avg_length"] == 0
    
    def test_single_summary(self):
        """Test stats with single summary."""
        summaries = ["This is a test summary"]
        stats = calculate_length_stats(summaries)
        
        length = len(summaries[0])
        assert stats["min_length"] == length
        assert stats["max_length"] == length
        assert stats["avg_length"] == length
        assert stats["std_length"] == 0.0


class TestConsistencyMetrics:
    """Tests for comprehensive consistency metrics."""
    
    def test_fully_consistent(self):
        """Test metrics for fully consistent summaries."""
        summaries = ["identical"] * 10
        metrics = calculate_consistency_metrics(summaries)
        
        assert metrics["total_runs"] == 10
        assert metrics["unique_summaries"] == 1
        assert metrics["consistency_score"] == 100.0
        assert metrics["is_deterministic"] is True
    
    def test_no_consistency(self):
        """Test metrics for completely inconsistent summaries."""
        summaries = [f"summary_{i}" for i in range(10)]
        metrics = calculate_consistency_metrics(summaries)
        
        assert metrics["total_runs"] == 10
        assert metrics["unique_summaries"] == 10
        assert metrics["consistency_score"] == 0.0
        assert metrics["is_deterministic"] is False
    
    def test_partial_consistency(self, sample_summaries):
        """Test metrics with partial consistency."""
        metrics = calculate_consistency_metrics(sample_summaries)
        
        assert metrics["total_runs"] == len(sample_summaries)
        assert 0 < metrics["consistency_score"] < 100
        assert metrics["is_deterministic"] is False
    
    def test_empty_summaries(self):
        """Test behavior with empty summaries."""
        metrics = calculate_consistency_metrics([])
        
        assert metrics["total_runs"] == 0
        assert metrics["unique_summaries"] == 0
        assert metrics["consistency_score"] == 0
        assert metrics["is_deterministic"] is False
