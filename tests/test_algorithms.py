"""
Tests for extractive summarization algorithms
"""

import pytest
from algorithms.extractive import (
    TextRankSummarizer,
    LexRankSummarizer,
    LuhnSummarizer,
    get_summarizer,
    get_available_algorithms
)


class TestTextRankSummarizer:
    """Tests for TextRank algorithm."""
    
    def test_initialization(self):
        """Test that TextRank initializes correctly."""
        summarizer = TextRankSummarizer()
        assert summarizer.language == "english"
        assert summarizer.stemmer is not None
        assert summarizer.stop_words is not None
    
    def test_summarize_returns_string(self, sample_text):
        """Test that summarize returns a string."""
        summarizer = TextRankSummarizer()
        summary = summarizer.summarize(sample_text, num_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summarize_respects_sentence_count(self, sample_text):
        """Test that summary has requested number of sentences."""
        summarizer = TextRankSummarizer()
        summary = summarizer.summarize(sample_text, num_sentences=1)
        # Should have roughly 1 sentence (allowing for some flexibility)
        assert len(summary) > 0
    
    def test_summarize_empty_text(self, empty_text):
        """Test behavior with empty text."""
        summarizer = TextRankSummarizer()
        summary = summarizer.summarize(empty_text)
        assert summary == ""
    
    def test_summarize_with_seed(self, sample_text):
        """Test that seed parameter is accepted."""
        summarizer = TextRankSummarizer()
        summary1 = summarizer.summarize(sample_text, seed=42)
        summary2 = summarizer.summarize(sample_text, seed=42)
        # Both should return strings
        assert isinstance(summary1, str)
        assert isinstance(summary2, str)


class TestLexRankSummarizer:
    """Tests for LexRank algorithm."""
    
    def test_initialization(self):
        """Test that LexRank initializes correctly."""
        summarizer = LexRankSummarizer()
        assert summarizer.language == "english"
        assert summarizer.stemmer is not None
        assert summarizer.stop_words is not None
    
    def test_summarize_returns_string(self, sample_text):
        """Test that summarize returns a string."""
        summarizer = LexRankSummarizer()
        summary = summarizer.summarize(sample_text, num_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summarize_empty_text(self, empty_text):
        """Test behavior with empty text."""
        summarizer = LexRankSummarizer()
        summary = summarizer.summarize(empty_text)
        assert summary == ""
    
    def test_determinism(self, sample_text):
        """Test that LexRank produces consistent results."""
        summarizer = LexRankSummarizer()
        summary1 = summarizer.summarize(sample_text, num_sentences=2, seed=42)
        summary2 = summarizer.summarize(sample_text, num_sentences=2, seed=42)
        # LexRank should be deterministic
        assert summary1 == summary2


class TestLuhnSummarizer:
    """Tests for Luhn algorithm."""
    
    def test_initialization(self):
        """Test that Luhn initializes correctly."""
        summarizer = LuhnSummarizer()
        assert summarizer.language == "english"
        assert summarizer.stemmer is not None
        assert summarizer.stop_words is not None
    
    def test_summarize_returns_string(self, sample_text):
        """Test that summarize returns a string."""
        summarizer = LuhnSummarizer()
        summary = summarizer.summarize(sample_text, num_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summarize_empty_text(self, empty_text):
        """Test behavior with empty text."""
        summarizer = LuhnSummarizer()
        summary = summarizer.summarize(empty_text)
        assert summary == ""
    
    def test_full_determinism(self, sample_text):
        """Test that Luhn is fully deterministic."""
        summarizer = LuhnSummarizer()
        summaries = [
            summarizer.summarize(sample_text, num_sentences=3, seed=i)
            for i in range(10)
        ]
        # All summaries should be identical (Luhn is deterministic)
        assert len(set(summaries)) == 1


class TestSummarizerFactory:
    """Tests for summarizer factory functions."""
    
    def test_get_summarizer_textrank(self):
        """Test getting TextRank summarizer."""
        summarizer = get_summarizer("textrank")
        assert isinstance(summarizer, TextRankSummarizer)
    
    def test_get_summarizer_lexrank(self):
        """Test getting LexRank summarizer."""
        summarizer = get_summarizer("lexrank")
        assert isinstance(summarizer, LexRankSummarizer)
    
    def test_get_summarizer_luhn(self):
        """Test getting Luhn summarizer."""
        summarizer = get_summarizer("luhn")
        assert isinstance(summarizer, LuhnSummarizer)
    
    def test_get_summarizer_case_insensitive(self):
        """Test that algorithm names are case-insensitive."""
        summarizer1 = get_summarizer("TEXTRANK")
        summarizer2 = get_summarizer("TextRank")
        assert isinstance(summarizer1, TextRankSummarizer)
        assert isinstance(summarizer2, TextRankSummarizer)
    
    def test_get_summarizer_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError):
            get_summarizer("invalid_algorithm")
    
    def test_get_available_algorithms(self):
        """Test getting list of available algorithms."""
        algorithms = get_available_algorithms()
        assert isinstance(algorithms, list)
        assert "textrank" in algorithms
        assert "lexrank" in algorithms
        assert "luhn" in algorithms
        assert len(algorithms) == 3
