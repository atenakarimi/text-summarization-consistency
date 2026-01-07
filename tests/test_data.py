"""
Tests for data loading and management functions
"""

import pytest
import pandas as pd
from pathlib import Path
from utils.data import (
    load_sample_articles,
    get_article_by_title,
    get_articles_by_category,
    get_available_categories,
    get_all_titles,
    clean_text,
    validate_text,
    get_text_statistics
)


class TestLoadSampleArticles:
    """Tests for loading sample articles."""
    
    def test_load_returns_dataframe(self):
        """Test that load_sample_articles returns a DataFrame."""
        df = load_sample_articles()
        assert isinstance(df, pd.DataFrame)
    
    def test_dataframe_has_expected_columns(self):
        """Test that DataFrame has expected columns."""
        df = load_sample_articles()
        expected_columns = ["title", "category", "content"]
        
        if not df.empty:
            for col in expected_columns:
                assert col in df.columns
    
    def test_articles_have_content(self):
        """Test that articles have non-empty content."""
        df = load_sample_articles()
        
        if not df.empty:
            assert all(df["content"].str.len() > 0)


class TestGetArticleByTitle:
    """Tests for retrieving articles by title."""
    
    def test_get_existing_article(self):
        """Test retrieving an existing article."""
        df = load_sample_articles()
        
        if not df.empty:
            first_title = df.iloc[0]["title"]
            article = get_article_by_title(first_title)
            
            assert article is not None
            assert "title" in article
            assert "category" in article
            assert "content" in article
    
    def test_get_nonexistent_article(self):
        """Test retrieving a non-existent article."""
        article = get_article_by_title("This Title Does Not Exist XYZ123")
        assert article is None


class TestGetArticlesByCategory:
    """Tests for retrieving articles by category."""
    
    def test_get_all_articles(self):
        """Test getting all articles."""
        articles = get_articles_by_category("all")
        assert isinstance(articles, list)
    
    def test_get_specific_category(self):
        """Test getting articles from specific category."""
        df = load_sample_articles()
        
        if not df.empty and "category" in df.columns:
            categories = df["category"].unique()
            if len(categories) > 0:
                category = categories[0]
                articles = get_articles_by_category(category)
                
                assert isinstance(articles, list)
                if articles:
                    assert all(a["category"] == category for a in articles)


class TestGetAvailableCategories:
    """Tests for getting available categories."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        categories = get_available_categories()
        assert isinstance(categories, list)
    
    def test_categories_are_sorted(self):
        """Test that categories are sorted."""
        categories = get_available_categories()
        if len(categories) > 1:
            assert categories == sorted(categories)


class TestGetAllTitles:
    """Tests for getting all article titles."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        titles = get_all_titles()
        assert isinstance(titles, list)
    
    def test_titles_are_sorted(self):
        """Test that titles are sorted."""
        titles = get_all_titles()
        if len(titles) > 1:
            assert titles == sorted(titles)


class TestCleanText:
    """Tests for text cleaning."""
    
    def test_removes_extra_whitespace(self):
        """Test removal of extra whitespace."""
        text = "This    has   extra    spaces"
        cleaned = clean_text(text)
        assert cleaned == "This has extra spaces"
    
    def test_removes_leading_trailing_whitespace(self):
        """Test removal of leading/trailing whitespace."""
        text = "   text with spaces   "
        cleaned = clean_text(text)
        assert cleaned == "text with spaces"
    
    def test_handles_empty_text(self):
        """Test handling of empty text."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_handles_newlines(self):
        """Test handling of newlines."""
        text = "Line1\n\nLine2\nLine3"
        cleaned = clean_text(text)
        assert "\n\n" not in cleaned


class TestValidateText:
    """Tests for text validation."""
    
    def test_valid_text(self):
        """Test that valid text passes validation."""
        text = "This is a sufficiently long text for summarization purposes."
        assert validate_text(text) is True
    
    def test_too_short_text(self):
        """Test that short text fails validation."""
        text = "Short"
        assert validate_text(text, min_length=50) is False
    
    def test_empty_text(self):
        """Test that empty text fails validation."""
        assert validate_text("") is False
        assert validate_text(None) is False
    
    def test_custom_min_length(self):
        """Test validation with custom minimum length."""
        text = "This is twenty chars"  # 20 characters
        assert validate_text(text, min_length=10) is True
        assert validate_text(text, min_length=30) is False


class TestGetTextStatistics:
    """Tests for text statistics calculation."""
    
    def test_statistics_structure(self):
        """Test that statistics have correct structure."""
        text = "This is a test. It has two sentences."
        stats = get_text_statistics(text)
        
        assert "char_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
    
    def test_character_count(self):
        """Test character counting."""
        text = "12345"
        stats = get_text_statistics(text)
        assert stats["char_count"] == 5
    
    def test_word_count(self):
        """Test word counting."""
        text = "One two three four"
        stats = get_text_statistics(text)
        assert stats["word_count"] == 4
    
    def test_sentence_count(self):
        """Test sentence counting."""
        text = "First. Second! Third?"
        stats = get_text_statistics(text)
        assert stats["sentence_count"] == 3
    
    def test_empty_text_statistics(self):
        """Test statistics for empty text."""
        stats = get_text_statistics("")
        assert stats["char_count"] == 0
        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0
