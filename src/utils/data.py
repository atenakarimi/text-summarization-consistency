"""
Data Loading and Management Module

Handles loading sample articles and text preprocessing.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import re


def load_sample_articles(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample articles from CSV file.
    
    Args:
        data_path: Path to CSV file. If None, uses default data/sample_articles.csv
        
    Returns:
        DataFrame with columns: title, category, content
    """
    if data_path is None:
        # Get path relative to this file
        current_dir = Path(__file__).parent.parent.parent
        data_path = current_dir / "data" / "sample_articles.csv"
    
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["title", "category", "content"])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=["title", "category", "content"])


def get_article_by_title(title: str, data_path: Optional[str] = None) -> Optional[Dict]:
    """
    Get article content by title.
    
    Args:
        title: Article title
        data_path: Path to CSV file
        
    Returns:
        Dictionary with article data or None if not found
    """
    df = load_sample_articles(data_path)
    
    if df.empty:
        return None
    
    matches = df[df['title'] == title]
    
    if matches.empty:
        return None
    
    row = matches.iloc[0]
    return {
        "title": row['title'],
        "category": row['category'],
        "content": row['content']
    }


def get_articles_by_category(category: str, data_path: Optional[str] = None) -> List[Dict]:
    """
    Get all articles in a specific category.
    
    Args:
        category: Category name
        data_path: Path to CSV file
        
    Returns:
        List of article dictionaries
    """
    df = load_sample_articles(data_path)
    
    if df.empty:
        return []
    
    if category.lower() == "all":
        filtered = df
    else:
        filtered = df[df['category'].str.lower() == category.lower()]
    
    articles = []
    for _, row in filtered.iterrows():
        articles.append({
            "title": row['title'],
            "category": row['category'],
            "content": row['content']
        })
    
    return articles


def get_available_categories(data_path: Optional[str] = None) -> List[str]:
    """
    Get list of available article categories.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        List of category names (sorted)
    """
    df = load_sample_articles(data_path)
    
    if df.empty or 'category' not in df.columns:
        return []
    
    categories = df['category'].unique().tolist()
    return sorted(categories)


def get_all_titles(data_path: Optional[str] = None) -> List[str]:
    """
    Get list of all article titles.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        List of titles (sorted)
    """
    df = load_sample_articles(data_path)
    
    if df.empty or 'title' not in df.columns:
        return []
    
    return sorted(df['title'].tolist())


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_text(text: str, min_length: int = 50) -> bool:
    """
    Validate that text is suitable for summarization.
    
    Args:
        text: Input text
        min_length: Minimum required length
        
    Returns:
        True if text is valid, False otherwise
    """
    if not text or not text.strip():
        return False
    
    if len(text.strip()) < min_length:
        return False
    
    return True


def get_text_statistics(text: str) -> Dict:
    """
    Get basic statistics about a text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0
        }
    
    # Character count
    char_count = len(text)
    
    # Word count
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count
    }
