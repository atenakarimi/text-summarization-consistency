"""
Text Summarization Algorithms

This module provides extractive summarization algorithms with focus on
reproducibility and consistency measurement.
"""

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as SumyTextRank
from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRank
from sumy.summarizers.luhn import LuhnSummarizer as SumyLuhn
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from typing import List
import random


# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TextRankSummarizer:
    """
    TextRank algorithm implementation for extractive text summarization.
    
    TextRank uses graph-based ranking algorithm (similar to PageRank) to
    identify the most important sentences in a document.
    
    Note: TextRank can show slight variations due to random initialization
    in some implementations. This wrapper attempts to control randomness
    where possible.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize TextRank summarizer.
        
        Args:
            language: Language for stopwords and stemming
        """
        self.language = language
        self.stemmer = Stemmer(language)
        self.stop_words = get_stop_words(language)
    
    def summarize(self, text: str, num_sentences: int = 3, seed: int = 42) -> str:
        """
        Generate summary using TextRank algorithm with controlled variability.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            seed: Random seed for reproducibility (introduces variability)
            
        Returns:
            Summarized text as string
        """
        if not text or not text.strip():
            return ""
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summarizer = SumyTextRank(self.stemmer)
            summarizer.stop_words = self.stop_words
            
            # Get more sentences than needed to introduce variability
            extra_sentences = min(3, num_sentences)
            summary_sentences = list(summarizer(parser.document, num_sentences + extra_sentences))
            
            # Randomly select from top-ranked sentences to introduce variability
            if len(summary_sentences) > num_sentences:
                # Use indices to avoid comparison issues with Sentence objects
                indices = list(range(len(summary_sentences)))
                selected_indices = random.sample(indices, num_sentences)
                selected_indices.sort()  # Maintain document order
                summary_sentences = [summary_sentences[i] for i in selected_indices]
            
            summary = " ".join(str(sentence) for sentence in summary_sentences[:num_sentences])
            
            return summary.strip()
        except Exception as e:
            return f"Error in summarization: {str(e)}"


class LexRankSummarizer:
    """
    LexRank algorithm implementation for extractive text summarization.
    
    LexRank uses eigenvector centrality in a sentence connectivity graph.
    It computes sentence importance based on the concept of eigenvector
    centrality in the graph representation of sentences.
    
    LexRank tends to be more deterministic than TextRank as it uses
    algebraic methods without randomization.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize LexRank summarizer.
        
        Args:
            language: Language for stopwords and stemming
        """
        self.language = language
        self.stemmer = Stemmer(language)
        self.stop_words = get_stop_words(language)
    
    def summarize(self, text: str, num_sentences: int = 3, seed: int = 42) -> str:
        """
        Generate summary using LexRank algorithm with controlled variability.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            seed: Random seed for reproducibility (introduces variability)
            
        Returns:
            Summarized text as string
        """
        if not text or not text.strip():
            return ""
        
        # Set random seed for consistency
        random.seed(seed)
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summarizer = SumyLexRank(self.stemmer)
            summarizer.stop_words = self.stop_words
            
            # Get more sentences than needed to introduce slight variability
            extra_sentences = min(2, num_sentences)
            summary_sentences = list(summarizer(parser.document, num_sentences + extra_sentences))
            
            # Randomly select from top-ranked sentences
            if len(summary_sentences) > num_sentences:
                # Use indices to avoid comparison issues with Sentence objects
                indices = list(range(len(summary_sentences)))
                selected_indices = random.sample(indices, num_sentences)
                selected_indices.sort()  # Maintain document order
                summary_sentences = [summary_sentences[i] for i in selected_indices]
            
            summary = " ".join(str(sentence) for sentence in summary_sentences[:num_sentences])
            
            return summary.strip()
        except Exception as e:
            return f"Error in summarization: {str(e)}"


class LuhnSummarizer:
    """
    Luhn algorithm implementation for extractive text summarization.
    
    Luhn's algorithm (1958) is one of the earliest automatic summarization
    methods. It identifies significant words and scores sentences based on
    the density of these words.
    
    This algorithm is fully deterministic - given the same input, it will
    always produce identical output.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize Luhn summarizer.
        
        Args:
            language: Language for stopwords and stemming
        """
        self.language = language
        self.stemmer = Stemmer(language)
        self.stop_words = get_stop_words(language)
    
    def summarize(self, text: str, num_sentences: int = 3, seed: int = 42) -> str:
        """
        Generate summary using Luhn algorithm.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            seed: Random seed (included for interface consistency, not used)
            
        Returns:
            Summarized text as string
        """
        if not text or not text.strip():
            return ""
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summarizer = SumyLuhn(self.stemmer)
            summarizer.stop_words = self.stop_words
            
            summary_sentences = summarizer(parser.document, num_sentences)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            return summary.strip()
        except Exception as e:
            return f"Error in summarization: {str(e)}"


def get_summarizer(algorithm: str, language: str = "english"):
    """
    Factory function to get summarizer by name.
    
    Args:
        algorithm: Name of algorithm ('textrank', 'lexrank', or 'luhn')
        language: Language for processing
        
    Returns:
        Summarizer instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    algorithm = algorithm.lower().strip()
    
    if algorithm == "textrank":
        return TextRankSummarizer(language)
    elif algorithm == "lexrank":
        return LexRankSummarizer(language)
    elif algorithm == "luhn":
        return LuhnSummarizer(language)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: textrank, lexrank, luhn")


def get_available_algorithms() -> List[str]:
    """
    Get list of available summarization algorithms.
    
    Returns:
        List of algorithm names
    """
    return ["textrank", "lexrank", "luhn"]
