"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial intelligence is transforming industries worldwide. Machine learning 
    algorithms can now process vast amounts of data efficiently. Deep learning models 
    have achieved remarkable results in image recognition. Natural language processing 
    enables computers to understand human language. The future of AI looks promising 
    with continued research and development. Ethical considerations remain important 
    as technology advances.
    """


@pytest.fixture
def short_text():
    """Short text that may not be suitable for summarization."""
    return "This is a very short text."


@pytest.fixture
def empty_text():
    """Empty text for edge case testing."""
    return ""


@pytest.fixture
def sample_summaries():
    """Sample list of summaries for consistency testing."""
    return [
        "This is the first summary. It contains some text.",
        "This is the first summary. It contains some text.",
        "This is a different summary. It has other content.",
        "This is the first summary. It contains some text.",
        "This is another unique summary entirely."
    ]
