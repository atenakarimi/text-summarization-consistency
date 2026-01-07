# Architecture Documentation

## System Overview

The Text Summarization Consistency Analyzer is designed with a modular architecture that separates concerns between data processing, algorithmic execution, metric calculation, and user interface presentation.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                      (Streamlit App)                        │
└────────────────┬────────────────────────────┬───────────────┘
                 │                            │
                 ▼                            ▼
┌────────────────────────────┐   ┌──────────────────────────┐
│    Algorithm Layer         │   │     Utility Layer        │
│  - TextRank Summarizer     │   │  - Data Loader           │
│  - LexRank Summarizer      │   │  - Text Processor        │
│  - Luhn Summarizer         │   │  - Metrics Calculator    │
│  - Consistency Analyzer    │   │  - Validation            │
└────────────────┬───────────┘   └──────────┬───────────────┘
                 │                          │
                 └──────────┬───────────────┘
                            ▼
                ┌──────────────────────┐
                │     Data Layer       │
                │  - CSV Storage       │
                │  - Article Database  │
                └──────────────────────┘
```

## Component Architecture

### 1. Algorithm Layer (`src/algorithms/`)

#### `extractive.py` - Summarization Algorithms

**Purpose**: Implements three extractive summarization algorithms with reproducibility focus.

**Classes**:

```python
class TextRankSummarizer:
    """Graph-based summarization using PageRank algorithm"""
    
    def __init__(self):
        # Initializes sentence tokenizer
    
    def summarize(self, text: str, num_sentences: int, seed: int = 42) -> str:
        """
        Generates summary using TextRank algorithm.
        
        Args:
            text: Input document
            num_sentences: Target summary length
            seed: Random seed for reproducibility
        
        Returns:
            Summary text with extracted sentences
        """
```

```python
class LexRankSummarizer:
    """Graph-based summarization using eigenvector centrality"""
    
    def summarize(self, text: str, num_sentences: int, seed: int = 42) -> str:
        """Uses IDF-modified cosine similarity for sentence ranking"""
```

```python
class LuhnSummarizer:
    """Frequency-based summarization (deterministic)"""
    
    def summarize(self, text: str, num_sentences: int, seed: int = 42) -> str:
        """Scores sentences by significant word clusters (seed ignored)"""
```

**Factory Pattern**:
```python
def get_summarizer(algorithm: str) -> BaseSummarizer:
    """Returns appropriate summarizer instance based on algorithm name"""
```

**Design Decisions**:
- All algorithms accept `seed` parameter for API consistency
- Luhn ignores seed (fully deterministic)
- TextRank/LexRank use seed for random initialization
- Clean interface: single `summarize()` method

#### `consistency.py` - Experiment Runner

**Purpose**: Orchestrates consistency experiments and multi-run analysis.

**Key Functions**:

```python
def run_consistency_experiment(
    text: str,
    algorithm: str,
    num_sentences: int,
    num_runs: int,
    base_seed: int = 42
) -> Dict[str, Any]:
    """
    Runs summarization algorithm multiple times and measures consistency.
    
    Process:
    1. Create summarizer instance
    2. Run num_runs times with different seeds
    3. Calculate consistency metrics
    4. Analyze length variance
    5. Return comprehensive results
    
    Returns:
        {
            'summaries': List[str],           # All generated summaries
            'summary_lengths': List[int],     # Length of each summary
            'avg_length': float,              # Mean summary length
            'length_variance': float,         # Std dev of lengths
            'unique_summaries': int,          # Count of distinct outputs
            'consistency_score': float,       # % identical summaries
            'similarity_matrix': np.ndarray,  # Pairwise similarities
            'avg_similarity': float           # Mean similarity
        }
    """
```

```python
def compare_algorithms(
    text: str,
    algorithms: List[str],
    num_sentences: int,
    num_runs: int
) -> Dict[str, Dict]:
    """
    Compares consistency across multiple algorithms.
    
    Returns:
        Dictionary mapping algorithm names to experiment results
    """
```

### 2. Utility Layer (`src/utils/`)

#### `metrics.py` - Evaluation Metrics

**Purpose**: Calculates similarity and consistency metrics for analysis.

**Functions**:

```python
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Word-based Jaccard similarity coefficient.
    
    Formula: |A ∩ B| / |A ∪ B|
    
    Returns:
        Float in [0, 1] where 1 = identical texts
    """
```

```python
def calculate_consistency_score(summaries: List[str]) -> float:
    """
    Percentage of summaries identical to most common output.
    
    Process:
    1. Count frequency of each unique summary
    2. Find most common summary
    3. Return: (count / total) * 100
    """
```

```python
def calculate_pairwise_similarities(summaries: List[str]) -> np.ndarray:
    """
    Computes similarity matrix for all summary pairs.
    
    Returns:
        NxN matrix where M[i,j] = similarity(summary_i, summary_j)
    """
```

```python
def calculate_average_similarity(summaries: List[str]) -> float:
    """Mean of all pairwise similarities (excluding diagonal)"""
```

```python
def calculate_length_statistics(summaries: List[str]) -> Dict[str, float]:
    """
    Returns:
        {
            'min': Shortest summary length,
            'max': Longest summary length,
            'mean': Average length,
            'std': Standard deviation
        }
    """
```

**Design Decisions**:
- Jaccard chosen for simplicity and interpretability
- Consistency score penalizes any variation
- Matrix calculations use NumPy for efficiency
- All functions handle edge cases (empty inputs, single summary)

#### `data.py` - Data Management

**Purpose**: Handles article loading, text processing, and validation.

**Functions**:

```python
def load_sample_articles() -> pd.DataFrame:
    """
    Loads articles from data/sample_articles.csv.
    
    Expected columns:
    - title: Article title
    - category: Technology/Science/Business
    - text: Full article text
    
    Returns:
        DataFrame with all articles
    """
```

```python
def get_article_by_title(title: str, df: pd.DataFrame) -> Optional[str]:
    """Retrieves article text by title (case-insensitive)"""
```

```python
def get_articles_by_category(category: str, df: pd.DataFrame) -> pd.DataFrame:
    """Filters articles by category"""
```

```python
def clean_text(text: str) -> str:
    """
    Normalizes text for processing:
    1. Remove extra whitespace
    2. Normalize newlines
    3. Strip leading/trailing spaces
    """
```

```python
def validate_text(text: str, min_sentences: int = 3) -> bool:
    """
    Validates text is suitable for summarization:
    - Not empty
    - Contains minimum sentences
    - Has reasonable length
    """
```

```python
def get_text_statistics(text: str) -> Dict[str, int]:
    """
    Returns:
        {
            'characters': Character count,
            'words': Word count,
            'sentences': Sentence count,
            'avg_sentence_length': Words per sentence
        }
    """
```

### 3. User Interface Layer (`src/app.py`)

**Purpose**: Streamlit web application providing interactive interface.

**Architecture**:

```
app.py
├── Configuration (page settings, theme)
├── Data Loading (load_sample_articles)
├── Sidebar
│   ├── Article Selection (dropdown or custom input)
│   ├── Algorithm Selection (radio buttons)
│   └── Experiment Configuration (sliders, number inputs)
├── Main Area
│   ├── Header & Instructions
│   ├── Article Information (title, category, stats)
│   ├── Experiment Trigger (button)
│   ├── Results Display
│   │   ├── Metric Cards (4-column layout)
│   │   ├── Visualizations (3 tabs)
│   │   │   ├── Consistency Gauge
│   │   │   ├── Similarity Heatmap
│   │   │   └── Length Distribution
│   │   ├── Summary Table
│   │   └── Export Button
│   └── Footer (author info, GitHub link)
└── Session State Management
```

**Key UI Components**:

1. **Metric Cards**: Display key metrics in colored boxes
   - Consistency Score (green/yellow/red based on value)
   - Average Similarity (0-1 scale)
   - Unique Summaries (integer count)
   - Length Variance (std dev)

2. **Consistency Gauge**: Plotly indicator chart
   - Green zone (80-100%): High consistency
   - Yellow zone (50-80%): Moderate consistency
   - Red zone (0-50%): Low consistency

3. **Similarity Heatmap**: Plotly heatmap
   - Interactive hover showing exact values
   - Color scale from white (0) to dark blue (1)
   - Diagonal always 1 (self-similarity)

4. **Length Distribution**: Plotly histogram
   - X-axis: Summary length (words)
   - Y-axis: Frequency
   - Shows spread of output sizes

**Design Principles**:
- Clean white theme (no dark mode)
- Professional color scheme (blue primary, green success, red warning)
- Responsive layout (works on mobile)
- Clear visual hierarchy
- Helpful tooltips and explanations

### 4. Testing Layer (`tests/`)

**Purpose**: Comprehensive test suite ensuring correctness and reliability.

**Test Structure**:

```python
# tests/conftest.py - Shared fixtures
@pytest.fixture
def sample_text():
    """Returns consistent text for testing"""

@pytest.fixture
def sample_articles():
    """Returns mock DataFrame with articles"""
```

```python
# tests/test_algorithms.py
def test_textrank_initialization()
def test_textrank_summarization()
def test_textrank_seed_consistency()
def test_lexrank_summarization()
def test_luhn_determinism()  # Critical: Luhn must be 100% consistent
def test_summarizer_factory()
def test_invalid_algorithm()
```

```python
# tests/test_consistency.py
def test_run_consistency_experiment()
def test_experiment_run_count()
def test_consistency_score_perfect()  # All summaries identical
def test_consistency_score_varied()   # Different summaries
def test_compare_algorithms()
def test_similarity_matrix_shape()
```

```python
# tests/test_metrics.py
def test_jaccard_similarity_identical()
def test_jaccard_similarity_different()
def test_jaccard_similarity_partial_overlap()
def test_consistency_score_calculation()
def test_pairwise_similarities()
def test_length_statistics()
```

```python
# tests/test_data.py
def test_load_sample_articles()
def test_get_article_by_title()
def test_get_articles_by_category()
def test_clean_text()
def test_validate_text()
def test_text_statistics()
```

**Test Coverage Goals**:
- Algorithm correctness: >90%
- Consistency metrics: 100%
- Data loading: >95%
- Edge cases: All handled

## Data Flow

### Typical User Session

1. **Application Launch**:
   ```
   User → Docker/Streamlit → Load app.py → Initialize components
   ```

2. **Article Selection**:
   ```
   User clicks dropdown → data.py:load_sample_articles()
   → Display article info → Show text statistics
   ```

3. **Experiment Configuration**:
   ```
   User selects algorithm → User adjusts parameters
   → Validate input ranges → Enable "Run" button
   ```

4. **Experiment Execution**:
   ```
   User clicks "Run Experiment"
   ↓
   app.py:run_consistency_experiment()
   ↓
   consistency.py:run_consistency_experiment()
   ↓
   Loop num_runs times:
     ├→ extractive.py:get_summarizer()
     ├→ summarizer.summarize(text, num_sentences, seed_i)
     └→ Store result
   ↓
   metrics.py:calculate_consistency_score()
   metrics.py:calculate_pairwise_similarities()
   metrics.py:calculate_length_statistics()
   ↓
   Return results dictionary
   ```

5. **Results Display**:
   ```
   app.py receives results
   ↓
   Display metric cards (consistency, similarity, unique, variance)
   ↓
   Generate visualizations:
     ├→ Plotly gauge chart
     ├→ Plotly heatmap
     └→ Plotly histogram
   ↓
   Show summary table
   ↓
   Enable CSV export
   ```

## Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Runtime environment |
| Streamlit | 1.32.0 | Web framework |
| sumy | 0.11.0 | Summarization algorithms |
| NLTK | 3.8.1 | NLP preprocessing |
| scikit-learn | 1.4.0 | TF-IDF, vectorization |
| NumPy | 1.26.0 | Numerical computations |
| Pandas | 2.2.0 | Data manipulation |
| Plotly | 5.18.0 | Interactive visualizations |
| Matplotlib | 3.8.0 | Static plotting |
| pytest | 8.0.0 | Testing framework |

### Infrastructure

- **Docker**: Containerization with python:3.11-slim base (final image ~450MB)
- **Docker Compose**: One-command deployment
- **GitHub Actions**: CI/CD with test, lint, and build jobs
- **Nix**: Reproducible development environments
- **Makefile**: Build automation

## Deployment Architecture

### Docker Container

```
┌─────────────────────────────────────┐
│  Container: text-summarization-consistency
│  ┌─────────────────────────────────┐
│  │  Streamlit Server (port 8501)  │
│  │  ├─ Health Check Endpoint      │
│  │  ├─ WebSocket Connection        │
│  │  └─ Static Asset Server         │
│  └─────────────────────────────────┘
│  ┌─────────────────────────────────┐
│  │  Application Code               │
│  │  ├─ /app/src/*                  │
│  │  ├─ /app/data/*                 │
│  │  └─ /app/requirements.txt       │
│  └─────────────────────────────────┘
│  ┌─────────────────────────────────┐
│  │  Python Environment             │
│  │  ├─ Python 3.11                 │
│  │  ├─ Installed packages          │
│  │  └─ NLTK data                   │
│  └─────────────────────────────────┘
└─────────────────────────────────────┘
         │
         ▼
    Host Machine
    Port 8501
```

**Container Features**:
- Non-root user (appuser, uid 1000)
- Health check every 30 seconds
- Automatic restart on failure
- Read-only root filesystem (where possible)
- Minimal attack surface

### CI/CD Pipeline

```
GitHub Push/PR
    ↓
┌───────────────┐
│  Test Job     │
│  - pytest     │
│  - coverage   │
└───────┬───────┘
        │
┌───────▼───────┐
│  Lint Job     │
│  - flake8     │
│  - black      │
│  - isort      │
└───────┬───────┘
        │
┌───────▼───────┐
│  Docker Job   │
│  - build      │
│  - health     │
│  - cleanup    │
└───────────────┘
```

## Performance Considerations

### Memory Usage

- **Baseline**: ~150MB (Streamlit + Python)
- **Per Article**: ~1-2MB (text loading)
- **Per Experiment**: ~5-10MB (10 runs × ~0.5MB per summary)
- **Peak**: ~200MB during heavy processing

### CPU Usage

- **Idle**: <5% (waiting for user input)
- **Processing**: 20-80% (depends on text length, algorithm)
- **TextRank/LexRank**: More CPU-intensive (graph construction)
- **Luhn**: Lightweight (simple frequency analysis)

### Disk Usage

- **Docker Image**: ~450MB
- **Source Code**: ~2MB
- **Data Files**: ~100KB
- **NLTK Data**: ~5MB

### Scalability

**Current Design**: Single-user, single-threaded
- Suitable for educational/research use
- Can handle 1-10 concurrent users
- No database required

**Future Enhancements**:
- Multi-processing for parallel runs
- Caching of experiment results
- Database storage for history
- API endpoint for programmatic access

## Security Considerations

1. **Container Security**:
   - Non-root user execution
   - Minimal base image (python:3.11-slim)
   - No unnecessary packages
   - Regular base image updates

2. **Input Validation**:
   - Text length limits (prevent DoS)
   - Parameter range validation
   - CSV sanitization

3. **No External Dependencies**:
   - No external API calls
   - No user authentication (local deployment)
   - No sensitive data storage

## Extensibility

### Adding New Algorithms

1. Create class in `src/algorithms/extractive.py`:
```python
class NewSummarizer:
    def summarize(self, text: str, num_sentences: int, seed: int = 42) -> str:
        # Implementation
        pass
```

2. Update factory function:
```python
def get_summarizer(algorithm: str):
    if algorithm == "new_algorithm":
        return NewSummarizer()
```

3. Add UI option in `src/app.py`:
```python
algorithm = st.radio("Algorithm", ["TextRank", "LexRank", "Luhn", "New Algorithm"])
```

### Adding New Metrics

1. Implement in `src/utils/metrics.py`:
```python
def calculate_new_metric(summaries: List[str]) -> float:
    # Calculation
    pass
```

2. Update experiment results in `src/algorithms/consistency.py`

3. Display in UI with new metric card

### Adding Data Sources

1. Extend `src/utils/data.py`:
```python
def load_from_api(url: str) -> pd.DataFrame:
    # Fetch and parse
    pass
```

2. Add UI option for data source selection

## Maintenance

### Regular Tasks

- Update Python packages monthly
- Rebuild Docker image with latest base
- Review and update test suite
- Monitor CI/CD pipeline health

### Monitoring

- Docker health check (/_stcore/health)
- Application logs (docker logs)
- Test coverage reports (Codecov)
- CI pipeline status (GitHub Actions)

---

**Architecture Version**: 1.0
**Last Updated**: 2024
**Maintained By**: Atena Karimi (atenakarimii2001@gmail.com)
