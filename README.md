# Text Summarization Consistency Analyzer

[![CI Pipeline](https://github.com/atenakarimi/text-summarization-consistency/actions/workflows/ci.yml/badge.svg)](https://github.com/atenakarimi/text-summarization-consistency/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A scientific tool for measuring and analyzing the **reproducibility and consistency** of text summarization algorithms. This project demonstrates how different extractive summarization methods behave when run multiple times on the same input, providing insights into their deterministic properties and result stability.

## 🎯 Purpose

While many text summarizers focus on producing "good" summaries, this tool addresses a fundamental question in computational reproducibility:

> **"How consistent are summarization algorithms when run multiple times?"**

This is critical for:
- **Scientific reproducibility**: Research requiring consistent results
- **Production systems**: Applications needing predictable behavior
- **Algorithm comparison**: Understanding deterministic vs. stochastic methods
- **Quality assurance**: Detecting unexpected variability in outputs

## ✨ Features

- **Three Extractive Algorithms**: TextRank, LexRank, and Luhn implementations
- **Consistency Measurement**: Quantifies reproducibility with scientific metrics
- **Interactive Web UI**: Clean Streamlit interface with real-time visualizations
- **Comprehensive Metrics**:
  - Consistency Score (0-100%)
  - Jaccard Similarity across runs
  - Length Variance Analysis
  - Unique Output Detection
- **Visualizations**:
  - Consistency Gauge with color-coded zones
  - Similarity Heatmap (interactive Plotly chart)
  - Length Distribution Histogram
- **Export Functionality**: Download results as CSV for further analysis
- **Sample Dataset**: 10 curated articles across Technology, Science, and Business

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/atenakarimi/text-summarization-consistency.git
cd text-summarization-consistency

# Start with docker-compose
docker-compose up

# Open browser to http://localhost:8501
```

That's it! The application will be available at [http://localhost:8501](http://localhost:8501).

### Option 2: Nix Shell (Reproducible Environment)

```bash
# Enter Nix shell
nix-shell

# Run the application
streamlit run src/app.py
```

### Option 3: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run application
streamlit run src/app.py
```

## 📊 Usage

1. **Select an Article**: Choose from 10 sample articles or paste your own text
2. **Choose Algorithm**: TextRank (graph-based), LexRank (similarity-based), or Luhn (frequency-based)
3. **Configure Experiment**:
   - Number of runs (3-20)
   - Summary length (2-10 sentences)
   - Random seed for reproducibility
4. **Run Experiment**: Click "Run Consistency Experiment"
5. **Analyze Results**:
   - View consistency score and metrics
   - Explore interactive visualizations
   - Examine individual summaries
   - Export data for further analysis

### Expected Results

- **Luhn Algorithm**: Should show ~100% consistency (fully deterministic)
- **TextRank/LexRank**: Will show variance due to random initialization (typically 60-85%)

## 📁 Project Structure

```
text-summarization-consistency/
├── src/
│   ├── app.py                     # Streamlit web application
│   ├── algorithms/
│   │   ├── extractive.py          # TextRank, LexRank, Luhn implementations
│   │   └── consistency.py         # Experiment runner and analysis
│   └── utils/
│       ├── data.py                # Data loading and text processing
│       └── metrics.py             # Evaluation metrics
├── data/
│   └── sample_articles.csv        # 10 curated articles
├── tests/
│   ├── test_algorithms.py         # Algorithm tests
│   ├── test_consistency.py        # Consistency experiment tests
│   ├── test_metrics.py            # Metrics calculation tests
│   └── test_data.py               # Data loading tests
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # One-command deployment
├── Makefile                        # Build automation
├── default.nix                     # Nix reproducible environment
├── requirements.txt                # Python dependencies
└── pytest.ini                      # Test configuration
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_algorithms.py -v

# Run linting
make lint

# Format code
make format
```

**Test Coverage**: 70+ tests covering:
- Algorithm correctness and determinism
- Consistency metrics calculation
- Data loading and validation
- Edge cases and error handling

## 🐳 Docker Commands

```bash
# Build image
make docker-build

# Run container
make docker-run

# View logs
make docker-logs

# Stop container
make docker-stop

# Complete workflow
make all  # Install + Test + Docker Build
```

## 📚 Algorithm Details

### TextRank
- **Type**: Graph-based extractive summarization
- **Method**: PageRank on sentence similarity graph
- **Consistency**: Variable (60-85%) due to random walk initialization
- **Best for**: Technical documents, news articles

### LexRank
- **Type**: Graph-based with eigenvector centrality
- **Method**: Modified PageRank with IDF-modified cosine similarity
- **Consistency**: Variable (65-90%) due to matrix operations
- **Best for**: Multi-document summarization, diverse content

### Luhn
- **Type**: Frequency-based heuristic
- **Method**: Scores sentences by significant word clusters
- **Consistency**: ~100% (fully deterministic)
- **Best for**: Reproducibility testing, simple summarization

## 🔬 Consistency Metrics Explained

- **Consistency Score**: Percentage of identical summaries across runs (100% = perfect consistency)
- **Average Similarity**: Mean Jaccard similarity between all summary pairs (0-1)
- **Unique Summaries**: Count of distinct outputs (1 = perfectly consistent)
- **Length Variance**: Standard deviation of summary lengths (0 = perfectly stable)

## 🛠️ Development

### Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- Nix (optional, for reproducible environment)

### Installation for Development

```bash
# Clone and install
git clone https://github.com/atenakarimi/text-summarization-consistency.git
cd text-summarization-consistency
make install

# Run tests
make test

# Start development server
./run.sh
```

### Adding New Algorithms

1. Create new class in [src/algorithms/extractive.py](src/algorithms/extractive.py)
2. Implement `summarize(text, num_sentences, seed)` method
3. Add factory function entry in `get_summarizer()`
4. Write tests in [tests/test_algorithms.py](tests/test_algorithms.py)

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{text_summarization_consistency,
  author = {Karimi, Atena},
  title = {Text Summarization Consistency Analyzer},
  year = {2024},
  url = {https://github.com/atenakarimi/text-summarization-consistency}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Use Black formatter (line length: 100)
- Add docstrings for all functions/classes
- Write tests for new features
- Ensure CI pipeline passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Atena Karimi**
- Email: atenakarimii2001@gmail.com
- Course: Reproducible Analytics Pipeline for Machine Learning, AI, and Data Science (RAP4MADS)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [sumy](https://github.com/miso-belica/sumy) for algorithm implementations
- Inspired by research in computational reproducibility
- Developed as a final project demonstrating reproducible research practices

## 📊 System Requirements

- **RAM**: 2GB minimum
- **Storage**: 500MB for Docker image, <100MB for source code
- **Network**: Internet connection for initial setup (Docker pull, pip install)
- **Browser**: Modern browser supporting HTML5 (Chrome, Firefox, Safari, Edge)

## 🔍 Troubleshooting

### Docker Issues

```bash
# Check container logs
docker logs text-summarization-consistency

# Restart container
docker-compose restart

# Rebuild image
docker-compose build --no-cache
```

### NLTK Data Missing

```bash
# Download manually
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Change 8502 to any available port
```

## 🎓 Educational Use

This project is designed for educational purposes and demonstrates:
- **Reproducible research principles**
- **Scientific experimentation with algorithms**
- **Docker containerization for consistency**
- **Comprehensive testing practices**
- **Clean code architecture**

Perfect for students learning about:
- Text summarization techniques
- Reproducibility in computational research
- Python application development
- CI/CD pipelines
- Docker deployment

---

**Questions or Issues?** Open an issue on GitHub or contact atenakarimii2001@gmail.com
