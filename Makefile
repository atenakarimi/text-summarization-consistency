# Makefile for Text Summarization Consistency Analyzer

.PHONY: help install test lint format clean docker-build docker-run docker-stop all

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop Docker container"
	@echo "  make all          - Install, test, and build"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=127 --extend-ignore=E203,W503

format:
	black src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-build:
	docker build -t text-summarization-consistency:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

all: install test docker-build
