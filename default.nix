# Text Summarization Consistency Analyzer - Nix Environment
# This file provides a reproducible development environment with all dependencies
# Following RAP4MADS best practices with pinned nixpkgs for full reproducibility

{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.05.tar.gz") {} }:

pkgs.mkShell {
  name = "text-summarization-consistency-env";
  
  buildInputs = with pkgs; [
    # Python 3.11
    python311
    python311Packages.pip
    python311Packages.virtualenv
    
    # Build dependencies
    gcc
    stdenv.cc.cc.lib
    zlib
    
    # Git for version control
    git
    
    # Docker tools (optional, for container builds)
    docker
    docker-compose
  ];
  
  shellHook = ''
    # Set up Python virtual environment
    if [ ! -d .venv ]; then
      echo "Creating Python virtual environment..."
      python -m venv .venv
    fi
    
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    
    # Install Python dependencies
    if [ -f requirements.txt ]; then
      echo "Installing Python dependencies..."
      pip install -r requirements.txt > /dev/null 2>&1
    fi
    
    # Download required NLTK data
    echo "Downloading NLTK data..."
    python -c "
import nltk
import os
nltk_data_dir = os.path.join(os.getcwd(), '.nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
" > /dev/null 2>&1
    
    # Set environment variables
    export NLTK_DATA="$PWD/.nltk_data"
    
    # Display environment info
    echo "=========================================================="
    echo "Text Summarization Consistency Analyzer - Dev Environment"
    echo "=========================================================="
    echo "Python: $(python --version)"
    echo "Pip: $(pip --version | cut -d' ' -f1-2)"
    echo ""
    echo "Available commands:"
    echo "  streamlit run src/app.py     - Run the application"
    echo "  pytest tests/                - Run all tests"
    echo "  pytest tests/ -v             - Run tests with verbose output"
    echo "  make help                    - Show all make targets"
    echo "  docker-compose up            - Run in Docker"
    echo ""
    echo "Quick start:"
    echo "  ./run.sh                     - Start the app (easiest)"
    echo "=========================================================="
  '';
  
  # Set library path for dynamic linking
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib";
}
