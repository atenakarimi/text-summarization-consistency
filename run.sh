#!/bin/bash

# Run script for local development
# Makes it easy to start the application without Docker

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# Install dependencies if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Run Streamlit app
streamlit run src/app.py
