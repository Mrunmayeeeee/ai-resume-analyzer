#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ['punkt', 'stopwords', 'averaged_perceptron_tagger']]"
