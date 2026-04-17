#!/usr/bin/env bash
# ARIA — Autonomous Research Intelligence Assistant
# One-shot environment setup script
set -e

echo "==> Setting up ARIA project environment"
echo "==> Using Python: $(python3 --version)"

# ── Create isolated virtual environment ────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "==> .venv already exists — skipping creation"
else
    echo "==> Creating virtual environment at .venv"
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "==> Virtual environment active: $(python --version)"

# ── Install dependencies ───────────────────────────────────────────────────────
pip install --upgrade pip --quiet
echo "==> Installing dependencies"
pip install -r requirements.txt

# ── Download language models ───────────────────────────────────────────────────
echo "==> Downloading spaCy English model"
python -m spacy download en_core_web_sm

echo "==> Downloading NLTK data"
python -c "
import nltk
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
print('NLTK data ready.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Run the app:               streamlit run app.py"
echo ""
echo "  VS Code: Select .venv/bin/python as your interpreter"
echo "  (Cmd+Shift+P → Python: Select Interpreter → .venv)"
