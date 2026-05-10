#!/bin/bash

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -q -r requirements.txt
fi

PORT="${PORT:-8501}"
echo ""
echo "Launching demo at http://localhost:$PORT"
echo "Press Ctrl+C to stop."
echo ""
streamlit run app.py --server.port "$PORT" --browser.gatherUsageStats false
