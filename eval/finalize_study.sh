#!/bin/bash


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate

echo "[1/3] Aggregating results into summary.csv..."
python eval/comprehensive_study.py summarize

echo ""
echo "[2/3] Generating comparison plots..."
python eval/plot_results.py

echo ""
echo "[3/3] Generating final HTML+PDF report..."
python eval/generate_study_report.py

echo ""
echo "Study artifacts:"
STUDY_NAME="${STUDY_NAME:-study}"
echo "  Summary CSV:  eval/results/${STUDY_NAME}/summary.csv"
echo "  Plots:        eval/results/${STUDY_NAME}/plots/*.png"
echo "  HTML report:  eval/results/${STUDY_NAME}/REPORT.html"
echo "  PDF report:   eval/results/${STUDY_NAME}/REPORT.pdf"
