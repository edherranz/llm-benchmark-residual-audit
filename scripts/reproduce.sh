#!/usr/bin/env bash
set -euo pipefail
python src/run_analysis.py
python src/make_paper_tables.py
(cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
