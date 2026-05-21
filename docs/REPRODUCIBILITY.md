# Reproducibility notes

Canonical run:

```bash
python src/run_analysis.py
```

Compile paper after running the analysis:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

One-command helper:

```bash
bash scripts/reproduce.sh
```

The random seed is fixed in the analysis code. The analysis is deterministic conditional on package versions and the input CSV files.
