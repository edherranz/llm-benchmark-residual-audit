# Predictable Benchmarks? Evaluation-Protocol-Adjusted LLM Benchmark Residual Audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20319985.svg)](https://doi.org/10.5281/zenodo.20319985)

This repository contains the data, code, figures, and LaTeX source for a publication-ready paper on how much public LLM benchmark scores can be predicted from public model metadata and observable evaluation-protocol covariates.

The project is not a claim to measure intelligence or estimate a causal scaling law. It is a residual-audit framework: predict the portion of benchmark variation explained by public metadata and measurement protocol, then use residuals as a review queue for benchmark informativeness, data quality, source mismatch, model specialization, and protocol comparability.

## Repository layout

```text
data/      benchmark provenance files and model feature coding
src/       analysis code and feature definitions
outputs/   generated CSV outputs and diagnostics, committed as the release snapshot
figures/   generated figures used by the paper, committed as the release snapshot
paper/     canonical LaTeX manuscript, bibliography, generated tables, and PDF
scripts/   one-command reproduction helper
tests/     data-integrity tests
docs/      data dictionary, feature-coding rubric, reproducibility notes, and AI-use disclosure
```

## Canonical result

The canonical run uses official-like sources, a continuous scale policy, a logit-transformed percent-score target, ridge regression, and family-grouped cross-validation. In the supplied data snapshot it uses 250 primary percent-score cells, 41 models, 13 families, and 12 benchmark variants. The canonical full model has a source-weighted MAE of 9.28 percentage points.

The release also reports stronger checks: leave-one-family-out, leave-one-benchmark-out, temporal holdout, closed-source parameter uncertainty, paired bootstrap confidence intervals, and ordinal recipe-coding perturbation sensitivity. Stronger validation is intentionally less favorable; that is part of the paper's argument.

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the analysis and regenerate paper tables:

```bash
python src/run_analysis.py
python src/make_paper_tables.py
```

Compile the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or run everything from the repository root:

```bash
bash scripts/reproduce.sh
```

Run tests:

```bash
pytest -q
```

## Data and coding cautions

Gemini 2.5, Gemini 3, and Gemini 3.1 are included with benchmark/version/scale separation where available. Gemini 3.5 was not included in this snapshot because comparable canonical percent-score cells had not been audited for this release.

Closed-source parameter counts, training tokens, and recipe variables are public proxies, not direct observations. Closed-source scale values are low-confidence anchors and are flagged with `param_confidence` and `scale_imputed`. The ordinal recipe variables are documented in `docs/FEATURE_CODING_RUBRIC.md`, and the per-model coding table is in `data/model_feature_coding.csv` after reproduction.

## License

The repository is released under the MIT license. Institutional or employer policies should be checked before public posting under an organizational affiliation.
