# v7.2 changelog

Publication-polish release addressing reviewer-style feedback.

- Added paired bootstrap confidence intervals for Full and Scale-only OOF MAE and their improvement.
- Renamed leave-one-benchmark metrics in `summary.json` to distinguish pooled held-out-cell MAE from the unweighted mean of per-benchmark fold MAEs.
- Added ordinal recipe-code perturbation sensitivity for `math_intensity`, `code_pretrain`, and `synthetic_quality`.
- Added `data/model_feature_coding.csv` / `outputs/model_feature_coding.csv` and a feature-coding rubric.
- Added explicit sparse-benchmark disclosure near the benchmark residual table.
- Added AI-use disclosure to the paper and a separate disclosure document.
- Clarified that closed-source Gemini scale values are anchoring proxies, not parameter estimates.
- Cleaned repository hygiene: single README, canonical `paper/main.tex`, `LICENSE`, filled `CITATION.cff`, and removed dead `benchmark_suite_v3.py`.
