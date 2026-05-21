# Feature coding rubric

This repository uses public model metadata and coarse ordinal recipe variables. These variables are audit controls, not direct observations of private training runs.

## Ordinal recipe variables

The following variables use a 0-2 scale:

- `math_intensity`: 0 = no public evidence of math-focused training or reasoning specialization; 1 = moderate public evidence, such as reported math data, math RL, or strong math emphasis in a technical report; 2 = explicit math/reasoning specialization, reasoning-model positioning, or repeated public emphasis on advanced mathematical reasoning.
- `code_pretrain`: 0 = no specific public evidence of code-heavy pretraining/post-training; 1 = moderate code emphasis or competitive coding benchmarks; 2 = explicit code specialization or strong public documentation of code-heavy training/post-training.
- `synthetic_quality`: 0 = no public evidence of synthetic-data/distillation emphasis; 1 = moderate evidence that synthetic data or distillation was material; 2 = explicit public emphasis on synthetic data, self-improvement, distillation, verifier-generated data, or reasoning traces.

These codes are intentionally coarse. The analysis includes `outputs/recipe_coding_sensitivity.csv`, where each model-level ordinal code is perturbed by one step and the canonical Full model is rerun. The public per-model coding file is `data/model_feature_coding.csv` and is also copied to `outputs/model_feature_coding.csv` after reproduction.

## Closed-source scale variables

For closed-source models, `active_params_B`, `total_params_B`, and `training_tokens_B` are low-confidence anchoring proxies. They are not factual claims and should not be cited as parameter estimates. Such rows have low `param_confidence` and `scale_imputed=1`. The analysis reports a closed-source scale/token perturbation stress test and a scale-bucket sensitivity run.

## Source and protocol variables

Source and protocol controls are coded from observable reporting metadata. They include source class, source-confidence weight, protocol-match flag, thinking/reasoning setting, tool access, and reasoning-effort field when available. These are not assumed to recover the full evaluation procedure; they are included to avoid treating obviously heterogeneous reported scores as exchangeable.
