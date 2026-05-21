"""
model_features.py - model-level feature definitions for the benchmark residual audit.

Important caution for public release:
  * active_params_B, total_params_B, and training_tokens_B are direct public
    values only for open or clearly documented models.
  * closed-source values are anchoring proxies used for robustness analysis, not
    factual parameter-count claims.
  * low-confidence/imputed rows are explicitly flagged through param_confidence
    and scale_imputed, and the analysis includes stress tests and scale-bucket
    sensitivity runs to reduce dependence on exact closed-source point values.
"""
from __future__ import annotations
from copy import deepcopy
from model_features_v3 import MODEL_FEATURES as MODEL_FEATURES_V3


def _base_convert(name: str, info: dict) -> dict:
    active = float(info.get("params", 1.0))
    total = active
    is_moe = 0
    # Open-weight MoE totals where public reporting is sufficiently clear for
    # use as capacity proxies. Active parameter values inherit v3's convention.
    moe_totals = {
        "Mixtral-8x7B": 46.7,
        "DeepSeek-V3-Base": 671.0,
        "DeepSeek-V3": 671.0,
        "DeepSeek-R1": 671.0,
        "DeepSeek-R1-0528": 671.0,
        "Qwen3-235B-A22B-Base": 235.0,
        "Qwen3-235B-A22B-Thinking": 235.0,
        "Qwen3-30B-A3B-Base": 30.0,
        "Llama-4-Scout": 109.0,
        "Llama-4-Maverick": 400.0,
    }
    if name in moe_totals:
        total = moe_totals[name]
        is_moe = 1
    closed = int(info.get("is_closed_source", 0))
    # Closed-source parameter counts are not observable and are retained only as
    # weak ordinal/informative priors. The v4 analysis includes sensitivity runs
    # that replace these with scale buckets.
    param_conf = 0.35 if closed else 0.95
    scale_imputed = 1 if closed else 0
    scale_bucket = 1
    if active >= 1000:
        scale_bucket = 5
    elif active >= 300:
        scale_bucket = 4
    elif active >= 70:
        scale_bucket = 3
    elif active >= 7:
        scale_bucket = 2
    out = dict(info)
    out.update(
        active_params_B=active,
        total_params_B=total,
        is_moe=is_moe,
        param_confidence=param_conf,
        scale_imputed=scale_imputed,
        scale_bucket=scale_bucket,
    )
    return out

MODEL_FEATURES = {name: _base_convert(name, deepcopy(info)) for name, info in MODEL_FEATURES_V3.items()}

# Additional Gemini-family rows. The numeric scale fields for closed Gemini
# models are deliberately low-confidence anchoring proxies. They should not be
# quoted as parameter estimates. Use param_confidence and scale_imputed when
# interpreting any result that depends on these rows.
MODEL_FEATURES.update({
    "Gemini-2.5-Flash": dict(
        params=70.0, active_params_B=70.0, total_params_B=400.0,
        context_length=1000000, family="Gemini", training_tokens_B=20000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.45,
        is_moe=1, param_confidence=0.25, scale_imputed=1, scale_bucket=3,
    ),
    "Gemini-3-Pro": dict(
        params=500.0, active_params_B=500.0, total_params_B=1200.0,
        context_length=1000000, family="Gemini", training_tokens_B=30000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.88,
        is_moe=1, param_confidence=0.20, scale_imputed=1, scale_bucket=4,
    ),
    "Gemini-3.1-Pro": dict(
        params=550.0, active_params_B=550.0, total_params_B=1400.0,
        context_length=1000000, family="Gemini", training_tokens_B=32000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2026.14,
        is_moe=1, param_confidence=0.20, scale_imputed=1, scale_bucket=4,
    ),
})


def models_with_min_coverage(provenance_df, min_benchmarks=4):
    counts = provenance_df.groupby("model").size()
    return counts[counts >= min_benchmarks].index.tolist()
