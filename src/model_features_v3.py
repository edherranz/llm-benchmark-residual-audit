"""
model_features_v3.py — Model architectural/recipe features for v3
=================================================================

This module contains the per-model STRUCTURAL fields (params, training tokens,
family, recipe features, etc.). Benchmark scores live separately in
benchmark_provenance_v3.csv.

Rationale: v2.2 had benchmark scores baked into a single dict per model.
That worked for 100 models × 11 benchmarks because all cells were populated.
In v3 most cells are NaN (modern benchmarks didn't exist when older models
were evaluated), so we separate (model × features) from (model × benchmark)
and let the analysis handle NaN per-benchmark.

Models are drawn from three pools:
  1. Modern frontier models that have substantial coverage on the new suite
     (DeepSeek V3/R1, Qwen3, Llama 4, Gemma 3, Claude 4, GPT-5, o3, etc.)
  2. Legacy v2.2 models that overlap meaningfully (still have ≥3 scores in
     the v3 suite via legacy MMLU/ARC-C/GSM8K/HumanEval/MBPP/MATH retention)
  3. Mid-tier 2024 models (Qwen2.5, Phi-3/4, Llama 3.x) that bridge the
     two pools

We DO NOT include pre-2023 models (Pythia, BLOOM, GPT-2) — every modern
benchmark in v3 postdates them, so they would carry 0 valid scores beyond
ARC-C and HumanEval at ≈0 accuracy, contributing nothing but extreme
end-point pull to the regression.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# MODEL FEATURES (no benchmark scores here — those live in provenance CSV)
# ----------------------------------------------------------------------------
MODEL_FEATURES = {

    # ========== 2023 BASELINE (kept for low-saturation anchors) ==========
    "Llama-2-7B": dict(
        params=7.0, context_length=4096, family="Llama2", training_tokens_B=2000,
        attention_type=0, position_encoding=1, vocab_size_k=32,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=0, code_pretrain=1, synthetic_quality=0,
        trained_web_only=1, is_distilled=0, release_year=2023.5,
    ),
    "Llama-2-70B": dict(
        params=70.0, context_length=4096, family="Llama2", training_tokens_B=2000,
        attention_type=1, position_encoding=1, vocab_size_k=32,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=0, code_pretrain=1, synthetic_quality=0,
        trained_web_only=1, is_distilled=0, release_year=2023.5,
    ),
    "Mistral-7B": dict(
        params=7.0, context_length=8192, family="Mistral", training_tokens_B=2500,
        attention_type=1, position_encoding=1, vocab_size_k=32,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2023.7,
    ),
    "Mixtral-8x7B": dict(
        params=12.9, context_length=32768, family="Mistral", training_tokens_B=3000,
        attention_type=1, position_encoding=1, vocab_size_k=32,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2023.9,
    ),
    "Phi-2": dict(
        params=2.7, context_length=2048, family="Phi", training_tokens_B=1400,
        attention_type=0, position_encoding=1, vocab_size_k=50,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2023.9,
    ),

    # ========== 2024 MID-FRONTIER ==========
    "Llama-3.1-8B": dict(
        params=8.0, context_length=128000, family="Llama3", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=128,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=2, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2024.5,
    ),
    "Llama-3.1-70B": dict(
        params=70.0, context_length=128000, family="Llama3", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=128,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=2, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2024.5,
    ),
    "Llama-3.1-405B": dict(
        params=405.0, context_length=128000, family="Llama3", training_tokens_B=15600,
        attention_type=1, position_encoding=1, vocab_size_k=128,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=2, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2024.5,
    ),
    "Llama-3.3-70B": dict(
        params=70.0, context_length=128000, family="Llama3", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=128,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2024.9,
    ),
    "Qwen2.5-7B": dict(
        params=7.0, context_length=128000, family="Qwen2", training_tokens_B=18000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.7,
    ),
    "Qwen2.5-72B": dict(
        params=72.0, context_length=128000, family="Qwen2", training_tokens_B=18000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.7,
    ),
    "Phi-4": dict(
        params=14.0, context_length=16384, family="Phi", training_tokens_B=9800,
        attention_type=1, position_encoding=1, vocab_size_k=100,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2024.9,
    ),
    "GPT-4o": dict(
        params=200.0, context_length=128000, family="GPT4", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=0, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.4,
    ),
    "Claude-3.5-Sonnet": dict(
        params=70.0, context_length=200000, family="Claude", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=64,
        is_reasoning_model=0, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.5,
    ),

    # ========== LATE 2024 / EARLY 2025 ==========
    "DeepSeek-V3-Base": dict(
        params=37.0, context_length=128000, family="DeepSeek", training_tokens_B=14800,
        attention_type=1, position_encoding=1, vocab_size_k=129,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.95,
    ),
    "DeepSeek-V3": dict(
        params=37.0, context_length=128000, family="DeepSeek", training_tokens_B=14800,
        attention_type=1, position_encoding=1, vocab_size_k=129,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.95,
    ),
    "DeepSeek-R1": dict(
        params=37.0, context_length=128000, family="DeepSeek", training_tokens_B=14800,
        attention_type=1, position_encoding=1, vocab_size_k=129,
        is_reasoning_model=1, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.1,
    ),
    "DeepSeek-R1-0528": dict(
        params=37.0, context_length=128000, family="DeepSeek", training_tokens_B=14800,
        attention_type=1, position_encoding=1, vocab_size_k=129,
        is_reasoning_model=1, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.42,
    ),
    "o1-1217": dict(
        params=200.0, context_length=128000, family="OpenAI_o", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2024.95,
    ),
    "o1-mini": dict(
        params=50.0, context_length=128000, family="OpenAI_o", training_tokens_B=12000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2024.7,
    ),
    "o3-mini": dict(
        params=50.0, context_length=128000, family="OpenAI_o", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.08,
    ),
    "Mistral-Small-3": dict(
        params=24.0, context_length=32000, family="Mistral", training_tokens_B=8000,
        attention_type=1, position_encoding=1, vocab_size_k=131,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=1,
        trained_web_only=0, is_distilled=0, release_year=2025.08,
    ),
    "Phi-4-mini": dict(
        params=3.8, context_length=128000, family="Phi", training_tokens_B=5000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.17,
    ),

    # ========== Q2 2025 ==========
    "Claude-3.7-Sonnet": dict(
        params=80.0, context_length=200000, family="Claude", training_tokens_B=15000,
        attention_type=1, position_encoding=1, vocab_size_k=64,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.17,
    ),
    "GPT-4.5": dict(
        params=2000.0, context_length=128000, family="GPT4", training_tokens_B=20000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=0, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.17,
    ),
    "Gemma-3-4B-Base": dict(
        params=4.0, context_length=128000, family="Gemma", training_tokens_B=4000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.25,
    ),
    "Gemma-3-12B-Base": dict(
        params=12.0, context_length=128000, family="Gemma", training_tokens_B=12000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.25,
    ),
    "Gemma-3-27B-Base": dict(
        params=27.0, context_length=128000, family="Gemma", training_tokens_B=14000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=1, code_pretrain=1, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.25,
    ),
    "Gemma-3-27B-IT": dict(
        params=27.0, context_length=128000, family="Gemma", training_tokens_B=14000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.25,
    ),
    "Llama-4-Scout": dict(
        params=17.0, context_length=10000000, family="Llama4", training_tokens_B=30000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.27,
    ),
    "Llama-4-Maverick": dict(
        params=17.0, context_length=1000000, family="Llama4", training_tokens_B=30000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=0, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.27,
    ),
    "o3": dict(
        params=200.0, context_length=128000, family="OpenAI_o", training_tokens_B=18000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.33,
    ),
    "Gemini-2.5-Pro": dict(
        params=400.0, context_length=2000000, family="Gemini", training_tokens_B=20000,
        attention_type=1, position_encoding=1, vocab_size_k=262,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.33,
    ),
    "Qwen3-14B-Base": dict(
        params=14.0, context_length=128000, family="Qwen3", training_tokens_B=36000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.42,
    ),
    "Qwen3-30B-A3B-Base": dict(
        params=3.0, context_length=128000, family="Qwen3", training_tokens_B=36000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.42,
    ),
    "Qwen3-32B-Base": dict(
        params=32.0, context_length=128000, family="Qwen3", training_tokens_B=36000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=1, release_year=2025.42,
    ),
    "Qwen3-235B-A22B-Base": dict(
        params=22.0, context_length=128000, family="Qwen3", training_tokens_B=36000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=0, is_instruct=0, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.42,
    ),
    "Qwen3-235B-A22B-Thinking": dict(
        params=22.0, context_length=128000, family="Qwen3", training_tokens_B=36000,
        attention_type=1, position_encoding=1, vocab_size_k=152,
        is_reasoning_model=1, is_instruct=1, is_closed_source=0,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.42,
    ),
    "Claude-Sonnet-4": dict(
        params=80.0, context_length=200000, family="Claude", training_tokens_B=18000,
        attention_type=1, position_encoding=1, vocab_size_k=64,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.42,
    ),
    "Claude-Opus-4": dict(
        params=200.0, context_length=200000, family="Claude", training_tokens_B=20000,
        attention_type=1, position_encoding=1, vocab_size_k=64,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.42,
    ),
    "Grok-4": dict(
        params=300.0, context_length=256000, family="Grok", training_tokens_B=18000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.50,
    ),
    "GPT-5": dict(
        params=500.0, context_length=400000, family="GPT4", training_tokens_B=30000,
        attention_type=1, position_encoding=1, vocab_size_k=200,
        is_reasoning_model=1, is_instruct=1, is_closed_source=1,
        math_intensity=2, code_pretrain=2, synthetic_quality=2,
        trained_web_only=0, is_distilled=0, release_year=2025.58,
    ),
}


def models_with_min_coverage(provenance_df, min_benchmarks=4):
    """Return list of models with at least N benchmark scores in the provenance."""
    counts = provenance_df.groupby('model').size()
    return counts[counts >= min_benchmarks].index.tolist()


# ----------------------------------------------------------------------------
# Pull v8 legacy MMLU/ARC_C/GSM8K/HumanEval/MBPP/MATH for overlap models
# This enriches v3's wide-format coverage on the retained legacy benchmarks
# ----------------------------------------------------------------------------
LEGACY_CARRYOVER_FROM_V8 = {
    # model_in_v3        : (model_name_in_v8, [legacy benchmarks to carry])
    "Llama-2-7B":          ("Llama-2-7B",      ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Llama-2-70B":         ("Llama-2-70B",     ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Mistral-7B":          ("Mistral-7B",      ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Mixtral-8x7B":        ("Mixtral-8x7B",    ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Phi-2":               ("Phi-2",           ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Llama-3.1-8B":        ("Llama-3.1-8B",    ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Llama-3.1-70B":       ("Llama-3.1-70B",   ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Llama-3.1-405B":      ("Llama-3.1-405B",  ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Llama-3.3-70B":       ("Llama-3.3-70B-Inst", ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Qwen2.5-7B":          ("Qwen2.5-7B",      ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Qwen2.5-72B":         ("Qwen2.5-72B",     ["MMLU","ARC_C","GSM8K","HumanEval","MBPP","MATH"]),
    "Phi-4":               ("Phi-4",           ["ARC_C"]),  # we already have most modern scores
    "GPT-4o":              ("GPT-4o",          ["ARC_C", "MBPP"]),
    "Claude-3.5-Sonnet":   ("Claude-3.5-Sonnet", ["ARC_C", "MBPP"]),
}


def import_legacy_carryover():
    """Generate provenance records for legacy benchmarks carried from v8."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model_database_v8 import MODELS_DATABASE
    records = []
    src_tag = "v8 canonical database (legacy benchmark carryover)"
    for v3_name, (v8_name, benchs) in LEGACY_CARRYOVER_FROM_V8.items():
        if v8_name not in MODELS_DATABASE:
            continue
        info = MODELS_DATABASE[v8_name]
        src_detail = info.get('source_detail', src_tag)
        src = info.get('source', 'leaderboard')
        for b in benchs:
            if b in info:
                records.append(dict(
                    model=v3_name, benchmark=b, score=float(info[b]),
                    source=src, source_detail=src_detail,
                    shot=None, protocol='legacy (see v8)', thinking='na',
                    date='2024 (v8 capture)',
                ))
    return records


if __name__ == '__main__':
    # Build merged provenance
    import pandas as pd
    from benchmark_provenance_v3 import PROVENANCE_RECORDS
    
    legacy = import_legacy_carryover()
    print(f"Legacy carryover records: {len(legacy)}")
    all_records = PROVENANCE_RECORDS + legacy
    df = pd.DataFrame(all_records)
    
    # De-dup: if same (model, benchmark) appears twice, keep first
    df = df.drop_duplicates(subset=['model', 'benchmark'], keep='first')
    print(f"Total records after dedup: {len(df)}")
    print(f"Unique models: {df['model'].nunique()}")
    
    df.to_csv('benchmark_provenance_v3_merged.csv', index=False)
    print("Wrote benchmark_provenance_v3_merged.csv")
    
    # Wide-format coverage check
    from benchmark_suite_v3 import BENCHMARKS_V3
    wide = df.pivot_table(index='model', columns='benchmark', values='score', aggfunc='first')
    for b in BENCHMARKS_V3:
        if b not in wide.columns:
            wide[b] = float('nan')
    wide = wide[BENCHMARKS_V3]
    
    # Restrict to models in MODEL_FEATURES
    wide = wide.loc[wide.index.isin(MODEL_FEATURES)]
    print(f"\nWide-format coverage ({len(wide)} models with features):")
    cov = wide.notna().sum() / len(wide) * 100
    print(f"{'Benchmark':<16} | {'%':>6} | n")
    print('-'*36)
    for b in BENCHMARKS_V3:
        c = cov[b]
        n = wide[b].notna().sum()
        print(f"{b:<16} | {c:5.1f}% | {n}/{len(wide)}")
    
    # Per-model coverage
    print(f"\nPer-model coverage (#benchmarks with scores):")
    per_model = wide.notna().sum(axis=1).sort_values(ascending=False)
    for m, n in per_model.head(10).items():
        print(f"  {m:<28} : {n}/12")
    print(f"  ... (top 10 of {len(per_model)})")
    print(f"  median per-model coverage: {per_model.median():.1f}/12")
    print(f"  min: {per_model.min()}, max: {per_model.max()}")
