"""
benchmark_suite_v4.py - benchmark metadata for the v4 LLM benchmark study.

The v4 suite makes two changes relative to v3:
  1. It separates benchmark *families* from benchmark *variants* so AIME 2024,
     AIME 2025, LiveCodeBench, and LiveCodeBench Pro are not silently mixed.
  2. It marks which variants are used in the primary percent-score regressions.
     Non-comparable score scales such as Elo are kept for provenance but excluded
     from percent-score prediction unless explicitly requested.
"""
from __future__ import annotations

BENCHMARK_SUITE_V4 = {
    # Legacy retained for continuity.
    "MMLU": dict(category="knowledge", family="MMLU", variant="MMLU_original",
                 metric_type="percent", score_direction="higher", saturation_2026=93,
                 n_choices=4, format="MCQ", output_space=2, retained_v22=True,
                 contamination_resistant=False, include_primary=True),
    "ARC_C": dict(category="reasoning", family="ARC", variant="ARC_Challenge",
                  metric_type="percent", score_direction="higher", saturation_2026=97,
                  n_choices=4, format="MCQ", output_space=2, retained_v22=True,
                  contamination_resistant=False, include_primary=True),
    "GSM8K": dict(category="math_easy", family="GSM8K", variant="GSM8K",
                  metric_type="percent", score_direction="higher", saturation_2026=98,
                  n_choices=0, format="free_form_math", output_space=1, retained_v22=True,
                  contamination_resistant=False, include_primary=True),
    "HumanEval": dict(category="code", family="HumanEval", variant="HumanEval",
                      metric_type="percent", score_direction="higher", saturation_2026=97,
                      n_choices=0, format="free_form_code", output_space=0, retained_v22=True,
                      contamination_resistant=False, include_primary=True),
    "MATH": dict(category="math_hard", family="MATH", variant="MATH_500_or_full",
                 metric_type="percent", score_direction="higher", saturation_2026=95,
                 n_choices=0, format="free_form_math", output_space=0, retained_v22=True,
                 contamination_resistant=False, include_primary=True),
    "MBPP": dict(category="code", family="MBPP", variant="MBPP",
                 metric_type="percent", score_direction="higher", saturation_2026=95,
                 n_choices=0, format="free_form_code", output_space=0, retained_v22=True,
                 contamination_resistant=False, include_primary=True),

    # Modern v3/v4 primary benchmarks.
    "MMLU_Pro": dict(category="knowledge", family="MMLU_Pro", variant="MMLU_Pro",
                     metric_type="percent", score_direction="higher", saturation_2026=89,
                     n_choices=10, format="MCQ", output_space=2, retained_v22=False,
                     contamination_resistant=True, include_primary=True),
    "BBH": dict(category="reasoning", family="BBH", variant="BBH",
                metric_type="percent", score_direction="higher", saturation_2026=92,
                n_choices=0, format="mixed", output_space=1, retained_v22=False,
                contamination_resistant=False, include_primary=True),
    "GPQA_Diamond": dict(category="science_reasoning", family="GPQA", variant="GPQA_Diamond",
                         metric_type="percent", score_direction="higher", saturation_2026=94,
                         n_choices=4, format="MCQ", output_space=2, retained_v22=False,
                         contamination_resistant=True, include_primary=True),
    "AIME_2024": dict(category="olympiad_math", family="AIME", variant="AIME_2024",
                      metric_type="percent", score_direction="higher", saturation_2026=95,
                      n_choices=0, format="free_form_math", output_space=1, retained_v22=False,
                      contamination_resistant=True, include_primary=True),
    "LiveCodeBench": dict(category="competitive_code", family="LiveCodeBench", variant="LCB_2024_2025_window",
                          metric_type="percent", score_direction="higher", saturation_2026=92,
                          n_choices=0, format="free_form_code", output_space=0, retained_v22=False,
                          contamination_resistant=True, include_primary=True),
    "IFEval": dict(category="instruction_following", family="IFEval", variant="IFEval_prompt_strict_or_reported",
                   metric_type="percent", score_direction="higher", saturation_2026=93,
                   n_choices=0, format="binary_verifiable", output_space=0, retained_v22=False,
                   contamination_resistant=False, include_primary=True),

    # Versioned extensions. They are retained in the data but excluded from the
    # primary v4 regression unless coverage becomes adequate.
    "AIME_2025": dict(category="olympiad_math", family="AIME", variant="AIME_2025",
                      metric_type="percent", score_direction="higher", saturation_2026=95,
                      n_choices=0, format="free_form_math", output_space=1, retained_v22=False,
                      contamination_resistant=True, include_primary=False),
    "LiveCodeBench_Pro_Elo": dict(category="competitive_code", family="LiveCodeBench", variant="LCB_Pro_Elo",
                                  metric_type="elo", score_direction="higher", saturation_2026=None,
                                  n_choices=0, format="free_form_code", output_space=0, retained_v22=False,
                                  contamination_resistant=True, include_primary=False),
    "MMMLU": dict(category="multilingual_knowledge", family="MMMLU", variant="MMMLU",
                  metric_type="percent", score_direction="higher", saturation_2026=93,
                  n_choices=4, format="MCQ", output_space=2, retained_v22=False,
                  contamination_resistant=False, include_primary=False),
    "Humanitys_Last_Exam": dict(category="frontier_reasoning", family="HLE", variant="HLE_full_text_mm",
                                metric_type="percent", score_direction="higher", saturation_2026=None,
                                n_choices=0, format="mixed", output_space=0, retained_v22=False,
                                contamination_resistant=True, include_primary=False),
    "Aider_Polyglot": dict(category="code", family="Aider", variant="Aider_Polyglot",
                           metric_type="percent", score_direction="higher", saturation_2026=None,
                           n_choices=0, format="free_form_code", output_space=0, retained_v22=False,
                           contamination_resistant=True, include_primary=False),
    "SWE_bench_Verified": dict(category="agentic_code", family="SWE_bench", variant="SWE_bench_Verified_single_attempt",
                               metric_type="percent", score_direction="higher", saturation_2026=None,
                               n_choices=0, format="agentic_code", output_space=0, retained_v22=False,
                               contamination_resistant=True, include_primary=False),
    "SimpleQA": dict(category="factuality", family="SimpleQA", variant="SimpleQA",
                     metric_type="percent", score_direction="higher", saturation_2026=None,
                     n_choices=0, format="short_answer", output_space=0, retained_v22=False,
                     contamination_resistant=True, include_primary=False),
    "FACTS_Grounding": dict(category="factuality", family="FACTS", variant="FACTS_Grounding",
                            metric_type="percent", score_direction="higher", saturation_2026=None,
                            n_choices=0, format="grounded_qa", output_space=0, retained_v22=False,
                            contamination_resistant=True, include_primary=False),
}

BENCHMARKS_V4 = list(BENCHMARK_SUITE_V4.keys())
PRIMARY_BENCHMARKS_V4 = [b for b, m in BENCHMARK_SUITE_V4.items() if m["include_primary"] and m["metric_type"] == "percent"]
EXTENSION_BENCHMARKS_V4 = [b for b in BENCHMARKS_V4 if b not in PRIMARY_BENCHMARKS_V4]

# Higher substitutability means a benchmark can more easily be improved by
# scale or surface-pattern memorization alone. Lower values are expected to
# show larger gains from recipe/reasoning features.
SUBSTITUTABILITY_HAND_V4 = {
    "MMLU": 4.0,
    "ARC_C": 4.0,
    "GSM8K": 2.5,
    "HumanEval": 1.8,
    "MATH": 1.8,
    "MBPP": 2.0,
    "MMLU_Pro": 2.7,
    "BBH": 2.3,
    "GPQA_Diamond": 1.3,
    "AIME_2024": 1.0,
    "LiveCodeBench": 1.0,
    "IFEval": 2.2,
    "AIME_2025": 1.0,
    "LiveCodeBench_Pro_Elo": 1.0,
    "MMMLU": 3.0,
    "Humanitys_Last_Exam": 0.8,
    "Aider_Polyglot": 1.0,
    "SWE_bench_Verified": 1.0,
    "SimpleQA": 1.2,
    "FACTS_Grounding": 1.2,
}
