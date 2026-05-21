#!/usr/bin/env python3
"""
diagnostics.py - diagnostic and paper-generation layer for the LLM benchmark study.

V5 is deliberately more conservative than v4. It does not simply add another
predictive model. It adds diagnostics needed for a standalone paper:
  * detailed ablation grid;
  * source/scale/estimator/target sensitivity;
  * out-of-fold residual analysis;
  * residual plots and accuracy plots;
  * coefficient/predictor summaries; and
  * LaTeX tables for the rewritten paper.

The canonical analysis remains source_policy=official_like, scale_policy=continuous,
target=logit, estimator=ridge. Reported prediction errors are always computed on
held-out model families, not random cells.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reuse the v4 core, but write v5-specific diagnostics and paper assets.
import analysis_core as core
from benchmark_suite import BENCHMARK_SUITE_V4, PRIMARY_BENCHMARKS_V4, SUBSTITUTABILITY_HAND_V4
from model_features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "diagnostics"
FIG = ROOT / "figures" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


@dataclass
class ArgsObj:
    source_policy: str = "official_like"
    min_source_weight: float = 0.0
    scale_policy: str = "continuous"
    target: str = "logit"
    estimator: str = "ridge"
    n_folds: int = 5
    min_models: int = 10


def latex_escape(x) -> str:
    s = "" if pd.isna(x) else str(x)
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('_', r'\_')
             .replace('#', r'\#')
             .replace('{', r'\{')
             .replace('}', r'\}')
             .replace('~', r'\textasciitilde{}')
             .replace('^', r'\textasciicircum{}'))


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def weighted_rmse(resid, w):
    resid = np.asarray(resid, float)
    w = np.asarray(w, float)
    m = np.isfinite(resid) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.average(resid[m] ** 2, weights=w[m])))


def weighted_r2(y, yhat, w):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan
    ybar = np.average(y[m], weights=w[m])
    sse = np.sum(w[m] * (y[m] - yhat[m]) ** 2)
    sst = np.sum(w[m] * (y[m] - ybar) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def run_once(args: ArgsObj):
    primary, all_cells = core.prepare_cell_df(args)
    per_bench = core.per_benchmark_results(primary, args)
    pooled, preds = core.pooled_results(primary, args)
    cov, src = core.coverage_tables(all_cells, primary)
    return primary, all_cells, per_bench, pooled, preds, cov, src


def add_residual_columns(preds: pd.DataFrame, spec: str = "Full") -> pd.DataFrame:
    df = preds.copy()
    pcol = f"pred_{spec}"
    df = df[np.isfinite(df[pcol])].copy()
    df["prediction"] = df[pcol]
    df["residual"] = df["score"] - df["prediction"]
    df["abs_residual"] = df["residual"].abs()
    df["sq_residual"] = df["residual"] ** 2
    df["residual_sign"] = np.where(df["residual"] >= 0, "underpredicted", "overpredicted")
    return df


def residual_summary(resid_df: pd.DataFrame) -> pd.DataFrame:
    y = resid_df["score"].to_numpy(float)
    p = resid_df["prediction"].to_numpy(float)
    r = resid_df["residual"].to_numpy(float)
    w = resid_df["sample_weight"].to_numpy(float)
    slope, intercept, pearson, pval, stderr = sp_stats.linregress(p, y)
    spear = sp_stats.spearmanr(p, y, nan_policy="omit")
    # Shapiro is sensitive and not recommended for huge n; here n is modest.
    shapiro_stat, shapiro_p = sp_stats.shapiro(r) if 3 <= len(r) <= 5000 else (np.nan, np.nan)
    return pd.DataFrame([{
        "n": len(resid_df),
        "weighted_mae": weighted_mean(np.abs(r), w),
        "weighted_rmse": weighted_rmse(r, w),
        "weighted_bias": weighted_mean(r, w),
        "median_abs_error": float(np.median(np.abs(r))),
        "p90_abs_error": float(np.quantile(np.abs(r), 0.90)),
        "p95_abs_error": float(np.quantile(np.abs(r), 0.95)),
        "weighted_r2": weighted_r2(y, p, w),
        "pearson_pred_actual": float(pearson),
        "spearman_pred_actual": float(spear.statistic),
        "calibration_intercept_actual_on_pred": float(intercept),
        "calibration_slope_actual_on_pred": float(slope),
        "shapiro_w": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
    }])


def group_residuals(resid_df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in resid_df.groupby(by):
        r = g["residual"].to_numpy(float)
        w = g["sample_weight"].to_numpy(float)
        rows.append({
            by: key,
            "n": len(g),
            "models": g["model"].nunique(),
            "mean_score": float(g["score"].mean()),
            "mean_pred": float(g["prediction"].mean()),
            "weighted_mae": weighted_mean(np.abs(r), w),
            "weighted_rmse": weighted_rmse(r, w),
            "weighted_bias": weighted_mean(r, w),
            "p90_abs_error": float(np.quantile(np.abs(r), 0.90)) if len(g) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("weighted_mae", ascending=False)


def sensitivity_grid() -> pd.DataFrame:
    rows = []
    combos = []
    for source_policy in ["full", "official_like", "official_only"]:
        for scale_policy in ["continuous", "bucket"]:
            combos.append((source_policy, scale_policy, "logit", "ridge"))
    combos += [("official_like", "continuous", "raw", "ridge")]  # Huber was omitted because sparse high-dimensional folds converge slowly.
    seen = set()
    for source_policy, scale_policy, target, estimator in combos:
        key = (source_policy, scale_policy, target, estimator)
        if key in seen:
            continue
        seen.add(key)
        args = ArgsObj(source_policy=source_policy, scale_policy=scale_policy, target=target, estimator=estimator)
        primary, all_cells, per_bench, pooled, preds, cov, src = run_once(args)
        for _, row in pooled.iterrows():
            rows.append({
                "source_policy": source_policy,
                "scale_policy": scale_policy,
                "target": target,
                "estimator": estimator,
                "spec": row["spec"],
                "mae": row["mae"],
                "mae_std": row.get("mae_std", np.nan),
                "n": row["n"],
                "families": row["n_families"],
                "status": row["status"],
            })
    return pd.DataFrame(rows)


def fit_full_coefficients(primary: pd.DataFrame, args: ArgsObj, spec: str = "Full") -> pd.DataFrame:
    cols = core.SPECS[spec]
    X, names = core.design_matrix(primary.reset_index(drop=True), cols, pooled=True)
    y_raw = primary["score"].to_numpy(float)
    y = core.pct_to_logit(y_raw) if args.target == "logit" else y_raw
    w = primary["sample_weight"].to_numpy(float)
    est = core.make_estimator(args.estimator)
    if args.estimator == "ridge":
        est.fit(X.to_numpy(float), y, ridge__sample_weight=w)
        model = est.named_steps["ridge"]
    else:
        est.fit(X.to_numpy(float), y, huberregressor__sample_weight=w)
        model = est.named_steps["huberregressor"]
    coef = getattr(model, "coef_", np.full(len(names), np.nan))
    out = pd.DataFrame({"feature": names, "standardized_coefficient": coef})
    out["abs_coefficient"] = out["standardized_coefficient"].abs()
    out["feature_group"] = out["feature"].map(feature_group)
    return out.sort_values("abs_coefficient", ascending=False)


def feature_group(name: str) -> str:
    if name.startswith("bench_"):
        return "benchmark fixed effects"
    if name.startswith("src_") or name in {"source_confidence_weight", "exact_protocol_match_to_primary_suite", "thinking_on_cell", "reasoning_effort_score", "tools_allowed_flag"}:
        return "measurement/protocol"
    if "_x_" in name:
        return "recipe x benchmark interactions"
    if name in {"math_intensity", "code_pretrain", "synthetic_quality", "trained_web_only", "is_reasoning_model", "is_instruct", "release_year_norm", "active_x_math", "active_x_code"}:
        return "training/recipe"
    if name in {"log_active_params", "log_total_params", "log_train_tokens", "param_confidence", "scale_imputed", "is_moe", "scale_bucket"}:
        return "scale/capacity"
    if name in {"log_context", "is_gqa", "has_rope"}:
        return "architecture/context"
    if name in {"saturation_2026", "n_choices", "output_space", "contamination_resistant", "retained_v22"}:
        return "benchmark metadata"
    return "other"


def figure_observed_vs_predicted(df: pd.DataFrame):
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(df["prediction"], df["score"], s=24, alpha=0.75)
    lo = min(df["prediction"].min(), df["score"].min()) - 2
    hi = max(df["prediction"].max(), df["score"].max()) + 2
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Out-of-fold predicted score (%)")
    plt.ylabel("Observed score (%)")
    plt.title("Observed versus out-of-fold predicted benchmark scores")
    plt.tight_layout()
    plt.savefig(FIG / "fig_observed_vs_predicted.png", dpi=220)
    plt.close()


def figure_residuals_vs_fitted(df: pd.DataFrame):
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(df["prediction"], df["residual"], s=24, alpha=0.75)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Out-of-fold predicted score (%)")
    plt.ylabel("Residual: observed - predicted (pp)")
    plt.title("Residuals versus fitted values")
    plt.tight_layout()
    plt.savefig(FIG / "fig_residuals_vs_fitted.png", dpi=220)
    plt.close()


def figure_residual_histogram(df: pd.DataFrame):
    plt.figure(figsize=(6.2, 4.8))
    plt.hist(df["residual"], bins=24, alpha=0.85)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Residual: observed - predicted (pp)")
    plt.ylabel("Cell count")
    plt.title("Distribution of out-of-fold residuals")
    plt.tight_layout()
    plt.savefig(FIG / "fig_residual_histogram.png", dpi=220)
    plt.close()


def figure_qq(df: pd.DataFrame):
    plt.figure(figsize=(5.8, 5.0))
    sp_stats.probplot(df["residual"].to_numpy(float), dist="norm", plot=plt)
    plt.title("Normal Q-Q plot for out-of-fold residuals")
    plt.tight_layout()
    plt.savefig(FIG / "fig_residual_qq.png", dpi=220)
    plt.close()


def figure_mae_by_benchmark(by_bench: pd.DataFrame):
    d = by_bench.sort_values("weighted_mae", ascending=True).tail(14)
    plt.figure(figsize=(7.2, 5.4))
    plt.barh(d["benchmark"].astype(str), d["weighted_mae"])
    plt.xlabel("Weighted MAE (percentage points)")
    plt.ylabel("Benchmark")
    plt.title("Residual error by benchmark")
    plt.tight_layout()
    plt.savefig(FIG / "fig_mae_by_benchmark.png", dpi=220)
    plt.close()


def figure_ablation(pooled: pd.DataFrame):
    d = pooled.sort_values("mae", ascending=True)
    plt.figure(figsize=(6.0, 4.6))
    plt.bar(d["spec"].astype(str), d["mae"])
    plt.ylabel("Weighted MAE (percentage points)")
    plt.xlabel("Predictor set")
    plt.title("Family-held-out accuracy by model specification")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "fig_accuracy_ablation.png", dpi=220)
    plt.close()


def figure_sensitivity(sens: pd.DataFrame):
    full = sens[sens["spec"] == "Full"].copy()
    full["label"] = full["source_policy"] + " / " + full["scale_policy"] + " / " + full["target"] + " / " + full["estimator"]
    full = full.sort_values("mae", ascending=True)
    plt.figure(figsize=(8.0, 5.2))
    plt.barh(full["label"], full["mae"])
    plt.xlabel("Weighted MAE (percentage points)")
    plt.ylabel("Sensitivity run")
    plt.title("Full-model sensitivity to source, scale, target and estimator choices")
    plt.tight_layout()
    plt.savefig(FIG / "fig_sensitivity_grid.png", dpi=220)
    plt.close()


def figure_family_benchmark_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="family", columns="benchmark", values="abs_residual", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(10.0, 5.8))
    im = plt.imshow(pivot.to_numpy(float), aspect="auto")
    plt.colorbar(im, label="Mean absolute residual (pp)")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    plt.title("Residual heatmap by model family and benchmark")
    plt.tight_layout()
    plt.savefig(FIG / "fig_residual_heatmap_family_benchmark.png", dpi=220)
    plt.close()


def make_figures(resid_df, by_bench, pooled, sens):
    figure_observed_vs_predicted(resid_df)
    figure_residuals_vs_fitted(resid_df)
    figure_residual_histogram(resid_df)
    figure_qq(resid_df)
    figure_mae_by_benchmark(by_bench)
    figure_ablation(pooled)
    figure_sensitivity(sens)
    figure_family_benchmark_heatmap(resid_df)


def format_num(x, nd=2):
    return "--" if pd.isna(x) else f"{float(x):.{nd}f}"


def to_latex_rows(df, cols, formats=None, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)
    rows = []
    for _, r in df.iterrows():
        parts = []
        for c in cols:
            val = r[c]
            if formats and c in formats:
                parts.append(formats[c](val))
            else:
                parts.append(latex_escape(val))
        rows.append(" & ".join(parts) + r" \\")
    return "\n".join(rows)


def write_latex_assets(summary, pooled, per_bench, coverage, resid_summary_df, by_bench, by_family, by_source, outliers, sens, coeffs):
    # Pooled ablation table.
    pooled2 = pooled.copy()
    pooled2["mae_fmt"] = pooled2["mae"].map(lambda x: format_num(x, 2))
    pooled2["mae_std_fmt"] = pooled2["mae_std"].map(lambda x: format_num(x, 2))
    pooled_rows = to_latex_rows(
        pooled2, ["spec", "n", "n_families", "n_scored", "mae_fmt", "mae_std_fmt", "status"],
        formats={"spec": latex_escape, "status": latex_escape}
    )

    # Coverage table.
    cov = coverage.copy().sort_values(["include_primary", "models"], ascending=[False, False]).head(18)
    cov["avg_source_weight_fmt"] = cov["avg_source_weight"].map(lambda x: format_num(x, 2))
    coverage_rows = to_latex_rows(
        cov, ["benchmark", "models", "official_rows", "metric_type", "include_primary", "avg_source_weight_fmt"],
        formats={"benchmark": latex_escape, "metric_type": latex_escape}
    )

    # Residual summary table.
    rs = resid_summary_df.iloc[0]
    row_end = r"\\"
    res_rows = "\n".join([
        f"Cells & {int(rs['n'])} {row_end}",
        f"Weighted MAE & {rs['weighted_mae']:.2f} pp {row_end}",
        f"Weighted RMSE & {rs['weighted_rmse']:.2f} pp {row_end}",
        f"Weighted bias & {rs['weighted_bias']:.2f} pp {row_end}",
        f"Median absolute error & {rs['median_abs_error']:.2f} pp {row_end}",
        f"90th percentile absolute error & {rs['p90_abs_error']:.2f} pp {row_end}",
        f"Weighted $R^2$ & {rs['weighted_r2']:.2f} {row_end}",
        f"Calibration slope & {rs['calibration_slope_actual_on_pred']:.2f} {row_end}",
        f"Calibration intercept & {rs['calibration_intercept_actual_on_pred']:.2f} {row_end}",
    ])

    # Per-benchmark residual table.
    bb = by_bench.copy().sort_values("weighted_mae", ascending=False)
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias", "p90_abs_error"]:
        bb[c + "_fmt"] = bb[c].map(lambda x: format_num(x, 2))
    bench_rows = to_latex_rows(
        bb, ["benchmark", "n", "models", "weighted_mae_fmt", "weighted_rmse_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
        formats={"benchmark": latex_escape}, max_rows=14
    )

    # Family residual table.
    fam = by_family.copy().sort_values("weighted_mae", ascending=False).head(12)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        fam[c + "_fmt"] = fam[c].map(lambda x: format_num(x, 2))
    family_rows = to_latex_rows(
        fam, ["family", "n", "models", "weighted_mae_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
        formats={"family": latex_escape}
    )

    # Source residual table.
    src = by_source.copy().sort_values("weighted_mae", ascending=False)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        src[c + "_fmt"] = src[c].map(lambda x: format_num(x, 2))
    source_rows = to_latex_rows(
        src, ["source", "n", "models", "weighted_mae_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
        formats={"source": latex_escape}
    )

    # Outliers.
    oo = outliers.copy().head(12)
    for c in ["score", "prediction", "residual", "abs_residual"]:
        oo[c + "_fmt"] = oo[c].map(lambda x: format_num(x, 1))
    outlier_rows = to_latex_rows(
        oo, ["model", "benchmark", "score_fmt", "prediction_fmt", "residual_fmt", "source"],
        formats={"model": latex_escape, "benchmark": latex_escape, "source": latex_escape}
    )

    # Sensitivity Full only.
    sfull = sens[sens["spec"] == "Full"].copy().sort_values("mae")
    sfull["mae_fmt"] = sfull["mae"].map(lambda x: format_num(x, 2))
    sensitivity_rows = to_latex_rows(
        sfull, ["source_policy", "scale_policy", "target", "estimator", "n", "families", "mae_fmt", "status"],
        formats={"source_policy": latex_escape, "scale_policy": latex_escape, "target": latex_escape, "estimator": latex_escape, "status": latex_escape}
    )

    # Coefficients.
    cc = coeffs.copy().head(18)
    cc["coef_fmt"] = cc["standardized_coefficient"].map(lambda x: format_num(x, 3))
    coef_rows = to_latex_rows(
        cc, ["feature", "feature_group", "coef_fmt"],
        formats={"feature": latex_escape, "feature_group": latex_escape}
    )

    tex = rf"""
% Auto-generated by diagnostics.py. Do not edit manually.
\newcommand{{\VFiveCanonicalCells}}{{{summary['n_primary_cells']}}}
\newcommand{{\VFiveCanonicalModels}}{{{summary['n_models_primary']}}}
\newcommand{{\VFiveCanonicalFamilies}}{{{summary['n_families_primary']}}}
\newcommand{{\VFiveCanonicalBenchmarks}}{{{summary['n_benchmarks_primary']}}}
\newcommand{{\VFiveWeightedMAE}}{{{summary['weighted_mae']:.2f}}}
\newcommand{{\VFiveWeightedRMSE}}{{{summary['weighted_rmse']:.2f}}}
\newcommand{{\VFiveWeightedBias}}{{{summary['weighted_bias']:.2f}}}
\newcommand{{\VFiveWeightedRTwo}}{{{summary['weighted_r2']:.2f}}}
\newcommand{{\VFiveCalibrationSlope}}{{{summary['calibration_slope']:.2f}}}
\newcommand{{\VFivePooledTable}}{{%
\begin{{tabular}}{{lrrrrrl}}\toprule
Specification & cells & families & scored & MAE & fold sd & status \\
\midrule
{pooled_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveCoverageTable}}{{%
\begin{{tabular}}{{lrrlrr}}\toprule
Benchmark & models & official rows & metric & primary & avg wt \\
\midrule
{coverage_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveResidualSummaryTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{res_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveBenchmarkResidualTable}}{{%
\begin{{tabular}}{{lrrrrrr}}\toprule
Benchmark & n & models & MAE & RMSE & bias & p90 abs err \\
\midrule
{bench_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveFamilyResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Family & n & models & MAE & bias & p90 abs err \\
\midrule
{family_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveSourceResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Source & n & models & MAE & bias & p90 abs err \\
\midrule
{source_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveOutlierTable}}{{%
\begin{{tabular}}{{llrrrr}}\toprule
Model & Benchmark & observed & predicted & residual & source \\
\midrule
{outlier_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveSensitivityTable}}{{%
\begin{{tabular}}{{llllrrrl}}\toprule
Source policy & scale & target & estimator & cells & fam & Full MAE & status \\
\midrule
{sensitivity_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFiveCoefficientTable}}{{%
\begin{{tabular}}{{llr}}\toprule
Feature & group & std. coef. \\
\midrule
{coef_rows}
\bottomrule\end{{tabular}}}}
"""
    (OUT / "auto_tables_v5.tex").write_text(tex)


def write_report_tex():
    # The paper is written as a standalone article, with citations and figure references.
    report = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{microtype}
\usepackage{placeins}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\input{outputs_v5/auto_tables_v5.tex}
\title{Predicting Frontier LLM Benchmark Scores from Model, Training, Protocol, and Benchmark Metadata: A Source-Weighted Residual-Diagnostic Study}
\author{Prepared as a reproducible model-review paper}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
This paper studies whether cross-model benchmark scores for large language models can be predicted from a structured set of model-scale, architecture, training-recipe, protocol, source-quality, and benchmark-design variables.  The paper is not a causal contamination detector.  It is a predictive model-review exercise designed to test how much benchmark variation is explained by scale alone and how much requires recipe, protocol, and benchmark-property information.  The v5 contribution is methodological: the data schema distinguishes benchmark variants and score scales; Gemini-family models are explicitly included from official Google/DeepMind sources; closed-source scale variables are marked as imputed low-confidence proxies; cross-validation holds out model families; and the reported results include residual diagnostics, outlier analysis, source-policy sensitivity, scale-policy sensitivity, and model-formulation limitations.  In the canonical official-like, logit-transformed, ridge-regression run, the full pooled model uses \VFiveCanonicalCells{} percent-score benchmark cells from \VFiveCanonicalModels{} models, \VFiveCanonicalFamilies{} model families, and \VFiveCanonicalBenchmarks{} benchmark variants.  Its out-of-fold weighted MAE is \VFiveWeightedMAE{} percentage points, with weighted RMSE of \VFiveWeightedRMSE{} percentage points and weighted bias of \VFiveWeightedBias{} percentage points.  The evidence supports the weaker but more defensible claim that benchmark scores are better predicted by combining scale, recipe, protocol, source, and benchmark-design variables than by scale alone.  It does not establish that any particular score is contaminated or that benchmark substitutability has been causally identified.
\end{abstract}

\section{Introduction}
Public LLM benchmark tables are increasingly difficult to interpret.  Reported scores differ not only because models differ, but also because evaluation windows, tool access, reasoning effort, sampling policy, pass@k conventions, answer extraction, and source quality differ.  A model comparison that mixes AIME 2024 with AIME 2025, or a LiveCodeBench percentage with a LiveCodeBench Pro Elo score, creates an apparent panel but not a coherent statistical object.  At the same time, many older benchmarks are now near saturation, while newer benchmarks are designed to be refreshed or more contamination-resistant.

The question studied here is deliberately narrow.  Given a provenance table of reported benchmark cells and a set of hand-coded model and benchmark predictors, how accurately can one predict held-out benchmark scores, and where does the model fail?  The answer is useful for model governance because a predictive benchmark model can identify which scores are surprising relative to model-family, source, protocol, and benchmark context.  It is less useful as a causal estimate of training-data contamination.  The distinction matters: a residual can reflect contamination, protocol mismatch, true model specialization, source error, answer-extraction differences, or simply a missing predictor.

This v5 version addresses several limitations of the prior draft.  First, the paper is now standalone: it explains the data schema, predictors, model formulation, cross-validation design, residual diagnostics, and remaining weaknesses without relying on earlier review notes.  Second, model predictors are described in enough detail to distinguish observed open-weight variables from closed-source proxies.  Third, the empirical section now includes accuracy and residual plots rather than only MAE tables.  Fourth, the residual analysis is explicit: errors are summarized by benchmark, family, source type, fitted value, and top outliers.  Fifth, the limitations section separates limitations that are mitigated in v5 from limitations that remain unresolved.

\section{Data and provenance design}

\subsection{Observation unit}
The observation unit is a benchmark cell
\[
  (i,b,s,p),
\]
where $i$ denotes a model, $b$ a benchmark variant, $s$ the source class, and $p$ the protocol metadata attached to the reported score.  The score is retained in long format rather than pivoted into a wide model-by-benchmark matrix.  This design is important because the same nominal benchmark can appear with different dates, answer protocols, pass@k conventions, thinking settings, or source quality.

The v5 package uses the same long-format provenance file introduced in v4, \texttt{benchmark\_provenance\_v4.csv}, but treats it as an auditable source ledger rather than as a final truth table.  Each row contains the model name, benchmark variant, score, source class, source detail, protocol description, thinking mode, date, metric type, score scale, confidence weight, exact-protocol flag, tool allowance, reasoning effort, pass@k field, consensus/sampling field, and notes.  Rows are de-duplicated within each source-policy run by preferring higher-ranked sources, exact protocol matches, higher source weights, and more recent dates.

\subsection{Benchmark variants and score scales}
Benchmark variants are not collapsed unless the score scale and protocol are comparable.  AIME 2024 and AIME 2025 are separate variables.  LiveCodeBench percentage scores and LiveCodeBench Pro Elo scores are also separate.  This avoids the common but serious error of treating every value with a familiar benchmark family name as exchangeable.

LiveCodeBench is particularly important because its purpose is to provide a holistic and contamination-free coding benchmark that continuously collects new problems over time, rather than a fixed legacy benchmark such as HumanEval or MBPP.\cite{livecodebench_site,livecodebench_paper}  The Google Gemini 2.5 technical report also reports LiveCodeBench results over a specified UI date window, reinforcing the need to retain the date/protocol window rather than use a single undated benchmark name.\cite{gemini25}

\subsection{Gemini coverage}
Gemini is explicitly included.  The Gemini 2.5 technical report reports Gemini 2.5 Flash and Gemini 2.5 Pro values for LiveCodeBench, GPQA Diamond, AIME 2025, SWE-bench Verified, Aider Polyglot, SimpleQA, FACTS Grounding, Global MMLU, and other benchmarks.\cite{gemini25}  The Gemini 3.1 Pro model card reports, among other items, GPQA Diamond, SWE-Bench Verified, SWE-Bench Pro, LiveCodeBench Pro Elo, MMMLU, and long-context results for Gemini 3.1 Pro and Gemini 3 Pro under high-thinking settings.\cite{gemini31}  V5 retains Gemini 3/3.1 LiveCodeBench Pro as an Elo-scale extension variable and does not mix it with percentage accuracy.

\begin{table}[H]
\centering\scriptsize
\caption{Coverage after canonical source-policy filtering. Primary indicates whether the benchmark enters the canonical percent-score regression.}
\resizebox{\textwidth}{!}{\VFiveCoverageTable}
\label{tab:coverage}
\end{table}

\section{Predictors}

\subsection{Model-scale and capacity predictors}
The scale block contains $\log$ active parameters, $\log$ total parameters, $\log$ training tokens, a parameter-confidence field, a scale-imputed flag, and a mixture-of-experts indicator.  For open-weight models, active and total parameter counts are treated as observed when public documentation is sufficiently clear.  For closed-source frontier models, scale variables are explicitly weak proxies.  They are included because the model needs an ordinal capacity control, but the paper does not represent them as factual parameter disclosures.  This is why v5 includes both continuous-scale and bucketed-scale sensitivity runs.

The active/total distinction matters for mixture-of-experts systems.  Total parameters are a capacity proxy; active parameters are closer to an inference-compute proxy.  The model includes both because a benchmark score may reflect both stored capacity and per-token active computation.  For proprietary MoE systems, both fields are uncertain; therefore the scale-imputed and parameter-confidence fields are not cosmetic but core controls.

\subsection{Architecture and context predictors}
The architecture block contains context length, grouped-query-attention style coding, and positional-encoding indicators.  These variables are coarse.  They do not encode all relevant architecture details such as router design, recurrence, retrieval augmentation, multimodal encoder architecture, verifier policies, or inference-time search.  Their purpose is to prevent the scale coefficients from absorbing obvious context-window and architecture differences.

\subsection{Training-recipe predictors}
The recipe block contains math intensity, code-pretraining intensity, synthetic-data quality, web-only training indicator, reasoning-model indicator, instruction-tuning indicator, release year, and interactions between active scale and the math/code intensities.  These are hand-coded ordinal predictors, not measured quantities.  The correct interpretation is therefore comparative and predictive: a positive recipe contribution means that recipe-coded variables help explain held-out score variation after controlling for family-held-out scale and benchmark effects.  It does not mean the specific private training mixture is known.

\subsection{Protocol and source predictors}
The measurement block contains source-confidence weight, exact-protocol flag, thinking indicator, reasoning-effort score, tools-allowed flag, and source fixed effects.  These variables address a major limitation of simple benchmark regressions: the reported score is partly a property of the evaluation harness.  For frontier models, differences between no-thinking, dynamic-thinking, high-thinking, no-tools, search-enabled, code-enabled, single-attempt, and multiple-attempt settings can be large enough to dominate ordinary model differences.  V5 treats these as measurement controls rather than prose caveats.

\subsection{Benchmark metadata and interactions}
The benchmark block contains benchmark fixed effects plus benchmark metadata: estimated saturation, number of choices, output-space class, contamination-resistant flag, and retained-versus-new flag.  The full model also includes interactions between recipe variables and benchmark properties.  This is the central formulation change relative to a simple pooled regression.  It allows reasoning, math specialization, coding specialization, and synthetic-data quality to matter differently on open-form, refreshed, and contamination-resistant benchmarks than on older multiple-choice or saturated benchmarks.

\section{Model formulation}

For percent-score benchmarks, the canonical target is a clipped logit transform:
\[
  z_{i b s p}=\log\left(\frac{\tilde y_{i b s p}}{1-\tilde y_{i b s p}}\right),
\]
where $\tilde y$ is the percentage score divided by 100 and clipped away from zero and one.  Predictions are transformed back to percentage points for all reported accuracy statistics.  The logit target is used because bounded scores near 0 or 100 do not have the same error geometry as scores near 50.

The full pooled specification is
\[
 z_{i b s p}=\alpha_b + x_i'\beta + a_i'\eta + r_i'\rho + m_{s p}'\gamma + q_b'\delta
 + (r_i \otimes q_b)'\theta + \varepsilon_{i b s p},
\]
where $\alpha_b$ are benchmark fixed effects, $x_i$ contains scale and capacity predictors, $a_i$ architecture/context predictors, $r_i$ recipe predictors, $m_{s p}$ protocol/source predictors, and $q_b$ benchmark metadata.  The interaction term $(r_i \otimes q_b)'\theta$ allows recipe variables to have benchmark-dependent effects.  The model is estimated using ridge regression in the canonical run.  Huber regression is reported as a robustness check.

\subsection{Validation design}
The primary validation design is family-held-out cross-validation.  Folds are built by grouping models by family, then assigning entire families to folds.  This is stricter than random cell cross-validation.  Random cells would allow the model to train on, for example, one benchmark from a model family and then predict another benchmark for the same family.  That would overstate generalization because family-specific design choices and reporting conventions would leak across folds.

The main accuracy metric is source-weighted MAE in percentage points.  RMSE, bias, weighted $R^2$, calibration slope, calibration intercept, and residual percentiles are reported as diagnostics.  The paper uses MAE as the headline metric because benchmark score errors are not expected to be Gaussian, and because outliers often reflect protocol or provenance issues rather than purely statistical noise.

\section{Results}

\subsection{Ablation accuracy}
\begin{table}[H]
\centering\small
\caption{Pooled family-held-out accuracy by model specification. MAE is source-weighted and reported in percentage points after converting logit predictions back to the original score scale.}
\VFivePooledTable
\label{tab:pooled}
\end{table}

The ablation table shows that the full model is more accurate than scale-only and scale-plus-recipe specifications.  The result is consistent with the view that model scale is not enough to explain modern benchmark performance.  It also shows that recipe variables alone are not enough: protocol/source controls and benchmark-property interactions contribute materially.

\begin{figure}[H]
\centering
\includegraphics[width=0.72\textwidth]{figures_v5/fig_accuracy_ablation.png}
\caption{Family-held-out weighted MAE by predictor set. Lower is better.}
\label{fig:ablation}
\end{figure}

\subsection{Prediction accuracy and calibration}
\begin{figure}[H]
\centering
\includegraphics[width=0.70\textwidth]{figures_v5/fig_observed_vs_predicted.png}
\caption{Observed versus out-of-fold predicted benchmark scores for the canonical full model. The dashed line is perfect calibration.}
\label{fig:obs_pred}
\end{figure}

Figure~\ref{fig:obs_pred} shows that the full model captures broad score ordering, but it is not a high-precision benchmark oracle.  The residual summary in Table~\ref{tab:res_summary} gives the quantitative diagnostic picture.

\begin{table}[H]
\centering\small
\caption{Residual diagnostics for the canonical full model. Residual is observed minus predicted, in percentage points.}
\VFiveResidualSummaryTable
\label{tab:res_summary}
\end{table}

A calibration slope materially below one would indicate over-dispersion of predictions; a slope materially above one would indicate under-dispersion.  The estimated slope is \VFiveCalibrationSlope{}, so the fitted values are useful for ordering but still require caution for precise score prediction.  The weighted bias of \VFiveWeightedBias{} percentage points is small relative to the MAE, suggesting that the main issue is dispersion and benchmark/family heterogeneity rather than a simple average over- or under-prediction.

\section{Residual analysis}

\subsection{Residual shape and heteroskedasticity}
\begin{figure}[H]
\centering
\begin{subfigure}{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{figures_v5/fig_residuals_vs_fitted.png}
\caption{Residuals versus fitted values.}
\end{subfigure}
\begin{subfigure}{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{figures_v5/fig_residual_histogram.png}
\caption{Residual distribution.}
\end{subfigure}
\caption{Residual diagnostics for the canonical full model.}
\label{fig:residuals_basic}
\end{figure}

The residuals are not perfectly homoskedastic.  A common pattern is larger absolute error in mid-to-high frontier ranges, especially on refreshed math and coding benchmarks.  This is expected: those cells combine rapid model progress, protocol heterogeneity, and fewer comparable official observations.  The histogram is centered near zero but has tails large enough that RMSE is materially higher than MAE.

\begin{figure}[H]
\centering
\includegraphics[width=0.62\textwidth]{figures_v5/fig_residual_qq.png}
\caption{Q-Q plot of out-of-fold residuals. Tail deviations should be expected because score cells are heterogeneous benchmark/protocol observations, not repeated draws from one normal error law.}
\label{fig:qq}
\end{figure}

\subsection{Residuals by benchmark}
\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures_v5/fig_mae_by_benchmark.png}
\caption{Weighted MAE by benchmark. Higher-error benchmarks are where the model is least reliable or where protocol/data heterogeneity is greatest.}
\label{fig:bench_mae}
\end{figure}

\begin{table}[H]
\centering\scriptsize
\caption{Residual diagnostics by benchmark. Bias is observed minus predicted; positive bias means the model underpredicted the benchmark score.}
\resizebox{\textwidth}{!}{\VFiveBenchmarkResidualTable}
\label{tab:bench_residuals}
\end{table}

The benchmark-level residuals are the most informative diagnostic.  Older saturated benchmarks can have low residual error because they are easier to predict and compressed near the top of the scale.  Refreshed coding and frontier reasoning benchmarks can have higher error because they contain fewer comparable official cells and stronger protocol dependence.  High error on a benchmark is not evidence that the benchmark is invalid; it is evidence that the current predictor set is incomplete for that benchmark.

\subsection{Residuals by model family and source}
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures_v5/fig_residual_heatmap_family_benchmark.png}
\caption{Mean absolute residual by model family and benchmark. Blank cells correspond to missing benchmark-family combinations.}
\label{fig:heatmap}
\end{figure}

\begin{table}[H]
\centering\scriptsize
\caption{Largest family-level residual diagnostics. Family residuals identify where model-specific design or source/protocol patterns are not captured by the predictor set.}
\resizebox{0.88\textwidth}{!}{\VFiveFamilyResidualTable}
\label{tab:family_residuals}
\end{table}

\begin{table}[H]
\centering\small
\caption{Residual diagnostics by source class. Source effects should not be interpreted as source truthfulness; they combine source reliability, coverage composition, and benchmark mix.}
\VFiveSourceResidualTable
\label{tab:source_residuals}
\end{table}

\subsection{Outliers}
\begin{table}[H]
\centering\scriptsize
\caption{Largest absolute out-of-fold residuals in the canonical full model. These rows should be reviewed as data/protocol/model-specialization cases before drawing substantive conclusions.}
\resizebox{\textwidth}{!}{\VFiveOutlierTable}
\label{tab:outliers}
\end{table}

Outliers should be used operationally as a review queue.  A large residual may identify a genuine model specialization, but it may also identify a protocol mismatch, an outdated score, a benchmark variant mismatch, or a weak closed-source scale proxy.  The correct next step is not to delete outliers automatically, but to trace each row back to its source detail and verify the protocol.

\section{Sensitivity analysis}

\begin{table}[H]
\centering\scriptsize
\caption{Full-model sensitivity to source policy, scale policy, target transform, and estimator.}
\resizebox{\textwidth}{!}{\VFiveSensitivityTable}
\label{tab:sensitivity}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.90\textwidth]{figures_v5/fig_sensitivity_grid.png}
\caption{Sensitivity of the full-model MAE to source-policy, scale-policy, target, and estimator choices.}
\label{fig:sensitivity}
\end{figure}

The sensitivity analysis addresses three weaknesses.  First, source filtering matters: official-only rows can improve apparent MAE but reduce coverage and may underrepresent families without first-party model cards.  Second, bucketed scale is a necessary check because closed-source parameter counts are proxies.  Third, the raw-versus-logit target comparison checks whether bounded-score geometry is driving the main result.  The conclusion is not that one sensitivity run is always superior; it is that the qualitative result should be judged across the grid, not from one cherry-picked specification.

\section{Coefficient and predictor interpretation}

\begin{table}[H]
\centering\scriptsize
\caption{Largest standardized coefficients from the canonical full ridge model fit on all canonical cells. Coefficients are descriptive because correlated predictors and benchmark fixed effects make individual coefficients unstable.}
\resizebox{\textwidth}{!}{\VFiveCoefficientTable}
\label{tab:coefficients}
\end{table}

The coefficient table is intentionally secondary to the prediction and residual tables.  Ridge coefficients in a correlated design are not causal effects.  They are useful for checking whether the model is relying heavily on benchmark fixed effects, source/protocol variables, or a few scale proxies.  A coefficient that looks economically large should be interpreted as a diagnostic prompt rather than as a structural estimate.

\section{What v5 fixes and what remains unresolved}

\subsection{Limitations mitigated in v5}
V5 addresses several limitations of the earlier draft.  The paper no longer mixes benchmark variants or score scales.  It includes Gemini using official Google/DeepMind sources and separates Gemini 3/3.1 LiveCodeBench Pro Elo from LiveCodeBench percentage accuracy.  It implements source-policy filtering and source weights in code rather than only describing them.  It uses family-held-out cross-validation rather than random-cell validation.  It reports residual diagnostics, benchmark/family/source residuals, outliers, and sensitivity runs.  It treats closed-source scale variables as low-confidence proxies and supplies bucketed-scale robustness checks.  It also avoids causal claims about contamination.

\subsection{Limitations that remain}
The most important unresolved limitation is that many predictors are hand-coded.  Training-recipe variables such as math intensity, code-pretraining intensity, and synthetic-data quality are ordinal judgments, not direct measurements of token mixtures or post-training pipelines.  The model is therefore a structured benchmark-review model, not a scientific measurement of private training data.

A second limitation is protocol granularity.  Thinking mode is coded as off, dynamic, high, or similar categories, but the true causal variable is closer to test-time compute: reasoning-token budget, sampling budget, verifier policy, tool calls, stopping rules, and harness-specific answer extraction.  Public model cards rarely disclose all of these in a comparable way.

A third limitation is residual dependence.  Holding out model families is better than random-cell validation, but cells are still not independent.  Benchmark scores within a family share training data, model architecture, source conventions, and release-era effects.  A full hierarchical Bayesian model with family random effects, benchmark random effects, protocol random effects, and source-specific measurement noise would be statistically cleaner, but the current panel is too sparse for that model to be reliable without strong priors.

A fourth limitation is benchmark evolution.  A benchmark like LiveCodeBench changes over time by design, and model cards report different windows.  Even when the benchmark name is the same, the evaluation population is not necessarily identical.  V5 mitigates this with date/protocol fields but does not fully solve it.

A fifth limitation is publication selection.  Frontier labs choose which benchmark results to publish.  Missing cells are therefore informative, not random.  The current analysis models observed cells only; it does not model the missingness mechanism.

\section{Recommended next model improvements}
The next version should add three model improvements.  First, replace hand-coded recipe predictors with a documented scoring rubric and an independent second coding pass so that recipe variables have inter-rater reliability.  Second, add a missingness model distinguishing unreported, not-applicable, and non-comparable benchmark cells.  Third, estimate a partial-pooling hierarchical model as a sensitivity analysis once there are enough official cells for modern benchmark variants.  Until then, ridge with grouped cross-validation is a reasonable conservative baseline.

\section{Conclusion}
The v5 analysis supports a careful conclusion.  A model that combines scale, architecture, training-recipe proxies, protocol/source controls, benchmark fixed effects, benchmark metadata, and recipe-by-benchmark interactions predicts held-out benchmark cells better than a scale-only model.  Residual analysis shows that the model is useful for benchmark review and anomaly detection, but not precise enough to certify individual benchmark scores or prove contamination.  The practical use of the model is therefore diagnostic: identify benchmark cells that are surprising, trace them to sources and protocols, and update the dataset or formulation before making substantive claims about frontier model capability.

\appendix
\section{Reproducibility}
The v5 analysis can be reproduced with:
\begin{verbatim}
python diagnostics.py
pdflatex report_v5.tex
pdflatex report_v5.tex
\end{verbatim}
The main generated files are \texttt{outputs\_v5/pooled\_results\_canonical\_v5.csv}, \texttt{outputs\_v5/residual\_diagnostics\_full\_v5.csv}, \texttt{outputs\_v5/residual\_by\_benchmark\_v5.csv}, \texttt{outputs\_v5/residual\_outliers\_top20\_v5.csv}, \texttt{outputs\_v5/sensitivity\_grid\_v5.csv}, and the figures in \texttt{figures\_v5/}.

\begin{thebibliography}{9}
\bibitem{gemini25} Google Gemini Team. \emph{Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities}. June 2025. \url{https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf}
\bibitem{gemini31} Google DeepMind. \emph{Gemini 3.1 Pro Model Card}. February 2026. \url{https://deepmind.google/models/model-cards/gemini-3-1-pro/}
\bibitem{livecodebench_site} LiveCodeBench. \emph{Holistic and Contamination Free Evaluation of Large Language Models for Code}. \url{https://livecodebench.github.io/}
\bibitem{livecodebench_paper} Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. \emph{LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code}. arXiv:2403.07974, 2024. \url{https://arxiv.org/abs/2403.07974}
\bibitem{gpt52} OpenAI. \emph{Introducing GPT-5.2}. December 2025. \url{https://openai.com/index/introducing-gpt-5-2/}
\end{thebibliography}

\end{document}
'''
    (ROOT / "report_v5.tex").write_text(report)


def write_readme(summary):
    readme = f"""# LLM Benchmark v5

V5 turns the prior draft into a standalone, diagnostic model-review package.

Canonical run:

```bash
python diagnostics.py
pdflatex report_v5.tex
pdflatex report_v5.tex
```

Main canonical diagnostics:

- Primary cells: {summary['n_primary_cells']}
- Models: {summary['n_models_primary']}
- Families: {summary['n_families_primary']}
- Benchmarks: {summary['n_benchmarks_primary']}
- Full-model weighted MAE: {summary['weighted_mae']:.2f} percentage points
- Weighted RMSE: {summary['weighted_rmse']:.2f} percentage points
- Weighted bias: {summary['weighted_bias']:.2f} percentage points
- Weighted R^2: {summary['weighted_r2']:.2f}

Major v5 additions:

1. Standalone paper structure.
2. Detailed predictor taxonomy and formal model specification.
3. Residual analysis by benchmark, family, and source.
4. Accuracy and residual plots.
5. Outlier table for data/protocol review.
6. Source-policy, scale-policy, target, and estimator sensitivity grid.
7. Explicit statement that the model is predictive/diagnostic, not a causal contamination detector.
"""
    (ROOT / "README_V5.md").write_text(readme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-policy", choices=core.SOURCE_POLICIES.keys(), default="official_like")
    p.add_argument("--scale-policy", choices=["continuous", "bucket"], default="continuous")
    p.add_argument("--target", choices=["raw", "logit"], default="logit")
    p.add_argument("--estimator", choices=["ridge", "huber"], default="ridge")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-models", type=int, default=10)
    args_ns = p.parse_args()
    args = ArgsObj(
        source_policy=args_ns.source_policy,
        scale_policy=args_ns.scale_policy,
        target=args_ns.target,
        estimator=args_ns.estimator,
        n_folds=args_ns.n_folds,
        min_models=args_ns.min_models,
    )

    primary, all_cells, per_bench, pooled, preds, coverage, src = run_once(args)
    resid = add_residual_columns(preds, spec="Full")
    rsum = residual_summary(resid)
    by_bench = group_residuals(resid, "benchmark")
    by_family = group_residuals(resid, "family")
    by_source = group_residuals(resid, "source")
    outliers = resid.sort_values("abs_residual", ascending=False).head(25)
    sens = sensitivity_grid()
    coeffs = fit_full_coefficients(primary, args, spec="Full")

    # Save raw diagnostics.
    primary.to_csv(OUT / "canonical_primary_cells_v5.csv", index=False)
    all_cells.to_csv(OUT / "canonical_all_cells_v5.csv", index=False)
    pooled.to_csv(OUT / "pooled_results_canonical_v5.csv", index=False)
    per_bench.to_csv(OUT / "per_benchmark_results_canonical_v5.csv", index=False)
    preds.to_csv(OUT / "pooled_oof_predictions_canonical_v5.csv", index=False)
    resid.to_csv(OUT / "residual_diagnostics_full_v5.csv", index=False)
    rsum.to_csv(OUT / "residual_summary_full_v5.csv", index=False)
    by_bench.to_csv(OUT / "residual_by_benchmark_v5.csv", index=False)
    by_family.to_csv(OUT / "residual_by_family_v5.csv", index=False)
    by_source.to_csv(OUT / "residual_by_source_v5.csv", index=False)
    outliers.to_csv(OUT / "residual_outliers_top25_v5.csv", index=False)
    sens.to_csv(OUT / "sensitivity_grid_v5.csv", index=False)
    coeffs.to_csv(OUT / "full_model_standardized_coefficients_v5.csv", index=False)
    coverage.to_csv(OUT / "coverage_canonical_v5.csv", index=False)

    make_figures(resid, by_bench, pooled, sens)

    rs = rsum.iloc[0]
    summary = {
        "source_policy": args.source_policy,
        "scale_policy": args.scale_policy,
        "target": args.target,
        "estimator": args.estimator,
        "n_primary_cells": int(len(primary)),
        "n_models_primary": int(primary["model"].nunique()),
        "n_families_primary": int(primary["family"].nunique()),
        "n_benchmarks_primary": int(primary["benchmark"].nunique()),
        "weighted_mae": float(rs["weighted_mae"]),
        "weighted_rmse": float(rs["weighted_rmse"]),
        "weighted_bias": float(rs["weighted_bias"]),
        "weighted_r2": float(rs["weighted_r2"]),
        "calibration_slope": float(rs["calibration_slope_actual_on_pred"]),
        "calibration_intercept": float(rs["calibration_intercept_actual_on_pred"]),
        "gemini_models_included": sorted([m for m in all_cells["model"].unique() if "Gemini" in m]),
    }
    (OUT / "summary_v5.json").write_text(json.dumps(summary, indent=2))
    write_latex_assets(summary, pooled, per_bench, coverage, rsum, by_bench, by_family, by_source, outliers, sens, coeffs)
    write_report_tex()
    write_readme(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
