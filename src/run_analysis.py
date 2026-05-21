#!/usr/bin/env python3
"""
run_analysis.py - evaluation-protocol-adjusted benchmark predictability study.

This release extends V5 in the direction requested for a publication-quality standalone paper:
  * clearer terminology: evaluation-protocol-adjusted rather than protocol-aware;
  * leave-one-family-out validation;
  * leave-one-benchmark-out validation using a no-benchmark-fixed-effect design;
  * temporal holdout validation for post-cutoff model releases;
  * closed-source scale / training-token perturbation stress test;
  * missingness diagnostics for non-random benchmark reporting;
  * additional residual plots by release era and protocol covariates; and
  * auto-generated LaTeX tables and a standalone paper.

The code intentionally keeps the v4/v5 data structure so the version history remains auditable.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analysis_core as core
import diagnostics as v5
from benchmark_suite import BENCHMARK_SUITE_V4, PRIMARY_BENCHMARKS_V4
from model_features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


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


def format_num(x, nd=2):
    return "--" if pd.isna(x) else f"{float(x):.{nd}f}"


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def weighted_mae(y, p, w):
    return weighted_mean(np.abs(np.asarray(y, float) - np.asarray(p, float)), w)


def weighted_rmse(y, p, w):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    return float(np.sqrt(np.average((y[m] - p[m]) ** 2, weights=w[m]))) if m.any() else np.nan


def weighted_r2(y, p, w):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan
    ybar = np.average(y[m], weights=w[m])
    sse = np.sum(w[m] * (y[m] - p[m]) ** 2)
    sst = np.sum(w[m] * (y[m] - ybar) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def design_matrix_v6(
    df: pd.DataFrame,
    spec_cols: list[str],
    include_benchmark_fe: bool = True,
    include_source_fe: bool = True,
    include_benchmark_meta: bool = True,
    include_interactions: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a design matrix that can be aligned across train/test splits."""
    X = df[spec_cols].copy()
    if include_benchmark_meta:
        for c in ["saturation_2026", "n_choices", "output_space", "contamination_resistant", "retained_v22"]:
            if c in df:
                vals = df[c]
                X[c] = vals.fillna(vals.median() if vals.notna().any() else 0.0)
    if include_interactions:
        for r in ["is_reasoning_model", "math_intensity", "code_pretrain", "synthetic_quality"]:
            if r in df:
                X[f"{r}_x_contres"] = df[r].astype(float) * df["contamination_resistant"].astype(float)
                X[f"{r}_x_outputspace"] = df[r].astype(float) * df["output_space"].fillna(0).astype(float)
    pieces = [X]
    if include_benchmark_fe:
        pieces.append(pd.get_dummies(df["benchmark"], prefix="bench", dtype=float))
    if include_source_fe:
        pieces.append(pd.get_dummies(df["source"], prefix="src", dtype=float))
    X = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float), list(X.columns)


def align_design(train_X: pd.DataFrame, test_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = list(train_X.columns)
    test_aligned = test_X.reindex(columns=cols, fill_value=0.0)
    return train_X.to_numpy(float), test_aligned.to_numpy(float)


def fit_train_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    args: ArgsObj,
    spec_cols: list[str] | None = None,
    include_benchmark_fe: bool = True,
    include_source_fe: bool = True,
    include_benchmark_meta: bool = True,
    include_interactions: bool = True,
) -> np.ndarray:
    spec_cols = spec_cols or core.SPECS["Full"]
    if len(train) < 8 or len(test) == 0:
        return np.full(len(test), np.nan)
    Xtr, _ = design_matrix_v6(train, spec_cols, include_benchmark_fe, include_source_fe, include_benchmark_meta, include_interactions)
    Xte, _ = design_matrix_v6(test, spec_cols, include_benchmark_fe, include_source_fe, include_benchmark_meta, include_interactions)
    Xtrv, Xtev = align_design(Xtr, Xte)
    y_raw = train["score"].to_numpy(float)
    y = core.pct_to_logit(y_raw) if args.target == "logit" else y_raw
    w = train["sample_weight"].to_numpy(float)
    est = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
    try:
        est.fit(Xtrv, y, ridge__sample_weight=w)
    except TypeError:
        est.fit(Xtrv, y)
    pred_fit = est.predict(Xtev)
    return core.logit_to_pct(pred_fit) if args.target == "logit" else pred_fit


def validation_metrics(name: str, df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df["score"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    ok = np.isfinite(pred)
    return {
        "validation": name,
        "n": int(ok.sum()),
        "models": int(df.loc[ok, "model"].nunique()) if ok.any() else 0,
        "families": int(df.loc[ok, "family"].nunique()) if ok.any() else 0,
        "benchmarks": int(df.loc[ok, "benchmark"].nunique()) if ok.any() else 0,
        "weighted_mae": weighted_mae(y[ok], pred[ok], w[ok]) if ok.any() else np.nan,
        "weighted_rmse": weighted_rmse(y[ok], pred[ok], w[ok]) if ok.any() else np.nan,
        "weighted_bias": weighted_mean(y[ok] - pred[ok], w[ok]) if ok.any() else np.nan,
        "weighted_r2": weighted_r2(y[ok], pred[ok], w[ok]) if ok.any() else np.nan,
    }


def leave_one_family(primary: pd.DataFrame, args: ArgsObj) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    for fam in sorted(primary["family"].unique()):
        test = primary[primary["family"] == fam].copy()
        train = primary[primary["family"] != fam].copy()
        pred = fit_train_predict(train, test, args, include_benchmark_fe=True)
        out = test[["model", "family", "benchmark", "score", "source", "sample_weight"]].copy()
        out["prediction"] = pred
        out["residual"] = out["score"] - out["prediction"]
        preds.append(out)
        m = validation_metrics(f"leave-family:{fam}", test, pred)
        m["heldout_family"] = fam
        rows.append(m)
    return pd.DataFrame(rows).sort_values("weighted_mae", ascending=False), pd.concat(preds, ignore_index=True)


def leave_one_benchmark(primary: pd.DataFrame, args: ArgsObj) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one benchmark out. Benchmark fixed effects are intentionally disabled.

    This asks a harder question than the canonical cell interpolation task: can the
    observed metadata predict a benchmark that was not in the training panel at all?
    """
    rows = []
    preds = []
    for bench in sorted(primary["benchmark"].unique()):
        test = primary[primary["benchmark"] == bench].copy()
        train = primary[primary["benchmark"] != bench].copy()
        pred = fit_train_predict(train, test, args, include_benchmark_fe=False)
        out = test[["model", "family", "benchmark", "score", "source", "sample_weight"]].copy()
        out["prediction"] = pred
        out["residual"] = out["score"] - out["prediction"]
        preds.append(out)
        m = validation_metrics(f"leave-benchmark:{bench}", test, pred)
        m["heldout_benchmark"] = bench
        rows.append(m)
    return pd.DataFrame(rows).sort_values("weighted_mae", ascending=False), pd.concat(preds, ignore_index=True)


def temporal_holdouts(primary: pd.DataFrame, args: ArgsObj) -> pd.DataFrame:
    rows = []
    for cutoff in [2.75, 2.95, 3.08, 3.25]:
        train = primary[primary["release_year_norm"] <= cutoff].copy()
        test = primary[primary["release_year_norm"] > cutoff].copy()
        if train["family"].nunique() < 4 or test["family"].nunique() < 2 or len(test) < 15:
            continue
        pred = fit_train_predict(train, test, args, include_benchmark_fe=True)
        m = validation_metrics(f"time-split:<=2022+{cutoff:.2f}", test, pred)
        m["cutoff_release_year_norm"] = cutoff
        m["train_cells"] = len(train)
        m["test_cells"] = len(test)
        m["train_models"] = train["model"].nunique()
        m["test_models"] = test["model"].nunique()
        m["cutoff_calendar_year"] = 2022 + cutoff
        rows.append(m)
    return pd.DataFrame(rows).sort_values("cutoff_release_year_norm")


def parameter_uncertainty(primary: pd.DataFrame, args: ArgsObj, n_iter: int = 60, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_iter):
        df = primary.copy()
        # Larger stress for closed-source and imputed scale rows; smaller for open models.
        conf = df["param_confidence"].astype(float).clip(0.05, 1.0)
        imputed = df["scale_imputed"].astype(float).clip(0, 1)
        closed = df["is_closed_source"].astype(float).clip(0, 1)
        sigma_params = 0.04 + 0.18 * imputed + 0.10 * closed + 0.12 * (1.0 - conf)
        sigma_tokens = 0.08 + 0.20 * imputed + 0.15 * closed + 0.10 * (1.0 - conf)
        for col in ["log_active_params", "log_total_params"]:
            df[col] = df[col].astype(float) + rng.normal(0.0, sigma_params.to_numpy(float))
        df["log_train_tokens"] = df["log_train_tokens"].astype(float) + rng.normal(0.0, sigma_tokens.to_numpy(float))
        pooled, preds = core.pooled_results(df, args)
        row = pooled[pooled["spec"] == "Full"].iloc[0].to_dict()
        row["iteration"] = i
        rows.append(row)
    out = pd.DataFrame(rows)
    return out



def bootstrap_oof_mae(preds: pd.DataFrame, n_iter: int = 1000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap on out-of-fold prediction rows.

    Rows are resampled as intact observation pairs so the Full-versus-Scale
    comparison uses the same cells in each bootstrap draw. Metrics remain
    source-weighted within each draw. This is a descriptive uncertainty band for
    the supplied data snapshot, not a population-level causal interval.
    """
    needed = ["score", "sample_weight", "pred_Full", "pred_Scale"]
    df = preds.dropna(subset=needed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    n = len(df)
    y = df["score"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    pf = df["pred_Full"].to_numpy(float)
    ps = df["pred_Scale"].to_numpy(float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        full = weighted_mae(y[idx], pf[idx], w[idx])
        scale = weighted_mae(y[idx], ps[idx], w[idx])
        rows.append({
            "iteration": i,
            "full_mae": full,
            "scale_mae": scale,
            "scale_minus_full_mae": scale - full,
        })
    boot = pd.DataFrame(rows)
    def q(col, prob):
        return float(boot[col].quantile(prob))
    summary = pd.DataFrame([{
        "iterations": int(n_iter),
        "cells": int(n),
        "full_mae_point": weighted_mae(y, pf, w),
        "full_mae_ci_low": q("full_mae", 0.025),
        "full_mae_ci_high": q("full_mae", 0.975),
        "scale_mae_point": weighted_mae(y, ps, w),
        "scale_mae_ci_low": q("scale_mae", 0.025),
        "scale_mae_ci_high": q("scale_mae", 0.975),
        "improvement_point": weighted_mae(y, ps, w) - weighted_mae(y, pf, w),
        "improvement_ci_low": q("scale_minus_full_mae", 0.025),
        "improvement_ci_high": q("scale_minus_full_mae", 0.975),
        "p_improvement_le_zero": float((boot["scale_minus_full_mae"] <= 0).mean()),
    }])
    return boot, summary


def ordinal_recipe_sensitivity(primary: pd.DataFrame, args: ArgsObj, n_iter: int = 60, seed: int = SEED + 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perturb hand-coded ordinal recipe variables by one step.

    The variables math_intensity, code_pretrain, and synthetic_quality are
    intentionally coarse 0--2 codings. To assess whether the headline result is
    an artifact of exact hand labels, this routine perturbs each model-level
    coding by -1, 0, or +1, clips to [0,2], recomputes active interactions, and
    re-runs the canonical pooled Full specification.
    """
    rng = np.random.default_rng(seed)
    recipe_cols = ["math_intensity", "code_pretrain", "synthetic_quality"]
    rows = []
    models = sorted(primary["model"].unique())
    for i in range(n_iter):
        df = primary.copy()
        for col in recipe_cols:
            shifts = {m: int(rng.choice([-1, 0, 1])) for m in models}
            df[col] = df.apply(lambda r: float(np.clip(r[col] + shifts[r["model"]], 0, 2)), axis=1)
        df["active_x_math"] = df["log_active_params"] * df["math_intensity"]
        df["active_x_code"] = df["log_active_params"] * df["code_pretrain"]
        pooled, _ = core.pooled_results(df, args)
        full = pooled[pooled["spec"] == "Full"].iloc[0].to_dict()
        rows.append({
            "iteration": i,
            "mae": float(full["mae"]),
            "mae_std": float(full.get("mae_std", np.nan)),
            "status": full.get("status", "unknown"),
            "n": int(full.get("n", len(df))),
            "n_families": int(full.get("n_families", df["family"].nunique())),
        })
    sens = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "iterations": int(len(sens)),
        "mae_mean": float(sens["mae"].mean()),
        "mae_std": float(sens["mae"].std()),
        "mae_p05": float(sens["mae"].quantile(0.05)),
        "mae_p95": float(sens["mae"].quantile(0.95)),
        "mae_min": float(sens["mae"].min()),
        "mae_max": float(sens["mae"].max()),
    }])
    return sens, summary


def write_feature_coding_table() -> pd.DataFrame:
    """Write the model-level coding table used by the analysis."""
    cols = [
        "family", "active_params_B", "total_params_B", "training_tokens_B", "param_confidence", "scale_imputed",
        "is_moe", "is_reasoning_model", "is_instruct", "math_intensity", "code_pretrain", "synthetic_quality",
        "trained_web_only", "is_closed_source", "release_year", "scale_bucket",
    ]
    rows = []
    for model, info in sorted(MODEL_FEATURES.items()):
        row = {"model": model}
        for c in cols:
            row[c] = info.get(c, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "model_feature_coding.csv", index=False)
    return out


def missingness_diagnostics(all_cells: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = sorted([m for m in MODEL_FEATURES if m in all_cells["model"].unique()])
    benches = [b for b in PRIMARY_BENCHMARKS_V4 if BENCHMARK_SUITE_V4.get(b, {}).get("include_primary", False)]
    observed = set(zip(primary["model"], primary["benchmark"]))
    feats = core.feature_rows("continuous")
    rows = []
    for m in models:
        f = feats[feats["model"] == m].iloc[0].to_dict()
        for b in benches:
            meta = BENCHMARK_SUITE_V4.get(b, {})
            rows.append({
                "model": m,
                "family": f["family"],
                "benchmark": b,
                "observed": int((m, b) in observed),
                "log_active_params": f["log_active_params"],
                "log_total_params": f["log_total_params"],
                "log_train_tokens": f["log_train_tokens"],
                "is_closed_source": f["is_closed_source"],
                "scale_imputed": f["scale_imputed"],
                "release_year_norm": f["release_year_norm"],
                "saturation_2026": meta.get("saturation_2026", np.nan),
                "contamination_resistant": int(meta.get("contamination_resistant", False)),
                "retained_v22": int(meta.get("retained_v22", False)),
                "output_space": meta.get("output_space", np.nan),
            })
    panel = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    by_bench = panel.groupby("benchmark").agg(models=("model", "nunique"), observed=("observed", "sum")).reset_index()
    by_bench["missing"] = by_bench["models"] - by_bench["observed"]
    by_bench["coverage_pct"] = 100.0 * by_bench["observed"] / by_bench["models"]
    by_family = panel.groupby("family").agg(cells=("observed", "size"), observed=("observed", "sum")).reset_index()
    by_family["missing"] = by_family["cells"] - by_family["observed"]
    by_family["coverage_pct"] = 100.0 * by_family["observed"] / by_family["cells"]

    # Small diagnostic missingness model. Interpret only as descriptive because cells are dependent.
    X = panel[["log_active_params", "log_total_params", "log_train_tokens", "is_closed_source", "scale_imputed", "release_year_norm", "saturation_2026", "contamination_resistant", "retained_v22", "output_space"]]
    y = panel["observed"].to_numpy(int)
    try:
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(X, y)
        prob = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        auc = np.nan
    summary = pd.DataFrame([{
        "possible_cells": len(panel),
        "observed_cells": int(panel["observed"].sum()),
        "coverage_pct": 100.0 * panel["observed"].mean(),
        "missingness_auc_descriptive": auc,
        "models": panel["model"].nunique(),
        "benchmarks": panel["benchmark"].nunique(),
    }])
    return panel, by_bench.sort_values("coverage_pct"), summary


def residual_by_protocol(resid: pd.DataFrame) -> pd.DataFrame:
    cols = ["source", "thinking", "tools_allowed", "reasoning_effort"]
    rows = []
    for col in cols:
        if col not in resid:
            continue
        for key, g in resid.groupby(col, dropna=False):
            if len(g) < 5:
                continue
            r = g["residual"].to_numpy(float)
            w = g["sample_weight"].to_numpy(float)
            rows.append({
                "protocol_dimension": col,
                "level": str(key),
                "n": len(g),
                "weighted_mae": weighted_mean(np.abs(r), w),
                "weighted_bias": weighted_mean(r, w),
                "p90_abs_error": float(np.quantile(np.abs(r), 0.90)),
            })
    return pd.DataFrame(rows).sort_values(["protocol_dimension", "weighted_mae"], ascending=[True, False])


def make_v6_figures(pooled, canonical_resid, by_bench, val_summary, temporal, stress, missing_panel):
    # Keep v5 figures in v6 names where useful.
    def savefig(name):
        plt.tight_layout()
        plt.savefig(FIG / name, dpi=220)
        plt.close()

    # Accuracy ablation.
    d = pooled.sort_values("mae", ascending=True)
    plt.figure(figsize=(6.0, 4.6))
    plt.bar(d["spec"].astype(str), d["mae"])
    plt.ylabel("Weighted MAE (percentage points)")
    plt.xlabel("Predictor set")
    plt.title("Family-grouped CV accuracy by model specification")
    plt.xticks(rotation=20, ha="right")
    savefig("accuracy_ablation.png")

    # Observed-predicted.
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(canonical_resid["prediction"], canonical_resid["score"], s=24, alpha=0.75)
    lo = min(canonical_resid["prediction"].min(), canonical_resid["score"].min()) - 2
    hi = max(canonical_resid["prediction"].max(), canonical_resid["score"].max()) + 2
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Out-of-fold predicted score (%)")
    plt.ylabel("Observed score (%)")
    plt.title("Observed versus predicted benchmark scores")
    savefig("observed_vs_predicted.png")

    # Residuals vs fitted and release.
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(canonical_resid["prediction"], canonical_resid["residual"], s=24, alpha=0.75)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Out-of-fold predicted score (%)")
    plt.ylabel("Residual: observed - predicted (pp)")
    plt.title("Residuals versus fitted values")
    savefig("residuals_vs_fitted.png")

    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(2022 + canonical_resid["release_year_norm"], canonical_resid["abs_residual"], s=24, alpha=0.75)
    plt.xlabel("Approximate model release year")
    plt.ylabel("Absolute residual (pp)")
    plt.title("Absolute residuals by model release era")
    savefig("abs_residual_by_release.png")

    # Benchmark MAE.
    d = by_bench.sort_values("weighted_mae", ascending=True)
    plt.figure(figsize=(7.2, 5.4))
    plt.barh(d["benchmark"].astype(str), d["weighted_mae"])
    plt.xlabel("Weighted MAE (percentage points)")
    plt.ylabel("Benchmark")
    plt.title("Residual error by benchmark")
    savefig("mae_by_benchmark.png")

    # Validation comparison.
    vd = val_summary.sort_values("weighted_mae", ascending=True)
    plt.figure(figsize=(7.5, 4.6))
    plt.barh(vd["validation"], vd["weighted_mae"])
    plt.xlabel("Weighted MAE (percentage points)")
    plt.ylabel("Validation regime")
    plt.title("Accuracy under stronger validation regimes")
    savefig("validation_comparison.png")

    # Temporal holdout.
    if len(temporal):
        plt.figure(figsize=(6.4, 4.6))
        plt.plot(temporal["cutoff_calendar_year"], temporal["weighted_mae"], marker="o")
        plt.xlabel("Training cutoff year")
        plt.ylabel("Test MAE on later models (pp)")
        plt.title("Temporal holdout stress test")
        savefig("temporal_holdout.png")

    # Param uncertainty histogram.
    if len(stress):
        plt.figure(figsize=(6.2, 4.5))
        plt.hist(stress["mae"].dropna(), bins=18, alpha=0.85)
        plt.axvline(stress["mae"].mean(), linestyle="--", linewidth=1)
        plt.xlabel("Full-model MAE under perturbed scale inputs (pp)")
        plt.ylabel("Monte Carlo count")
        plt.title("Closed-source scale-uncertainty stress test")
        savefig("param_uncertainty.png")

    # Missingness heatmap, aggregated by family/benchmark.
    pivot = missing_panel.pivot_table(index="family", columns="benchmark", values="observed", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    plt.figure(figsize=(9.5, 5.8))
    im = plt.imshow(pivot.to_numpy(float), aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Coverage share")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    plt.title("Benchmark reporting coverage by family")
    savefig("missingness_heatmap.png")


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


def prepare_latex_assets(summary, pooled, rsum, by_bench, by_family, by_source, outliers, sensitivity, coeffs,
                         val_summary, loo_family, loo_benchmark, temporal, stress_summary, missing_bench,
                         missing_summary, protocol_resid):
    # Pooled ablation table.
    pooled2 = pooled.copy()
    for c in ["mae", "mae_std"]:
        pooled2[c + "_fmt"] = pooled2[c].map(lambda x: format_num(x, 2))
    pooled_rows = to_latex_rows(pooled2, ["spec", "n", "n_families", "n_scored", "mae_fmt", "mae_std_fmt", "status"],
                                formats={"spec": latex_escape, "status": latex_escape})

    # Residual summary.
    rs = rsum.iloc[0]
    res_rows = "\n".join([
        rf"Cells & {int(rs['n'])} \\",
        rf"Weighted MAE & {rs['weighted_mae']:.2f} pp \\",
        rf"Weighted RMSE & {rs['weighted_rmse']:.2f} pp \\",
        rf"Weighted bias & {rs['weighted_bias']:.2f} pp \\",
        rf"Median absolute error & {rs['median_abs_error']:.2f} pp \\",
        rf"90th percentile absolute error & {rs['p90_abs_error']:.2f} pp \\",
        rf"Weighted $R^2$ & {rs['weighted_r2']:.2f} \\",
        rf"Calibration slope & {rs['calibration_slope_actual_on_pred']:.2f} \\",
        rf"Calibration intercept & {rs['calibration_intercept_actual_on_pred']:.2f} \\",
    ])

    # Benchmark residuals.
    bb = by_bench.copy().sort_values("weighted_mae", ascending=False)
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias", "p90_abs_error"]:
        bb[c + "_fmt"] = bb[c].map(lambda x: format_num(x, 2))
    bench_rows = to_latex_rows(bb, ["benchmark", "n", "models", "weighted_mae_fmt", "weighted_rmse_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
                               formats={"benchmark": latex_escape}, max_rows=16)

    fam = by_family.copy().sort_values("weighted_mae", ascending=False).head(12)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        fam[c + "_fmt"] = fam[c].map(lambda x: format_num(x, 2))
    family_rows = to_latex_rows(fam, ["family", "n", "models", "weighted_mae_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
                                formats={"family": latex_escape})

    src = by_source.copy().sort_values("weighted_mae", ascending=False)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        src[c + "_fmt"] = src[c].map(lambda x: format_num(x, 2))
    source_rows = to_latex_rows(src, ["source", "n", "models", "weighted_mae_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
                                formats={"source": latex_escape})

    oo = outliers.copy().head(12)
    for c in ["score", "prediction", "residual", "abs_residual"]:
        oo[c + "_fmt"] = oo[c].map(lambda x: format_num(x, 1))
    outlier_rows = to_latex_rows(oo, ["model", "benchmark", "score_fmt", "prediction_fmt", "residual_fmt", "source"],
                                 formats={"model": latex_escape, "benchmark": latex_escape, "source": latex_escape})

    sfull = sensitivity[sensitivity["spec"] == "Full"].copy().sort_values("mae")
    sfull["mae_fmt"] = sfull["mae"].map(lambda x: format_num(x, 2))
    sensitivity_rows = to_latex_rows(sfull, ["source_policy", "scale_policy", "target", "estimator", "n", "families", "mae_fmt", "status"],
                                     formats={"source_policy": latex_escape, "scale_policy": latex_escape, "target": latex_escape, "estimator": latex_escape, "status": latex_escape})

    cc = coeffs.copy().head(18)
    cc["coef_fmt"] = cc["standardized_coefficient"].map(lambda x: format_num(x, 3))
    coef_rows = to_latex_rows(cc, ["feature", "feature_group", "coef_fmt"],
                              formats={"feature": latex_escape, "feature_group": latex_escape})

    vv = val_summary.copy().sort_values("weighted_mae")
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias", "weighted_r2"]:
        vv[c + "_fmt"] = vv[c].map(lambda x: format_num(x, 2))
    val_rows = to_latex_rows(vv, ["validation", "n", "models", "families", "benchmarks", "weighted_mae_fmt", "weighted_rmse_fmt", "weighted_bias_fmt", "weighted_r2_fmt"],
                             formats={"validation": latex_escape})

    lf = loo_family.copy().head(10)
    for c in ["weighted_mae", "weighted_bias"]:
        lf[c + "_fmt"] = lf[c].map(lambda x: format_num(x, 2))
    lf_rows = to_latex_rows(lf, ["heldout_family", "n", "models", "benchmarks", "weighted_mae_fmt", "weighted_bias_fmt"],
                            formats={"heldout_family": latex_escape})

    lb = loo_benchmark.copy().head(14)
    for c in ["weighted_mae", "weighted_bias"]:
        lb[c + "_fmt"] = lb[c].map(lambda x: format_num(x, 2))
    lb_rows = to_latex_rows(lb, ["heldout_benchmark", "n", "models", "families", "weighted_mae_fmt", "weighted_bias_fmt"],
                            formats={"heldout_benchmark": latex_escape})

    tmp = temporal.copy()
    if len(tmp):
        for c in ["weighted_mae", "weighted_rmse", "weighted_bias"]:
            tmp[c + "_fmt"] = tmp[c].map(lambda x: format_num(x, 2))
        tmp["cutoff_fmt"] = tmp["cutoff_calendar_year"].map(lambda x: format_num(x, 2))
        temporal_rows = to_latex_rows(tmp, ["cutoff_fmt", "train_cells", "test_cells", "test_models", "weighted_mae_fmt", "weighted_rmse_fmt", "weighted_bias_fmt"])
    else:
        temporal_rows = r"No valid temporal split & -- & -- & -- & -- & -- & -- \\" 

    ss = stress_summary.iloc[0]
    stress_rows = "\n".join([
        rf"Iterations & {int(ss['iterations'])} \\",
        rf"Mean MAE & {ss['mae_mean']:.2f} pp \\",
        rf"MAE standard deviation & {ss['mae_std']:.2f} pp \\",
        rf"5th--95th percentile MAE & {ss['mae_p05']:.2f}--{ss['mae_p95']:.2f} pp \\",
    ])

    mb = missing_bench.copy()
    mb["coverage_fmt"] = mb["coverage_pct"].map(lambda x: format_num(x, 1))
    miss_bench_rows = to_latex_rows(mb, ["benchmark", "observed", "missing", "coverage_fmt"], formats={"benchmark": latex_escape})

    ms = missing_summary.iloc[0]
    missing_rows = "\n".join([
        rf"Possible model-benchmark cells & {int(ms['possible_cells'])} \\",
        rf"Observed canonical cells & {int(ms['observed_cells'])} \\",
        rf"Coverage & {ms['coverage_pct']:.1f}\% \\",
        rf"Descriptive missingness AUC & {ms['missingness_auc_descriptive']:.2f} \\",
    ])

    pr = protocol_resid.copy().head(18)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        pr[c + "_fmt"] = pr[c].map(lambda x: format_num(x, 2))
    protocol_rows = to_latex_rows(pr, ["protocol_dimension", "level", "n", "weighted_mae_fmt", "weighted_bias_fmt", "p90_abs_error_fmt"],
                                  formats={"protocol_dimension": latex_escape, "level": latex_escape})

    tex = rf"""
% Auto-generated by run_analysis.py. Do not edit manually.
\newcommand{{\VSixCanonicalCells}}{{{summary['n_primary_cells']}}}
\newcommand{{\VSixCanonicalModels}}{{{summary['n_models_primary']}}}
\newcommand{{\VSixCanonicalFamilies}}{{{summary['n_families_primary']}}}
\newcommand{{\VSixCanonicalBenchmarks}}{{{summary['n_benchmarks_primary']}}}
\newcommand{{\VSixWeightedMAE}}{{{summary['weighted_mae']:.2f}}}
\newcommand{{\VSixWeightedRMSE}}{{{summary['weighted_rmse']:.2f}}}
\newcommand{{\VSixWeightedBias}}{{{summary['weighted_bias']:.2f}}}
\newcommand{{\VSixWeightedRTwo}}{{{summary['weighted_r2']:.2f}}}
\newcommand{{\VSixCalibrationSlope}}{{{summary['calibration_slope']:.2f}}}
\newcommand{{\VSixPooledTable}}{{%
\begin{{tabular}}{{lrrrrrl}}\toprule
Specification & cells & families & scored & MAE & fold sd & status \\
\midrule
{pooled_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixResidualSummaryTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{res_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixBenchmarkResidualTable}}{{%
\begin{{tabular}}{{lrrrrrr}}\toprule
Benchmark & cells & models & MAE & RMSE & bias & p90 AE \\
\midrule
{bench_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixFamilyResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Family & cells & models & MAE & bias & p90 AE \\
\midrule
{family_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixSourceResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Source & cells & models & MAE & bias & p90 AE \\
\midrule
{source_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixOutlierTable}}{{%
\begin{{tabular}}{{llrrrl}}\toprule
Model & Benchmark & observed & predicted & residual & source \\
\midrule
{outlier_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixSensitivityTable}}{{%
\begin{{tabular}}{{llllrrrl}}\toprule
Source policy & scale & target & estimator & cells & families & MAE & status \\
\midrule
{sensitivity_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixCoefficientTable}}{{%
\begin{{tabular}}{{llr}}\toprule
Feature & group & standardized coefficient \\
\midrule
{coef_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixValidationTable}}{{%
\begin{{tabular}}{{lrrrrrrrr}}\toprule
Validation & cells & models & families & benchmarks & MAE & RMSE & bias & $R^2$ \\
\midrule
{val_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixLeaveFamilyTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Held-out family & cells & models & benchmarks & MAE & bias \\
\midrule
{lf_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixLeaveBenchmarkTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Held-out benchmark & cells & models & families & MAE & bias \\
\midrule
{lb_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixTemporalTable}}{{%
\begin{{tabular}}{{rrrrrrr}}\toprule
Cutoff yr & train cells & test cells & test models & MAE & RMSE & bias \\
\midrule
{temporal_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixStressTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{stress_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixMissingSummaryTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{missing_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixMissingBenchmarkTable}}{{%
\begin{{tabular}}{{lrrr}}\toprule
Benchmark & observed & missing & coverage \% \\
\midrule
{miss_bench_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\VSixProtocolResidualTable}}{{%
\begin{{tabular}}{{llrrrr}}\toprule
Dimension & level & cells & MAE & bias & p90 AE \\
\midrule
{protocol_rows}
\bottomrule\end{{tabular}}}}
"""
    (OUT / "auto_tables.tex").write_text(tex)


def write_report_tex():
    report = r'''
\documentclass[11pt]{article}
\usepackage[margin=0.95in]{geometry}
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
\usepackage{pdflscape}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\input{outputs/auto_tables.tex}

\title{Predictable Benchmarks? An Evaluation-Protocol-Adjusted Residual Analysis of Public LLM Benchmark Scores}
\author{Ed Herranz\\Independent model validation draft}
\date{Data snapshot -- May 2026}

\begin{document}
\maketitle

\begin{abstract}
Public leaderboards often treat reported benchmark scores as comparable point estimates of model capability.  In practice, the reported score is a measurement generated under a particular evaluation protocol: benchmark version, prompt setting, thinking or reasoning mode, sampling policy, tool access, source type, and score scale.  This paper studies how much of the cross-sectional variation in public LLM benchmark scores can be predicted from observable model metadata and evaluation-protocol covariates.  The goal is not to infer intelligence, prove contamination, or estimate a causal law.  The goal is narrower: estimate how much of public benchmark performance is predictable from public metadata, and use the remaining residuals as a diagnostic for benchmark informativeness, protocol mismatch, model specialization, and data-review priorities.

The canonical dataset contains \VSixCanonicalCells{} percent-score cells, \VSixCanonicalModels{} models, \VSixCanonicalFamilies{} model families, and \VSixCanonicalBenchmarks{} benchmark variants.  Gemini is included explicitly, with Gemini 2.5 and Gemini 3/3.1 variants separated by benchmark version and score scale where needed.  The canonical evaluation-protocol-adjusted ridge model obtains a source-weighted family-grouped cross-validation MAE of \VSixWeightedMAE{} percentage points, compared with weaker scale-only and scale-plus-recipe baselines.  Stronger tests show why the result should be interpreted carefully: leave-one-family-out validation, leave-one-benchmark-out validation, temporal holdouts, and closed-source scale perturbations produce materially different error profiles.  The main finding is therefore not that public benchmark scores can be predicted perfectly.  It is that a substantial portion of leaderboard variation is predictable from public metadata and measurement protocol, while residuals identify the cells and benchmark families that require substantive review.
\end{abstract}

\section{Introduction}
Large language model evaluation has become a public ranking exercise.  Model cards, technical reports, vendor blogs, and leaderboards report scores on MMLU, GPQA, AIME, HumanEval, LiveCodeBench, and many related tasks.  These scores are often read as if they were directly comparable measurements of a common latent capability.  That interpretation is too strong.  Benchmark scores are not only functions of the trained model.  They are also functions of benchmark version, prompting, answer extraction, sampling, tool access, thinking mode, source conventions, and publication choices.

This paper asks a deliberately modest question: how predictable are reported public LLM benchmark scores once one conditions on model metadata and observable evaluation protocol?  If a benchmark score is largely predictable from model family, scale, training-recipe proxies, and measurement conditions, then the raw score may contain less independent information than the leaderboard presentation suggests.  Conversely, a large residual is useful: it may identify a genuinely distinctive capability, a benchmark-specific training effect, a protocol mismatch, an outdated source, or a data-quality issue.

The contribution is not a new scaling law.  Scaling-law and benchmark-extrapolation work already shows that language-model performance can often be modeled as a smooth function of scale, training data, loss, family effects, or latent skills.  The contribution here is an evaluation-protocol-adjusted residual audit.  The paper treats public benchmark scores as heterogeneous measurements and asks whether residuals, rather than raw scores alone, are the more useful object for model validation.

\section{Related work and positioning}
Classical scaling laws show that language-model loss and downstream behavior are often predictable from model size, data, and compute.  Kaplan et al. established early neural scaling laws for language modeling, while Hoffmann et al. showed that compute-optimal training requires scaling model size and data together.  Later observational scaling-law work argues that public models can be used to infer smooth capability trends even when training runs are not controlled experiments.  Sloth/Skills Scaling Laws similarly predicts multi-benchmark performance across families by modeling low-dimensional latent skills.  Epoch AI's benchmark extrapolation work maps estimated loss or scaled compute to downstream benchmark performance.

This paper differs in three ways.  First, it does not try to estimate a universal capability axis or a causal compute law.  Second, it includes source and evaluation-protocol covariates as first-class measurement variables.  Third, it emphasizes residual diagnostics and validation stress tests rather than only headline predictive accuracy.  The intended use is benchmark governance: identify predictable benchmark cells, non-comparable measurements, surprising residuals, and gaps in public reporting.

\section{Terminology: evaluation-protocol-adjusted modeling}
This paper uses the term \emph{evaluation-protocol-adjusted} rather than \emph{protocol-aware}.  A benchmark score is evaluation-protocol-adjusted when it is modeled conditional not only on model attributes, but also on observable features of the procedure that generated the score.  These include benchmark version, score scale, prompt setting, chain-of-thought or thinking mode, test-time sampling, pass@k or consensus policy, tool access, source type, and whether the reported row appears to match the benchmark's primary protocol.

This distinction matters.  An AIME score obtained with many samples and majority voting is not exchangeable with a pass@1 no-tool AIME score.  A LiveCodeBench percentage score is not exchangeable with a LiveCodeBench Pro Elo score.  A vendor model-card row is not the same measurement object as a third-party aggregator row.  Evaluation-protocol adjustment is therefore a measurement-control device, not a claim that all protocol heterogeneity is fully solved.

\section{Data design}
\subsection{Observation unit}
The observation unit is a model-benchmark-source-protocol cell.  The dependent variable is a reported score on a 0--100 percent scale for canonical analysis.  Non-percent scores, such as Elo-style LiveCodeBench Pro, are retained in provenance files but excluded from the canonical percent-score regression unless separately normalized.  Benchmark versions are not collapsed when doing so would mix distinct tasks.  For example, AIME 2024 and AIME 2025 are separate benchmark variables, and LiveCodeBench percent scores are separate from LiveCodeBench Pro Elo scores.

\subsection{Gemini inclusion}
Gemini models are explicitly included.  Gemini 2.5 Pro and Gemini 2.5 Flash are represented using the Gemini 2.5 technical report where comparable benchmark cells are available.  Gemini 3 Pro and Gemini 3.1 Pro are included using the Gemini 3/3.1 model-card reporting where the benchmark version and score scale can be separated.  The paper does not mix Gemini 3.1 Pro's LiveCodeBench Pro Elo with percent-score LiveCodeBench rows.  This is an example of evaluation-protocol adjustment: retaining the information without forcing incompatible scales into one regression target.

\subsection{Predictor taxonomy}
The predictors are grouped into six classes:
\begin{enumerate}[leftmargin=*]
\item \textbf{Scale and capacity}: active parameter proxy, total parameter proxy, training-token proxy, scale bucket, MoE flag, parameter-confidence score, and scale-imputation flag.  Closed-source scale values are treated as uncertain proxies, not facts.
\item \textbf{Architecture and context}: context length, grouped-query attention flag, rotary-position-embedding flag, and MoE indicators.
\item \textbf{Training and post-training recipe proxies}: math intensity, code-pretraining intensity, synthetic-data quality, instruction tuning, reasoning-model flag, trained-web-only flag, and release-era control.  These are ordinal codings based on public documentation and should be treated as review variables, not direct measurements of private training mixtures.
\item \textbf{Evaluation-protocol variables}: source-confidence weight, exact-protocol-match flag, thinking-mode flag, reasoning-effort score, and tool-access flag.
\item \textbf{Benchmark metadata}: saturation proxy, multiple-choice/output-space proxy, contamination-resistance flag, and whether the benchmark is retained from older leaderboard suites.
\item \textbf{Interactions}: recipe-by-benchmark-property interactions, allowing reasoning, math, code, and synthetic-data proxies to matter differently on open-form, refreshed, or contamination-resistant tasks.
\end{enumerate}

\subsection{Source policy and sample weights}
The canonical run uses an official-like source policy: official model cards, first-party technical reports, cross-reported benchmark tables, recognized leaderboards, and simple-evals style sources.  Rows receive source-confidence weights, adjusted modestly for exact protocol matches.  These weights are not claims of truth.  They are a pragmatic way to avoid treating all source classes as equally reliable.

\section{Model formulation}
For percent-score benchmarks, the canonical target is a clipped logit transform:
\[
  z_{i b s p}=\log\left(\frac{\tilde y_{i b s p}}{1-\tilde y_{i b s p}}\right),
\]
where $\tilde y$ is the percentage score divided by 100 and clipped away from zero and one.  Predictions are transformed back to percentage points before reporting MAE, RMSE, bias, and residual diagnostics.

The full pooled specification is
\[
 z_{i b s p}=\alpha_b + x_i'\beta + a_i'\eta + r_i'\rho + m_{s p}'\gamma + q_b'\delta
 + (r_i \otimes q_b)'\theta + \varepsilon_{i b s p},
\]
where $\alpha_b$ are benchmark fixed effects, $x_i$ contains scale predictors, $a_i$ architecture/context predictors, $r_i$ training-recipe proxies, $m_{sp}$ evaluation-protocol/source predictors, and $q_b$ benchmark metadata.  The interaction term allows recipe variables to differ by benchmark properties.  The canonical model is ridge regression because the panel is sparse and the predictors are collinear.  Coefficients are descriptive, not structural.

\section{Validation design}
The analysis uses five validation regimes.
\begin{enumerate}[leftmargin=*]
\item \textbf{Family-grouped cross-validation}: entire model families are assigned to folds.  This is the canonical accuracy measure.
\item \textbf{Leave-one-family-out}: each family is held out entirely.  This is stricter and exposes whether family lineage is being used implicitly.
\item \textbf{Leave-one-benchmark-out}: one benchmark is held out and benchmark fixed effects are disabled.  This asks whether model and benchmark metadata can generalize to an unseen benchmark rather than interpolate among known benchmark fixed effects.
\item \textbf{Temporal holdout}: models released after a cutoff are predicted from earlier-release models.  This approximates the use case of forecasting future leaderboard cells.
\item \textbf{Closed-source parameter uncertainty}: active-parameter, total-parameter, and training-token proxies are perturbed more heavily for closed-source or imputed rows.  This checks whether the headline result is fragile to uncertain scale metadata.
\end{enumerate}

\section{Results}
\subsection{Canonical ablation}
\begin{table}[H]
\centering\small
\caption{Canonical family-grouped cross-validation accuracy by predictor set. MAE is source-weighted and reported in percentage points.}
\VSixPooledTable
\label{tab:pooled}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.72\textwidth]{figures/accuracy_ablation.png}
\caption{Family-grouped cross-validation weighted MAE by predictor set. Lower is better.}
\label{fig:ablation}
\end{figure}

The full model improves on the scale-only and scale-plus-recipe baselines.  The interpretation is not that the model has discovered a stable causal law.  The interpretation is that public benchmark scores are substantially compressible using public metadata and measurement controls.  That is already a meaningful governance result: a raw leaderboard row should not be treated as entirely independent evidence of capability.

\subsection{Accuracy and calibration}
\begin{figure}[H]
\centering
\includegraphics[width=0.70\textwidth]{figures/observed_vs_predicted.png}
\caption{Observed versus out-of-fold predicted scores for the canonical full model.}
\label{fig:obs_pred}
\end{figure}

\begin{table}[H]
\centering\small
\caption{Canonical residual diagnostics. Residual is observed minus predicted.}
\VSixResidualSummaryTable
\label{tab:res_summary}
\end{table}

The calibration slope is \VSixCalibrationSlope{} and the weighted bias is \VSixWeightedBias{} percentage points.  The model is useful for ordering and residual review, but the residual tails are too large for the model to certify individual score cells.

\section{Residual analysis}
\subsection{Residual plots}
\begin{figure}[H]
\centering
\begin{subfigure}{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/residuals_vs_fitted.png}
\caption{Residuals versus fitted values.}
\end{subfigure}
\begin{subfigure}{0.49\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/abs_residual_by_release.png}
\caption{Absolute residuals by release era.}
\end{subfigure}
\caption{Residual diagnostics by prediction level and release era.}
\label{fig:resids}
\end{figure}

Residuals are centered near zero but heteroskedastic.  Larger residuals often occur in frontier-era rows and refreshed math/coding benchmarks, where protocols and model-reporting conventions vary the most.

\subsection{Residuals by benchmark, family, source, and protocol}
\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/mae_by_benchmark.png}
\caption{Weighted MAE by benchmark.}
\label{fig:bench_mae}
\end{figure}

\begin{table}[H]
\centering\scriptsize
\caption{Residual diagnostics by benchmark. Positive bias means the model underpredicted the benchmark score.}
\resizebox{\textwidth}{!}{\VSixBenchmarkResidualTable}
\label{tab:bench_residuals}
\end{table}

\begin{table}[H]
\centering\scriptsize
\caption{Largest family-level residual diagnostics.}
\resizebox{0.88\textwidth}{!}{\VSixFamilyResidualTable}
\label{tab:family_residuals}
\end{table}

\begin{table}[H]
\centering\small
\caption{Residual diagnostics by source class.}
\VSixSourceResidualTable
\label{tab:source_residuals}
\end{table}

\begin{table}[H]
\centering\scriptsize
\caption{Residual diagnostics by evaluation-protocol dimensions.}
\resizebox{\textwidth}{!}{\VSixProtocolResidualTable}
\label{tab:protocol_residuals}
\end{table}

\subsection{Outliers as review queue}
\begin{table}[H]
\centering\scriptsize
\caption{Largest absolute out-of-fold residuals. These rows should be reviewed as data, protocol, or specialization cases before drawing substantive conclusions.}
\resizebox{\textwidth}{!}{\VSixOutlierTable}
\label{tab:outliers}
\end{table}

Outliers are not automatically errors.  They are a review queue.  A large positive residual may indicate genuine model specialization or a favorable hidden protocol.  A large negative residual may indicate an unfavorable protocol, a stale score, an overestimated scale proxy, or a benchmark-specific weakness.

\section{Stronger validation and limitation stress tests}
\subsection{Validation-regime comparison}
\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/validation_comparison.png}
\caption{Weighted MAE under stronger validation regimes.}
\label{fig:validation}
\end{figure}

\begin{table}[H]
\centering\scriptsize
\caption{Validation-regime comparison. Leave-one-benchmark-out disables benchmark fixed effects, so it is not directly comparable to the canonical interpolation task.}
\resizebox{\textwidth}{!}{\VSixValidationTable}
\label{tab:validation}
\end{table}

The leave-one-benchmark-out task is intentionally harder because the model cannot rely on the held-out benchmark fixed effect.  A degradation in that setting is not a failure of the canonical model; it is evidence that benchmark identity contains information not fully captured by the current benchmark metadata.

\subsection{Leave-one-family and leave-one-benchmark details}
\begin{table}[H]
\centering\scriptsize
\caption{Highest-error leave-one-family-out cases.}
\resizebox{0.88\textwidth}{!}{\VSixLeaveFamilyTable}
\label{tab:lofo}
\end{table}

\begin{table}[H]
\centering\scriptsize
\caption{Leave-one-benchmark-out results. Benchmark fixed effects are disabled in this validation.}
\resizebox{0.88\textwidth}{!}{\VSixLeaveBenchmarkTable}
\label{tab:lobo}
\end{table}

\subsection{Temporal holdout}
\begin{figure}[H]
\centering
\includegraphics[width=0.70\textwidth]{figures/temporal_holdout.png}
\caption{Temporal holdout MAE on models released after the training cutoff.}
\label{fig:temporal}
\end{figure}

\begin{table}[H]
\centering\small
\caption{Temporal holdout stress tests.}
\VSixTemporalTable
\label{tab:temporal}
\end{table}

Temporal validation is important because leaderboard prediction is often used prospectively.  The current dataset is still too small for a definitive future-forecasting claim, but the test helps identify whether the result is merely retrospective interpolation.

\subsection{Closed-source scale uncertainty}
\begin{figure}[H]
\centering
\includegraphics[width=0.70\textwidth]{figures/param_uncertainty.png}
\caption{Distribution of full-model MAE after perturbing uncertain scale and token proxies.}
\label{fig:stress}
\end{figure}

\begin{table}[H]
\centering\small
\caption{Closed-source scale and token uncertainty stress test.}
\VSixStressTable
\label{tab:stress}
\end{table}

The perturbation analysis directly addresses the concern that closed-source parameter counts and training-token estimates are weak inputs.  If the MAE distribution were highly unstable, the headline conclusion would be fragile.  The stress test should be interpreted as a sensitivity check, not a fully Bayesian uncertainty propagation.

\section{Missingness and publication selection}
Missing benchmark cells are not random.  Labs choose which benchmarks to report, and different sources emphasize different tasks.  This release therefore adds a missingness diagnostic over possible model-benchmark cells.

\begin{figure}[H]
\centering
\includegraphics[width=0.90\textwidth]{figures/missingness_heatmap.png}
\caption{Canonical benchmark reporting coverage by model family.}
\label{fig:missingness}
\end{figure}

\begin{table}[H]
\centering\small
\caption{Missingness summary. The AUC is descriptive and should not be interpreted causally.}
\VSixMissingSummaryTable
\label{tab:missing_summary}
\end{table}

\begin{table}[H]
\centering\scriptsize
\caption{Coverage by benchmark in the possible model-benchmark panel.}
\VSixMissingBenchmarkTable
\label{tab:missing_benchmark}
\end{table}

This is one of the biggest remaining limitations.  The regression uses observed cells; it does not yet model the strategic decision to publish, omit, or selectively highlight benchmark results.  A future version should model the observation process jointly with score prediction.

\section{Sensitivity analysis and coefficients}
\begin{table}[H]
\centering\scriptsize
\caption{Full-model sensitivity to source policy, scale policy, target transform, and estimator.}
\resizebox{\textwidth}{!}{\VSixSensitivityTable}
\label{tab:sensitivity}
\end{table}

\begin{table}[H]
\centering\scriptsize
\caption{Largest standardized coefficients from the canonical full ridge model. Coefficients are descriptive because predictors are correlated and regularized.}
\resizebox{\textwidth}{!}{\VSixCoefficientTable}
\label{tab:coefficients}
\end{table}

Coefficients are not causal effects.  Family lineage, scale, release era, and recipe proxies are correlated.  The coefficient table is useful mainly for auditing which variables drive fitted values and whether the signs are grossly implausible.

\section{Limitations addressed in This release and limitations that remain}
\begin{longtable}{p{0.22\textwidth}p{0.36\textwidth}p{0.34\textwidth}}
\caption{Limitations, This release mitigation, and remaining risk.}\\
\toprule
Limitation & This release mitigation & Remaining risk \\
\midrule
\endfirsthead
\toprule
Limitation & This release mitigation & Remaining risk \\
\midrule
\endhead
Closed-source scale uncertainty & Parameter-confidence fields, scale-imputation flags, bucketed-scale sensitivity, and perturbation stress test & No public audit of actual active parameters, routing, training tokens, or post-training data \\
Evaluation-protocol heterogeneity & Evaluation-protocol covariates, source weights, benchmark-version separation, and incompatible score-scale exclusion & Test-time compute, verifier policy, answer extraction, and exact prompts are often undisclosed \\
Family leakage & Family-grouped CV and leave-one-family-out validation & Families are still broad labels; within-family model variants are not independent \\
Benchmark interpolation & Leave-one-benchmark-out validation with benchmark fixed effects disabled & Benchmark metadata is still too coarse to predict truly novel benchmarks reliably \\
Temporal overfitting & Temporal holdout by release era & Public data are sparse and frontier model releases are clustered \\
Missing-not-at-random reporting & Missingness panel, family/benchmark coverage heatmap, descriptive missingness model & Score model still conditions on observed cells rather than jointly modeling selection \\
Hand-coded recipe variables & Explicit predictor taxonomy and documented ordinal role & Recipe proxies still require independent coding, inter-rater reliability, and better source documentation \\
\bottomrule
\end{longtable}

\section{Publication-worthy claim}
The safest publishable claim is not that LLM intelligence can be predicted from architecture.  Nor is it that benchmark contamination has been proven.  The claim is more precise:
\begin{quote}
A substantial share of public LLM benchmark-score variation is predictable from public model metadata and evaluation-protocol covariates.  Therefore, residuals from an evaluation-protocol-adjusted benchmark model provide a useful diagnostic for benchmark informativeness, protocol comparability, model specialization, and data-quality review.
\end{quote}

This framing distinguishes the paper from scaling laws.  Scaling-law papers ask how performance changes with scale, loss, compute, or latent capability.  This paper asks how much independent information remains in public benchmark rows after conditioning on observable metadata and measurement conditions.

\section{Conclusion}
The This release analysis strengthens the prior draft by making the paper standalone, replacing informal terminology, adding richer predictor definitions, and adding validation regimes that directly test the largest weaknesses.  The full model predicts canonical held-out benchmark cells better than scale-only baselines, but stronger validations show that the model is not a benchmark oracle.  Its main value is diagnostic.  Predictable rows reveal where leaderboards may be partly recapitulating public metadata and protocol choices.  Unpredictable rows identify where substantive review is most needed.

\appendix
\section{Feature-coding rubric}
Recipe proxies should be read as ordinal review variables:
\begin{itemize}[leftmargin=*]
\item \textbf{Math intensity}: 0 for no specific public evidence; 1 for general post-training or math examples; 2 for explicit math-heavy tuning/evaluation emphasis; 3 for model family positioned around frontier reasoning/math performance.
\item \textbf{Code pretraining}: 0 for no public coding emphasis; 1 for general code capability; 2 for substantial code tuning or coding benchmarks; 3 for code-specialized or agentic-coding model family.
\item \textbf{Synthetic quality}: 0 for no public evidence; 1 for ordinary instruction/synthetic data; 2 for documented synthetic/self-improvement/data-quality emphasis; 3 for central model-family claim.
\item \textbf{Reasoning model}: 1 for models explicitly marketed or evaluated as reasoning/thinking models; 0 otherwise.
\end{itemize}
A publication version should add a second independent coding pass and report agreement statistics.

\section{Reproducibility}
The package is reproducible with:
\begin{verbatim}
python run_analysis.py
pdflatex report_v6.tex
pdflatex report_v6.tex
\end{verbatim}
The main generated files are in \texttt{outputs\_v6/} and \texttt{figures\_v6/}.

\begin{thebibliography}{12}
\bibitem{kaplan} Kaplan, J., McCandlish, S., Henighan, T., et al. \emph{Scaling Laws for Neural Language Models}. arXiv:2001.08361, 2020. \url{https://arxiv.org/abs/2001.08361}
\bibitem{chinchilla} Hoffmann, J., Borgeaud, S., Mensch, A., et al. \emph{Training Compute-Optimal Large Language Models}. arXiv:2203.15556, 2022. \url{https://arxiv.org/abs/2203.15556}
\bibitem{observational} Ruan, Y., Maddison, C. J., and Hashimoto, T. \emph{Observational Scaling Laws and the Predictability of Language Model Performance}. NeurIPS 2024. \url{https://proceedings.neurips.cc/paper_files/paper/2024/file/1cded4f97cf5f01a284c574110b7e3b9-Paper-Conference.pdf}
\bibitem{sloth} Polo, F. M., Somerstep, S., Choshen, L., Sun, Y., and Yurochkin, M. \emph{Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families}. ICLR submission page, 2025. \url{https://openreview.net/forum?id=D5v491uCzm}
\bibitem{epoch} Epoch AI. \emph{How Predictable is Language Model Benchmark Performance?} 2023. \url{https://epoch.ai/blog/how-predictable-is-language-model-benchmark-performance}
\bibitem{livecodebench} Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. \emph{LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code}. arXiv:2403.07974, 2024. \url{https://arxiv.org/abs/2403.07974}
\bibitem{livecodebench_site} LiveCodeBench. \emph{Holistic and Contamination Free Evaluation of Large Language Models for Code}. \url{https://livecodebench.github.io/}
\bibitem{gemini25} Google Gemini Team. \emph{Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities}. 2025. \url{https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf}
\bibitem{gemini31} Google DeepMind. \emph{Gemini 3.1 Pro Model Card}. February 2026. \url{https://deepmind.google/models/model-cards/gemini-3-1-pro/}
\end{thebibliography}

\end{document}
'''
    (ROOT / "paper" / "generated_report_from_script.tex").write_text(report)


def write_readme(summary):
    """Write a generated run summary without overwriting the curated README."""
    txt = f"""# Generated run summary

Canonical run:

- Source policy: {summary['source_policy']}
- Scale policy: {summary['scale_policy']}
- Target: {summary['target']}
- Estimator: {summary['estimator']}
- Primary percent-score cells: {summary['n_primary_cells']}
- Models: {summary['n_models_primary']}
- Families: {summary['n_families_primary']}
- Benchmarks: {summary['n_benchmarks_primary']}
- Full-model weighted MAE: {summary['weighted_mae']:.2f} percentage points
- Weighted RMSE: {summary['weighted_rmse']:.2f} percentage points
- Weighted bias: {summary['weighted_bias']:.2f} percentage points
- Weighted R^2: {summary['weighted_r2']:.2f}
- Bootstrap full-model MAE 95% CI: {summary['bootstrap_full_mae_ci_low']:.2f}--{summary['bootstrap_full_mae_ci_high']:.2f} percentage points
- Ordinal recipe sensitivity MAE 5th--95th percentile: {summary['recipe_sensitivity_mae_p05']:.2f}--{summary['recipe_sensitivity_mae_p95']:.2f} percentage points

This file is generated by `src/run_analysis.py`. The canonical human-facing repository description is `README.md`.
"""
    (OUT / "run_summary.md").write_text(txt)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-policy", choices=core.SOURCE_POLICIES.keys(), default="official_like")
    p.add_argument("--scale-policy", choices=["continuous", "bucket"], default="continuous")
    p.add_argument("--target", choices=["raw", "logit"], default="logit")
    p.add_argument("--estimator", choices=["ridge", "huber"], default="ridge")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-models", type=int, default=10)
    p.add_argument("--stress-iters", type=int, default=60)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--recipe-sensitivity-iters", type=int, default=60)
    ns = p.parse_args()
    args = ArgsObj(ns.source_policy, 0.0, ns.scale_policy, ns.target, ns.estimator, ns.n_folds, ns.min_models)

    primary, all_cells, per_bench, pooled, preds, coverage, src = v5.run_once(args)
    canonical_resid = v5.add_residual_columns(preds, spec="Full")
    # Add protocol/release columns back for richer diagnostics.
    enrich_cols = ["model", "benchmark", "thinking", "tools_allowed", "reasoning_effort", "release_year_norm", "source_detail"]
    canonical_resid = canonical_resid.merge(primary[enrich_cols], on=["model", "benchmark"], how="left")
    rsum = v5.residual_summary(canonical_resid)
    by_bench = v5.group_residuals(canonical_resid, "benchmark")
    by_family = v5.group_residuals(canonical_resid, "family")
    by_source = v5.group_residuals(canonical_resid, "source")
    outliers = canonical_resid.sort_values("abs_residual", ascending=False).head(30)
    sensitivity = v5.sensitivity_grid()
    coeffs = v5.fit_full_coefficients(primary, args, spec="Full")

    loo_family, loo_family_preds = leave_one_family(primary, args)
    loo_bench, loo_bench_preds = leave_one_benchmark(primary, args)
    temporal = temporal_holdouts(primary, args)
    stress = parameter_uncertainty(primary, args, n_iter=ns.stress_iters)
    stress_summary = pd.DataFrame([{
        "iterations": len(stress),
        "mae_mean": float(stress["mae"].mean()),
        "mae_std": float(stress["mae"].std()),
        "mae_p05": float(stress["mae"].quantile(0.05)),
        "mae_p95": float(stress["mae"].quantile(0.95)),
    }])
    bootstrap, bootstrap_summary = bootstrap_oof_mae(preds, n_iter=ns.bootstrap_iters)
    recipe_sens, recipe_sens_summary = ordinal_recipe_sensitivity(primary, args, n_iter=ns.recipe_sensitivity_iters)
    feature_coding = write_feature_coding_table()
    missing_panel, missing_bench, missing_summary = missingness_diagnostics(all_cells, primary)
    protocol_resid = residual_by_protocol(canonical_resid)

    val_rows = []
    val_rows.append(validation_metrics("family-grouped-CV", canonical_resid, canonical_resid["prediction"].to_numpy(float)))
    val_rows.append(validation_metrics("leave-one-family-out", loo_family_preds, loo_family_preds["prediction"].to_numpy(float)))
    val_rows.append(validation_metrics("leave-one-benchmark-out", loo_bench_preds, loo_bench_preds["prediction"].to_numpy(float)))
    if len(temporal):
        # Overall temporal summary: use the split with cutoff nearest 2025.0 if available; otherwise first.
        tv = temporal.iloc[(temporal["cutoff_calendar_year"] - 2025.0).abs().argmin()].to_dict()
        tv["validation"] = "temporal-holdout"
        val_rows.append({k: tv[k] for k in ["validation", "n", "models", "families", "benchmarks", "weighted_mae", "weighted_rmse", "weighted_bias", "weighted_r2"]})
    val_summary = pd.DataFrame(val_rows)

    # Save diagnostics.
    primary.to_csv(OUT / "canonical_primary_cells.csv", index=False)
    all_cells.to_csv(OUT / "canonical_all_cells.csv", index=False)
    pooled.to_csv(OUT / "pooled_results_canonical.csv", index=False)
    per_bench.to_csv(OUT / "per_benchmark_results_canonical.csv", index=False)
    preds.to_csv(OUT / "pooled_oof_predictions_canonical.csv", index=False)
    canonical_resid.to_csv(OUT / "residual_diagnostics_full.csv", index=False)
    rsum.to_csv(OUT / "residual_summary_full.csv", index=False)
    by_bench.to_csv(OUT / "residual_by_benchmark.csv", index=False)
    by_family.to_csv(OUT / "residual_by_family.csv", index=False)
    by_source.to_csv(OUT / "residual_by_source.csv", index=False)
    protocol_resid.to_csv(OUT / "residual_by_protocol.csv", index=False)
    outliers.to_csv(OUT / "residual_outliers_top30.csv", index=False)
    sensitivity.to_csv(OUT / "sensitivity_grid.csv", index=False)
    coeffs.to_csv(OUT / "full_model_standardized_coefficients.csv", index=False)
    loo_family.to_csv(OUT / "leave_one_family.csv", index=False)
    loo_family_preds.to_csv(OUT / "leave_one_family_predictions.csv", index=False)
    loo_bench.to_csv(OUT / "leave_one_benchmark.csv", index=False)
    loo_bench_preds.to_csv(OUT / "leave_one_benchmark_predictions.csv", index=False)
    temporal.to_csv(OUT / "temporal_holdout.csv", index=False)
    stress.to_csv(OUT / "parameter_uncertainty_stress.csv", index=False)
    stress_summary.to_csv(OUT / "parameter_uncertainty_summary.csv", index=False)
    bootstrap.to_csv(OUT / "bootstrap_mae_ci.csv", index=False)
    bootstrap_summary.to_csv(OUT / "bootstrap_mae_summary.csv", index=False)
    recipe_sens.to_csv(OUT / "recipe_coding_sensitivity.csv", index=False)
    recipe_sens_summary.to_csv(OUT / "recipe_coding_sensitivity_summary.csv", index=False)
    feature_coding.to_csv(OUT / "model_feature_coding.csv", index=False)
    missing_panel.to_csv(OUT / "missingness_panel.csv", index=False)
    missing_bench.to_csv(OUT / "missingness_by_benchmark.csv", index=False)
    missing_summary.to_csv(OUT / "missingness_summary.csv", index=False)
    val_summary.to_csv(OUT / "validation_regime_summary.csv", index=False)

    make_v6_figures(pooled, canonical_resid, by_bench, val_summary, temporal, stress, missing_panel)

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
        "leave_one_family_mae_fold_mean": float(loo_family["weighted_mae"].mean()),
        "leave_one_benchmark_mae_fold_mean": float(loo_bench["weighted_mae"].mean()),
        "leave_one_family_mae_pooled": float(val_summary.loc[val_summary["validation"] == "leave-one-family-out", "weighted_mae"].iloc[0]),
        "leave_one_benchmark_mae_pooled": float(val_summary.loc[val_summary["validation"] == "leave-one-benchmark-out", "weighted_mae"].iloc[0]),
        "bootstrap_full_mae_ci_low": float(bootstrap_summary["full_mae_ci_low"].iloc[0]),
        "bootstrap_full_mae_ci_high": float(bootstrap_summary["full_mae_ci_high"].iloc[0]),
        "bootstrap_improvement_ci_low": float(bootstrap_summary["improvement_ci_low"].iloc[0]),
        "bootstrap_improvement_ci_high": float(bootstrap_summary["improvement_ci_high"].iloc[0]),
        "bootstrap_improvement_p_le_zero": float(bootstrap_summary["p_improvement_le_zero"].iloc[0]),
        "recipe_sensitivity_mae_p05": float(recipe_sens_summary["mae_p05"].iloc[0]),
        "recipe_sensitivity_mae_p95": float(recipe_sens_summary["mae_p95"].iloc[0]),
        "stress_mae_mean": float(stress_summary["mae_mean"].iloc[0]),
        "stress_mae_p05": float(stress_summary["mae_p05"].iloc[0]),
        "stress_mae_p95": float(stress_summary["mae_p95"].iloc[0]),
        "missingness_coverage_pct": float(missing_summary["coverage_pct"].iloc[0]),
        "gemini_models_included": sorted([m for m in all_cells["model"].unique() if "Gemini" in m]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    prepare_latex_assets(summary, pooled, rsum, by_bench, by_family, by_source, outliers, sensitivity, coeffs,
                         val_summary, loo_family, loo_bench, temporal, stress_summary, missing_bench,
                         missing_summary, protocol_resid)
    write_report_tex()
    write_readme(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
