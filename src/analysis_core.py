#!/usr/bin/env python3
"""
analysis_core.py - v4 analysis pipeline.

Main v4 changes relative to v3:
  * source-policy filtering and source-confidence sample weights;
  * protocol/thinking metadata used as measurement controls;
  * active-vs-total parameter separation plus closed-source imputation flags;
  * bounded-score logit target by default;
  * regularized pooled cell-level model with benchmark fixed effects and
    benchmark-property interactions, plus per-benchmark diagnostics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from scipy import stats as sp_stats

from benchmark_suite import BENCHMARK_SUITE_V4, PRIMARY_BENCHMARKS_V4, SUBSTITUTABILITY_HAND_V4
from model_features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "core"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

SOURCE_POLICIES = {
    "full": None,
    "official_like": {"official", "cross_report", "leaderboard", "simple_evals"},
    "official_only": {"official"},
    "high_confidence": {"official", "cross_report", "leaderboard", "simple_evals"},
}
SOURCE_RANK = {"official": 0, "cross_report": 1, "simple_evals": 2, "leaderboard": 3, "aggregator": 4, "estimated": 5, "carried_forward": 6}
REASONING_ORDER = {"none": 0, "off": 0, "unknown": 0.25, "standard": 0.65, "dynamic": 0.70, "high": 1.0, "on": 0.70}


def pct_to_logit(y_pct: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(np.asarray(y_pct, dtype=float) / 100.0, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def logit_to_pct(z: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def grouped_family_folds(families: np.ndarray, n_folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Greedy family-grouped folds with approximately balanced row counts."""
    rng = np.random.RandomState(seed)
    families = np.asarray(families)
    unique = np.array(sorted(pd.unique(families)))
    n_folds = max(2, min(n_folds, len(unique)))
    sizes = {f: int(np.sum(families == f)) for f in unique}
    buckets: dict[int, list[str]] = {}
    for f in unique:
        buckets.setdefault(sizes[f], []).append(f)
    ordered = []
    for size in sorted(buckets, reverse=True):
        vals = buckets[size]
        rng.shuffle(vals)
        ordered.extend(vals)
    fold_counts = np.zeros(n_folds, dtype=int)
    fam_to_fold = {}
    for f in ordered:
        k = int(np.argmin(fold_counts))
        fam_to_fold[f] = k
        fold_counts[k] += sizes[f]
    return np.array([fam_to_fold[f] for f in families])


def load_provenance(source_policy: str, min_source_weight: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "benchmark_provenance.csv")
    allowed = SOURCE_POLICIES[source_policy]
    if allowed is not None:
        df = df[df["source"].isin(allowed)].copy()
    if min_source_weight > 0:
        df = df[df["source_confidence_weight"] >= min_source_weight].copy()
    df = df[df["model"].isin(MODEL_FEATURES)].copy()
    # Keep percent scores in the canonical range. Non-percent scores are retained
    # for coverage tables but not used in the primary analysis.
    df = df[(df["metric_type"] != "percent") | df["score"].between(0, 100)].copy()
    # Multiple rows can exist for the same model/benchmark after v4 extensions.
    # Choose by source rank, exact protocol flag, confidence weight, and date.
    df["_source_rank"] = df["source"].map(SOURCE_RANK).fillna(99)
    df["_date_sort"] = pd.to_datetime(df["date"].astype(str) + "-01", errors="coerce")
    df = df.sort_values(
        ["model", "benchmark", "_source_rank", "exact_protocol_match_to_primary_suite", "source_confidence_weight", "_date_sort"],
        ascending=[True, True, True, False, False, False],
    )
    df = df.drop_duplicates(["model", "benchmark"], keep="first")
    return df.drop(columns=["_source_rank", "_date_sort"])


def feature_rows(scale_policy: str = "continuous") -> pd.DataFrame:
    rows = []
    for name, info in MODEL_FEATURES.items():
        active = float(info.get("active_params_B", info.get("params", 1.0)))
        total = float(info.get("total_params_B", active))
        if scale_policy == "bucket":
            active = float(info.get("scale_bucket", 1))
            total = active
        tokens = float(max(info.get("training_tokens_B", 1), 1))
        ctx = float(max(info.get("context_length", 1), 1))
        row = {
            "model": name,
            "family": info.get("family", name.split("-")[0]),
            "log_active_params": np.log(max(active, 0.01)),
            "log_total_params": np.log(max(total, 0.01)),
            "log_train_tokens": np.log(tokens),
            "scale_bucket": float(info.get("scale_bucket", 1)),
            "param_confidence": float(info.get("param_confidence", 0.5)),
            "scale_imputed": int(info.get("scale_imputed", info.get("is_closed_source", 0))),
            "is_moe": int(info.get("is_moe", 0)),
            "log_context": np.log(ctx),
            "is_gqa": 1 if info.get("attention_type") == 1 else 0,
            "has_rope": 1 if info.get("position_encoding") == 1 else 0,
            "math_intensity": float(info.get("math_intensity", 0)),
            "code_pretrain": float(info.get("code_pretrain", 0)),
            "synthetic_quality": float(info.get("synthetic_quality", 0)),
            "trained_web_only": float(info.get("trained_web_only", 0)),
            "is_reasoning_model": float(info.get("is_reasoning_model", 0)),
            "is_instruct": float(info.get("is_instruct", 0)),
            "is_closed_source": float(info.get("is_closed_source", 0)),
            "release_year_norm": float(info.get("release_year", 2023) - 2022),
        }
        row["active_x_math"] = row["log_active_params"] * row["math_intensity"]
        row["active_x_code"] = row["log_active_params"] * row["code_pretrain"]
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_cell_df(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    prov = load_provenance(args.source_policy, args.min_source_weight)
    feats = feature_rows(args.scale_policy)
    df = prov.merge(feats, on="model", how="inner")
    for key in ["saturation_2026", "n_choices", "output_space"]:
        df[key] = df["benchmark"].map(lambda b: BENCHMARK_SUITE_V4.get(b, {}).get(key, np.nan))
    df["contamination_resistant"] = df["benchmark"].map(lambda b: int(BENCHMARK_SUITE_V4.get(b, {}).get("contamination_resistant", False)))
    df["retained_v22"] = df["benchmark"].map(lambda b: int(BENCHMARK_SUITE_V4.get(b, {}).get("retained_v22", False)))
    df["include_primary"] = df["benchmark"].map(lambda b: int(BENCHMARK_SUITE_V4.get(b, {}).get("include_primary", False)))
    df["metric_is_percent"] = (df["metric_type"] == "percent").astype(int)
    df["thinking_on_cell"] = df["thinking"].fillna("unknown").str.lower().map(lambda x: 1.0 if x in {"on", "dynamic"} else 0.0)
    df["reasoning_effort_score"] = df["reasoning_effort"].fillna("unknown").str.lower().map(lambda x: REASONING_ORDER.get(x, 0.25))
    df["tools_allowed_flag"] = df["tools_allowed"].fillna("unknown").str.lower().map(lambda x: 1.0 if x in {"yes", "tools", "search+code", "search", "code"} else 0.0)
    df["sample_weight"] = df["source_confidence_weight"].astype(float) * (0.85 + 0.15 * df["exact_protocol_match_to_primary_suite"].astype(float))
    primary = df[(df["include_primary"] == 1) & (df["metric_type"] == "percent")].copy()
    return primary, df


BASE_SCALE = ["log_active_params", "log_total_params", "log_train_tokens", "param_confidence", "scale_imputed", "is_moe"]
BASE_ARCH = BASE_SCALE + ["log_context", "is_gqa", "has_rope"]
BASE_RECIPE = [
    "log_active_params", "log_total_params", "log_train_tokens", "param_confidence", "scale_imputed", "is_moe",
    "math_intensity", "code_pretrain", "synthetic_quality", "trained_web_only", "is_reasoning_model", "is_instruct",
    "release_year_norm", "active_x_math", "active_x_code",
]
MEASUREMENT = ["source_confidence_weight", "exact_protocol_match_to_primary_suite", "thinking_on_cell", "reasoning_effort_score", "tools_allowed_flag"]
SPECS = {
    "Scale": BASE_SCALE,
    "Scale+Arch": BASE_ARCH,
    "Scale+Recipe": BASE_RECIPE,
    "Full": sorted(set(BASE_ARCH + BASE_RECIPE + MEASUREMENT)),
}


def design_matrix(df: pd.DataFrame, spec_cols: list[str], pooled: bool) -> tuple[pd.DataFrame, list[str]]:
    X = df[spec_cols].copy()
    if pooled:
        meta_cols = ["saturation_2026", "n_choices", "output_space", "contamination_resistant", "retained_v22"]
        for c in meta_cols:
            if c in df:
                X[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
        bench_dum = pd.get_dummies(df["benchmark"], prefix="bench", dtype=float)
        source_dum = pd.get_dummies(df["source"], prefix="src", dtype=float)
        # Recipe x benchmark-property interactions: these are the central v4
        # model-formulation fix; they let reasoning/math/code recipes matter
        # differently for contaminated/saturated/open-form tasks.
        for r in ["is_reasoning_model", "math_intensity", "code_pretrain", "synthetic_quality"]:
            if r in df:
                X[f"{r}_x_contres"] = df[r].astype(float) * df["contamination_resistant"].astype(float)
                X[f"{r}_x_outputspace"] = df[r].astype(float) * df["output_space"].fillna(0).astype(float)
        X = pd.concat([X, bench_dum, source_dum], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float), list(X.columns)


def make_estimator(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
    if name == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000))
    raise ValueError(name)


def fit_predict_cv(df: pd.DataFrame, spec_cols: list[str], args, pooled: bool = False) -> tuple[np.ndarray, dict]:
    if len(df) == 0 or df["family"].nunique() < 2:
        return np.full(len(df), np.nan), {"status": "too_sparse", "n": len(df), "n_families": int(df["family"].nunique())}
    y_raw = df["score"].to_numpy(float)
    y_fit = pct_to_logit(y_raw) if args.target == "logit" else y_raw
    X, cols = design_matrix(df, spec_cols, pooled=pooled)
    Xv = X.to_numpy(float)
    weights = df["sample_weight"].to_numpy(float)
    folds = grouped_family_folds(df["family"].to_numpy(), n_folds=args.n_folds)
    pred_fit = np.full(len(df), np.nan)
    fold_mae = []
    for fold in sorted(set(folds)):
        te = folds == fold
        tr = ~te
        # Keep enough rows for regularized fit; otherwise skip the fold rather
        # than using zero sentinels.
        if te.sum() == 0 or tr.sum() < max(6, min(10, len(cols) // 2)):
            continue
        est = make_estimator(args.estimator)
        try:
            if args.estimator == "ridge":
                est.fit(Xv[tr], y_fit[tr], ridge__sample_weight=weights[tr])
            else:
                est.fit(Xv[tr], y_fit[tr], huberregressor__sample_weight=weights[tr])
        except TypeError:
            est.fit(Xv[tr], y_fit[tr])
        pred_fit[te] = est.predict(Xv[te])
        pred_pct_fold = logit_to_pct(pred_fit[te]) if args.target == "logit" else pred_fit[te]
        fold_mae.append(float(np.average(np.abs(y_raw[te] - pred_pct_fold), weights=weights[te])))
    ok = ~np.isnan(pred_fit)
    pred_pct = np.full(len(df), np.nan)
    pred_pct[ok] = logit_to_pct(pred_fit[ok]) if args.target == "logit" else pred_fit[ok]
    status = "ok" if ok.sum() == len(df) else ("partial" if ok.any() else "no_scored_folds")
    mae = float(np.average(np.abs(y_raw[ok] - pred_pct[ok]), weights=weights[ok])) if ok.any() else np.nan
    return pred_pct, {"status": status, "mae": mae, "mae_std": float(np.std(fold_mae)) if fold_mae else np.nan,
                      "n": len(df), "n_families": int(df["family"].nunique()), "n_scored": int(ok.sum())}


def per_benchmark_results(primary: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for bench in PRIMARY_BENCHMARKS_V4:
        sub = primary[primary["benchmark"] == bench].reset_index(drop=True)
        meta = BENCHMARK_SUITE_V4[bench]
        row = {"benchmark": bench, "n": len(sub), "n_families": int(sub["family"].nunique()) if len(sub) else 0,
               "contamination_resistant": int(meta["contamination_resistant"]), "retained_v22": int(meta["retained_v22"]),
               "format": meta["format"], "source_policy": args.source_policy}
        maes = {}
        for spec, cols in SPECS.items():
            if len(sub) < args.min_models or sub["family"].nunique() < 2:
                row[f"{spec}_mae"] = np.nan
                row[f"{spec}_status"] = "too_sparse"
            else:
                pred, stat = fit_predict_cv(sub, cols, args, pooled=False)
                row[f"{spec}_mae"] = stat["mae"]
                row[f"{spec}_status"] = stat["status"]
            maes[spec] = row[f"{spec}_mae"]
        valid = {k: v for k, v in maes.items() if not pd.isna(v)}
        row["best"] = min(valid, key=valid.get) if valid else "NA"
        if not pd.isna(row["Scale_mae"]) and row["Scale_mae"] != 0 and not pd.isna(row["Scale+Recipe_mae"]):
            row["recipe_impact_pct"] = 100.0 * (row["Scale_mae"] - row["Scale+Recipe_mae"]) / row["Scale_mae"]
        else:
            row["recipe_impact_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_results(primary: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = primary[["model", "family", "benchmark", "score", "source", "sample_weight"]].copy().reset_index(drop=True)
    summary = []
    for spec, cols in SPECS.items():
        pred, stat = fit_predict_cv(primary.reset_index(drop=True), cols, args, pooled=True)
        preds[f"pred_{spec}"] = pred
        preds[f"abs_err_{spec}"] = np.abs(preds["score"] - pred)
        row = {"spec": spec, **stat}
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    return summary_df, preds


def coverage_tables(all_cells: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = all_cells.groupby("benchmark").agg(
        rows=("score", "size"), models=("model", "nunique"), official_rows=("source", lambda s: int((s == "official").sum())),
        avg_source_weight=("source_confidence_weight", "mean"), metric_type=("metric_type", "first"),
        include_primary=("include_primary", "max"),
    ).reset_index().sort_values(["include_primary", "models"], ascending=[False, False])
    src = all_cells.groupby(["source", "metric_type"]).size().reset_index(name="rows").sort_values("rows", ascending=False)
    return cov, src



def latex_escape(x) -> str:
    s = str(x)
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('_', r'\_')
             .replace('#', r'\#')
             .replace('{', r'\{')
             .replace('}', r'\}'))

def write_latex_tables(per_bench: pd.DataFrame, pooled_summary: pd.DataFrame, cov: pd.DataFrame, args) -> None:
    # Compact tables intentionally, to keep the report readable.
    pb = per_bench.copy()
    for c in ["Scale_mae", "Scale+Recipe_mae", "Full_mae", "recipe_impact_pct"]:
        pb[c] = pb[c].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")
    rows = []
    lb = r"\\"
    for _, r in pb.iterrows():
        rows.append(f"{latex_escape(r['benchmark'])} & {int(r['n'])} & {int(r['n_families'])} & {r['Scale_mae']} & {r['Scale+Recipe_mae']} & {r['Full_mae']} & {r['recipe_impact_pct']} & {latex_escape(r['best'])} " + lb)
    table1 = "\n".join(rows)

    ps = pooled_summary.copy()
    ps["mae"] = ps["mae"].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")
    ps["n_scored"] = ps["n_scored"].fillna(0).astype(int)
    rows2 = [f"{latex_escape(r['spec'])} & {int(r['n'])} & {int(r['n_families'])} & {int(r['n_scored'])} & {r['mae']} & {latex_escape(r['status'])} " + lb for _, r in ps.iterrows()]

    cv = cov.copy().head(18)
    cv["avg_source_weight"] = cv["avg_source_weight"].map(lambda x: f"{x:.2f}")
    rows3 = [f"{latex_escape(r['benchmark'])} & {int(r['models'])} & {int(r['official_rows'])} & {latex_escape(r['metric_type'])} & {r['avg_source_weight']} " + lb for _, r in cv.iterrows()]

    tex = rf"""
% Auto-generated by analysis_core.py
\newcommand{{\VFourSourcePolicy}}{{{args.source_policy}}}
\newcommand{{\VFourTarget}}{{{args.target}}}
\newcommand{{\VFourScalePolicy}}{{{args.scale_policy}}}
\newcommand{{\VFourNCells}}{{{len(per_bench)}}}
\newcommand{{\VFourPooledTable}}{{%
\begin{{tabular}}{{lrrrrl}}\toprule
Spec & cells & families & scored & weighted MAE & status \\
\midrule
{chr(10).join(rows2)}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFourPerBenchmarkTable}}{{%
\begin{{tabular}}{{lrrrrrrl}}\toprule
Benchmark & n & fam & Scale & Scale+Recipe & Full & Impact\% & Best \\
\midrule
{table1}
\bottomrule\end{{tabular}}}}
\newcommand{{\VFourCoverageTable}}{{%
\begin{{tabular}}{{lrrlr}}\toprule
Benchmark & models & official rows & metric & avg wt \\
\midrule
{chr(10).join(rows3)}
\bottomrule\end{{tabular}}}}
"""
    (OUT / "auto_tables_v4.tex").write_text(tex)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-policy", choices=SOURCE_POLICIES.keys(), default="official_like")
    p.add_argument("--min-source-weight", type=float, default=0.0)
    p.add_argument("--scale-policy", choices=["continuous", "bucket"], default="continuous")
    p.add_argument("--target", choices=["raw", "logit"], default="logit")
    p.add_argument("--estimator", choices=["ridge", "huber"], default="ridge")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-models", type=int, default=10)
    args = p.parse_args()

    primary, all_cells = prepare_cell_df(args)
    per_bench = per_benchmark_results(primary, args)
    pooled, preds = pooled_results(primary, args)
    cov, src = coverage_tables(all_cells, primary)

    suffix = f"{args.source_policy}_{args.scale_policy}_{args.target}_{args.estimator}"
    per_bench.to_csv(OUT / f"per_benchmark_results_{suffix}.csv", index=False)
    pooled.to_csv(OUT / f"pooled_results_{suffix}.csv", index=False)
    preds.to_csv(OUT / f"pooled_oof_predictions_{suffix}.csv", index=False)
    cov.to_csv(OUT / f"coverage_{suffix}.csv", index=False)
    src.to_csv(OUT / f"source_counts_{suffix}.csv", index=False)

    valid = per_bench.dropna(subset=["recipe_impact_pct"])
    if len(valid) >= 4:
        sub = valid["benchmark"].map(SUBSTITUTABILITY_HAND_V4).astype(float)
        rho, pval = sp_stats.spearmanr(sub, valid["recipe_impact_pct"])
    else:
        rho, pval = np.nan, np.nan
    pooled_best = pooled.dropna(subset=["mae"]).sort_values("mae").head(1)
    summary = {
        "source_policy": args.source_policy,
        "scale_policy": args.scale_policy,
        "target": args.target,
        "estimator": args.estimator,
        "n_primary_cells": int(len(primary)),
        "n_models_primary": int(primary["model"].nunique()),
        "n_families_primary": int(primary["family"].nunique()),
        "n_benchmarks_primary": int(primary["benchmark"].nunique()),
        "source_counts": all_cells["source"].value_counts().to_dict(),
        "pooled_best_spec": pooled_best["spec"].iloc[0] if not pooled_best.empty else None,
        "pooled_best_mae": float(pooled_best["mae"].iloc[0]) if not pooled_best.empty else None,
        "recipe_impact_spearman_substitutability": float(rho) if not pd.isna(rho) else None,
        "recipe_impact_spearman_p": float(pval) if not pd.isna(pval) else None,
        "gemini_models_included": sorted([m for m in all_cells["model"].unique() if "Gemini" in m]),
    }
    (OUT / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2))
    # Also write a canonical summary for the paper.
    if args.source_policy == "official_like" and args.scale_policy == "continuous" and args.target == "logit" and args.estimator == "ridge":
        (OUT / "summary_v4.json").write_text(json.dumps(summary, indent=2))
        write_latex_tables(per_bench, pooled, cov, args)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
