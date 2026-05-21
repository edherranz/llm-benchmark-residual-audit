#!/usr/bin/env python3
"""Generate LaTeX tables/macros for paper/main.tex from outputs/*.csv."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = ROOT / "paper"


def esc(x) -> str:
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


def fmt(x, n=2) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.{n}f}"
    except Exception:
        return esc(x)


def rows(df: pd.DataFrame, cols: list[str], fmts: dict | None = None) -> str:
    fmts = fmts or {}
    out = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            val = r[c]
            if c in fmts:
                val = fmts[c](val)
            elif isinstance(val, float):
                val = fmt(val, 2)
            elif isinstance(val, int):
                val = str(val)
            else:
                val = esc(val)
            cells.append(str(val))
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out)


def main() -> None:
    summary = json.load(open(OUT / "summary.json"))
    pooled = pd.read_csv(OUT / "pooled_results_canonical.csv")
    pooled["mae_f"] = pooled["mae"].map(lambda x: fmt(x, 2))
    pooled["sd_f"] = pooled["mae_std"].map(lambda x: fmt(x, 2))
    ab_rows = rows(pooled, ["spec", "n", "n_families", "n_scored", "mae_f", "sd_f", "status"], {"spec": esc, "status": esc})

    rs = pd.read_csv(OUT / "residual_summary_full.csv").iloc[0]
    res_summary = "\n".join([
        rf"Cells & {int(rs['n'])} \\",
        rf"Weighted MAE & {rs['weighted_mae']:.2f} pp \\",
        rf"Weighted RMSE & {rs['weighted_rmse']:.2f} pp \\",
        rf"Weighted bias & {rs['weighted_bias']:.2f} pp \\",
        rf"Weighted $R^2$ & {rs['weighted_r2']:.2f} \\",
        rf"Calibration slope & {rs['calibration_slope_actual_on_pred']:.2f} \\",
        rf"Calibration intercept & {rs['calibration_intercept_actual_on_pred']:.2f} \\",
    ])

    val = pd.read_csv(OUT / "validation_regime_summary.csv")
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias", "weighted_r2"]:
        val[c + "_f"] = val[c].map(lambda x: fmt(x, 2))
    val_rows = rows(val, ["validation", "n", "models", "families", "benchmarks", "weighted_mae_f", "weighted_rmse_f", "weighted_bias_f", "weighted_r2_f"], {"validation": esc})

    br = pd.read_csv(OUT / "residual_by_benchmark.csv")
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias", "p90_abs_error"]:
        br[c + "_f"] = br[c].map(lambda x: fmt(x, 2))
    bench_rows = rows(br, ["benchmark", "n", "models", "weighted_mae_f", "weighted_rmse_f", "weighted_bias_f", "p90_abs_error_f"], {"benchmark": esc})

    fr = pd.read_csv(OUT / "residual_by_family.csv").head(10)
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        fr[c + "_f"] = fr[c].map(lambda x: fmt(x, 2))
    fam_rows = rows(fr, ["family", "n", "models", "weighted_mae_f", "weighted_bias_f", "p90_abs_error_f"], {"family": esc})

    sr = pd.read_csv(OUT / "residual_by_source.csv")
    for c in ["weighted_mae", "weighted_bias", "p90_abs_error"]:
        sr[c + "_f"] = sr[c].map(lambda x: fmt(x, 2))
    src_rows = rows(sr, ["source", "n", "models", "weighted_mae_f", "weighted_bias_f", "p90_abs_error_f"], {"source": esc})

    oo = pd.read_csv(OUT / "residual_outliers_top30.csv").head(15)
    for c in ["score", "prediction", "residual"]:
        oo[c + "_f"] = oo[c].map(lambda x: fmt(x, 1))
    out_rows = rows(oo, ["model", "benchmark", "score_f", "prediction_f", "residual_f", "source"], {"model": esc, "benchmark": esc, "source": esc})

    sg = pd.read_csv(OUT / "sensitivity_grid.csv")
    sg = sg[sg["spec"] == "Full"].sort_values("mae").head(12).copy()
    sg["mae_f"] = sg["mae"].map(lambda x: fmt(x, 2))
    sens_rows = rows(sg, ["source_policy", "scale_policy", "target", "estimator", "n", "families", "mae_f", "status"],
                     {"source_policy": esc, "scale_policy": esc, "target": esc, "estimator": esc, "status": esc})

    cf = pd.read_csv(OUT / "full_model_standardized_coefficients.csv").head(20)
    cf["coef_f"] = cf["standardized_coefficient"].map(lambda x: fmt(x, 3))
    coef_rows = rows(cf, ["feature", "feature_group", "coef_f"], {"feature": esc, "feature_group": esc})

    ms = pd.read_csv(OUT / "missingness_summary.csv").iloc[0]
    missing_summary = "\n".join([
        rf"Possible model-benchmark cells & {int(ms['possible_cells'])} \\",
        rf"Observed canonical cells & {int(ms['observed_cells'])} \\",
        rf"Coverage & {ms['coverage_pct']:.1f}\% \\",
        rf"Descriptive missingness AUC & {ms['missingness_auc_descriptive']:.2f} \\",
    ])

    ss = pd.read_csv(OUT / "parameter_uncertainty_summary.csv").iloc[0]
    stress_summary = "\n".join([
        rf"Iterations & {int(ss['iterations'])} \\",
        rf"Mean MAE & {ss['mae_mean']:.2f} pp \\",
        rf"MAE standard deviation & {ss['mae_std']:.2f} pp \\",
        rf"5th--95th percentile MAE & {ss['mae_p05']:.2f}--{ss['mae_p95']:.2f} pp \\",
    ])

    bs = pd.read_csv(OUT / "bootstrap_mae_summary.csv").iloc[0]
    bootstrap_summary = "\n".join([
        rf"Bootstrap iterations & {int(bs['iterations'])} \\",
        rf"Full-model MAE & {bs['full_mae_point']:.2f} pp \\",
        rf"Full-model 95\% CI & {bs['full_mae_ci_low']:.2f}--{bs['full_mae_ci_high']:.2f} pp \\",
        rf"Scale-only MAE & {bs['scale_mae_point']:.2f} pp \\",
        rf"Improvement over scale-only & {bs['improvement_point']:.2f} pp \\",
        rf"Improvement 95\% CI & {bs['improvement_ci_low']:.2f}--{bs['improvement_ci_high']:.2f} pp \\",
        rf"Bootstrap Pr(improvement $\leq$ 0) & {bs['p_improvement_le_zero']:.3f} \\",
    ])

    rc = pd.read_csv(OUT / "recipe_coding_sensitivity_summary.csv").iloc[0]
    recipe_sensitivity_summary = "\n".join([
        rf"Perturbation iterations & {int(rc['iterations'])} \\",
        rf"Mean MAE & {rc['mae_mean']:.2f} pp \\",
        rf"MAE standard deviation & {rc['mae_std']:.2f} pp \\",
        rf"5th--95th percentile MAE & {rc['mae_p05']:.2f}--{rc['mae_p95']:.2f} pp \\",
        rf"Min--max MAE & {rc['mae_min']:.2f}--{rc['mae_max']:.2f} pp \\",
    ])

    tmp = pd.read_csv(OUT / "temporal_holdout.csv")
    for c in ["weighted_mae", "weighted_rmse", "weighted_bias"]:
        tmp[c + "_f"] = tmp[c].map(lambda x: fmt(x, 2))
    tmp["cutoff_f"] = tmp["cutoff_calendar_year"].map(lambda x: fmt(x, 2))
    temp_rows = rows(tmp, ["cutoff_f", "train_cells", "test_cells", "test_models", "weighted_mae_f", "weighted_rmse_f", "weighted_bias_f"])

    temporal_mae = val[val.validation == "temporal-holdout"]["weighted_mae"].iloc[0]
    lobo_mae = val[val.validation == "leave-one-benchmark-out"]["weighted_mae"].iloc[0]

    tex = rf'''
% Auto-generated from outputs/*.csv. Do not edit manually.
\newcommand{{\CanonicalCells}}{{{summary['n_primary_cells']}}}
\newcommand{{\CanonicalModels}}{{{summary['n_models_primary']}}}
\newcommand{{\CanonicalFamilies}}{{{summary['n_families_primary']}}}
\newcommand{{\CanonicalBenchmarks}}{{{summary['n_benchmarks_primary']}}}
\newcommand{{\CanonicalMAE}}{{{summary['weighted_mae']:.2f}}}
\newcommand{{\CanonicalRMSE}}{{{summary['weighted_rmse']:.2f}}}
\newcommand{{\CanonicalBias}}{{{summary['weighted_bias']:.2f}}}
\newcommand{{\CanonicalRTwo}}{{{summary['weighted_r2']:.2f}}}
\newcommand{{\CalibrationSlope}}{{{summary['calibration_slope']:.2f}}}
\newcommand{{\TemporalMAE}}{{{temporal_mae:.2f}}}
\newcommand{{\LOBOMAE}}{{{lobo_mae:.2f}}}
\newcommand{{\CoveragePct}}{{{summary['missingness_coverage_pct']:.1f}}}
\newcommand{{\StressMAELow}}{{{summary['stress_mae_p05']:.2f}}}
\newcommand{{\StressMAEHigh}}{{{summary['stress_mae_p95']:.2f}}}
\newcommand{{\BootstrapFullMAELow}}{{{summary['bootstrap_full_mae_ci_low']:.2f}}}
\newcommand{{\BootstrapFullMAEHigh}}{{{summary['bootstrap_full_mae_ci_high']:.2f}}}
\newcommand{{\BootstrapImprovementLow}}{{{summary['bootstrap_improvement_ci_low']:.2f}}}
\newcommand{{\BootstrapImprovementHigh}}{{{summary['bootstrap_improvement_ci_high']:.2f}}}
\newcommand{{\RecipeSensitivityLow}}{{{summary['recipe_sensitivity_mae_p05']:.2f}}}
\newcommand{{\RecipeSensitivityHigh}}{{{summary['recipe_sensitivity_mae_p95']:.2f}}}
\newcommand{{\AblationTable}}{{%
\begin{{tabular}}{{lrrrrrl}}\toprule
Specification & cells & families & scored & MAE & fold sd & status \\
\midrule
{ab_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\ResidualSummaryTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{res_summary}
\bottomrule\end{{tabular}}}}
\newcommand{{\ValidationTable}}{{%
\begin{{tabular}}{{lrrrrrrrr}}\toprule
Validation & cells & models & families & benchmarks & MAE & RMSE & bias & $R^2$ \\
\midrule
{val_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\BenchmarkResidualTable}}{{%
\begin{{tabular}}{{lrrrrrr}}\toprule
Benchmark & cells & models & MAE & RMSE & bias & p90 AE \\
\midrule
{bench_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\FamilyResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Family & cells & models & MAE & bias & p90 AE \\
\midrule
{fam_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\SourceResidualTable}}{{%
\begin{{tabular}}{{lrrrrr}}\toprule
Source & cells & models & MAE & bias & p90 AE \\
\midrule
{src_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\OutlierTable}}{{%
\begin{{tabular}}{{llrrrl}}\toprule
Model & Benchmark & observed & predicted & residual & source \\
\midrule
{out_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\SensitivityTable}}{{%
\begin{{tabular}}{{llllrrrl}}\toprule
Source policy & scale & target & estimator & cells & families & MAE & status \\
\midrule
{sens_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\CoefficientTable}}{{%
\begin{{tabular}}{{llr}}\toprule
Feature & group & standardized coefficient \\
\midrule
{coef_rows}
\bottomrule\end{{tabular}}}}
\newcommand{{\MissingSummaryTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{missing_summary}
\bottomrule\end{{tabular}}}}
\newcommand{{\StressTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{stress_summary}
\bottomrule\end{{tabular}}}}
\newcommand{{\BootstrapTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{bootstrap_summary}
\bottomrule\end{{tabular}}}}
\newcommand{{\RecipeSensitivityTable}}{{%
\begin{{tabular}}{{lr}}\toprule
Diagnostic & Value \\
\midrule
{recipe_sensitivity_summary}
\bottomrule\end{{tabular}}}}
\newcommand{{\TemporalTable}}{{%
\begin{{tabular}}{{rrrrrrr}}\toprule
Cutoff yr & train cells & test cells & test models & MAE & RMSE & bias \\
\midrule
{temp_rows}
\bottomrule\end{{tabular}}}}
'''
    (PAPER / "generated_tables.tex").write_text(tex)
    print(f"Wrote {PAPER / 'generated_tables.tex'}")


if __name__ == "__main__":
    main()
