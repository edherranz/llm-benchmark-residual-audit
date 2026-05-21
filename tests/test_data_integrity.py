from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_provenance_required_columns_present():
    df = pd.read_csv(ROOT / "data" / "benchmark_provenance.csv")
    required = {
        "model", "benchmark", "score", "source", "source_confidence_weight",
        "source_detail", "exact_protocol_match_to_primary_suite", "score_scale"
    }
    missing = required.difference(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"


def test_primary_outputs_have_expected_scope():
    df = pd.read_csv(ROOT / "outputs" / "canonical_primary_cells.csv")
    assert len(df) >= 200
    assert df["model"].nunique() >= 35
    assert df["benchmark"].nunique() >= 10
    assert df["score"].between(0, 100).all()


def test_gemini_models_are_included():
    df = pd.read_csv(ROOT / "outputs" / "canonical_all_cells.csv")
    gemini = sorted(m for m in df["model"].unique() if "Gemini" in m)
    assert "Gemini-2.5-Pro" in gemini
    assert "Gemini-3.1-Pro" in gemini


def test_non_percent_livecodebench_pro_not_in_primary_percent_target():
    df = pd.read_csv(ROOT / "outputs" / "canonical_primary_cells.csv")
    assert "LiveCodeBench_Pro_Elo" not in set(df["benchmark"])
