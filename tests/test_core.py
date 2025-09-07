# tests/test_core.py
import numpy as np
import pandas as pd
from semantic_sense import AnomalyDetector


def _assert_common_output(out: pd.DataFrame, n_rows: int):
    assert isinstance(out, pd.DataFrame)
    # Required columns from your current core.py
    for col in ["row_text", "centroid_distance", "rank", "is_anomaly"]:
        assert col in out.columns, f"Missing column: {col}"

    # Length matches input rows
    assert len(out) == n_rows

    # Types / value checks
    assert out["centroid_distance"].dtype.kind in "fc"  # float
    assert set(out["is_anomaly"].unique()).issubset({0, 1})


def test_hybrid_mode_basic():
    # Mixed numeric + categorical
    df = pd.DataFrame({
        "A": [1, 2, 3, 100],     # numeric outlier in last row
        "B": ["x", "x", "x", "y"]  # mostly same category
    })
    det = AnomalyDetector(mode="hybrid", numeric_weight=1.0)
    # With 4 rows, 25% ≈ 1 anomaly flagged
    out = det.detect(df, top_percent=25.0)
    _assert_common_output(out, len(df))


def test_text_mode_basic():
    # Pure text columns
    df = pd.DataFrame({
        "A": ["foo", "bar", "baz"],
        "B": ["x", "y", "z"]
    })
    det = AnomalyDetector(mode="text")
    # With 3 rows, ~34% flags ≈ 1 row
    out = det.detect(df, top_percent=34.0)
    _assert_common_output(out, len(df))


def test_hybrid_vs_text_diff_scores():
    # Ensure hybrid (with numerics) produces different scoring than text-only
    df = pd.DataFrame({
        "A": [1, 2, 3, 100],      # numeric signal
        "B": ["x", "x", "x", "y"] # textual signal
    })

    det_text = AnomalyDetector(mode="text")
    out_text = det_text.detect(df, top_percent=25.0)

    det_hybrid = AnomalyDetector(mode="hybrid", numeric_weight=2.0)
    out_hybrid = det_hybrid.detect(df, top_percent=25.0)

    # Compare sorted distances to be robust to sorting inside detect()
    t_sorted = np.sort(out_text["centroid_distance"].to_numpy())
    h_sorted = np.sort(out_hybrid["centroid_distance"].to_numpy())

    # They should not be identical when numerics are added with weight
    assert not np.allclose(t_sorted, h_sorted), \
        "Hybrid and Text centroid distances are unexpectedly identical."
