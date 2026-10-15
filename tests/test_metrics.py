import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_alignment.metrics import evaluate_alignment


def test_exact_inverse_has_near_zero_error() -> None:
    layout = A4CalibrationLayout(dpi=150.0)
    canonical_to_source = np.asarray(
        [[0.72, 0.03, 120.0], [-0.02, 0.70, 85.0], [0.00002, -0.00001, 1.0]],
        dtype=np.float64,
    )
    metrics = evaluate_alignment(canonical_to_source, np.linalg.inv(canonical_to_source), layout)
    assert metrics.max_error_px < 0.001
    assert metrics.max_error_mm < 0.001
