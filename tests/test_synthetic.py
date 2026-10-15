from pathlib import Path
import json

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_alignment.synthetic import generate_synthetic_benchmark


def test_generates_reproducible_manifest_and_images(tmp_path: Path) -> None:
    layout = A4CalibrationLayout(dpi=72.0)
    manifest_path = generate_synthetic_benchmark(
        tmp_path / "cases",
        layout=layout,
        count=3,
        seed=1234,
        canvas_size=(620, 840),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 3
    assert manifest["expected_valid_count"] == 3
    assert all((manifest_path.parent / case["image"]).is_file() for case in manifest["cases"])
