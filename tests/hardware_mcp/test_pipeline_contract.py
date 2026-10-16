"""A real PrintGPT approval export satisfies the hardware print contract."""

import json

from printgpt_hardware_mcp.config import HardwareConfig
from printgpt_mvp.example import run_example

from hw_fakes import FakePrinter, FakeScanner, make_service


class _RealTime:
    def __call__(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


def test_exported_toner_page_validates_and_prepares(tmp_path):
    artifacts = tmp_path / "artifacts"
    pdf_path = run_example(artifacts / "demo")
    print_dir = pdf_path.parent
    manifest = json.loads((print_dir / "print_approval.json").read_text(encoding="utf-8"))
    assert manifest["approval"]["state"] == "approved" and manifest["final_toner_only"] is True
    assert manifest["validation"] == {"accepted": True, "deterministic": True, "reasons": [], "simulated": False}

    config = HardwareConfig(artifacts_root=artifacts, scans_dir=tmp_path / "Scans")
    printer = FakePrinter()
    service = make_service(config, printer, FakeScanner(), _RealTime())
    for name in ("approved_toner_only.png", "approved_a4_actual_size.pdf"):
        report = service.validate_print_artifact(str(print_dir / name))
        assert report.valid, report.errors
    prepared = service.prepare_print_job(str(print_dir / "approved_toner_only.png"))
    assert prepared.artifact_sha256 == manifest["artifacts"]["toner_png"]["sha256"]
    # The unfinished source page next to the run is never printable.
    source = next((artifacts / "demo" / "input").iterdir())
    assert not service.validate_print_artifact(str(source)).valid
    assert printer.submitted == []
