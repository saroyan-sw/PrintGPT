"""Deterministic checks that an artifact is an approved, toner-only A4 page.

Validation reads only the artifact, its sibling ``print_approval.json`` and file
metadata. It returns structured findings and, when valid, the exact binary
raster that would be printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ARTIFACT_EXTENSIONS, HardwareConfig
from .models import ArtifactCheck, ArtifactValidation, HardwareError, PrintSettings

APPROVAL_SCHEMA = "printgpt-print-approval/1"
APPROVAL_FILENAME = "print_approval.json"
A4_MM = (210.0, 297.0)
A4_300DPI_PX = (2480, 3508)
PAGE_TOLERANCE_MM = 0.5
MAX_INK_COVERAGE = 0.60
MAX_GRAY_FRACTION_PDF = 0.005
CLOCK_SKEW = timedelta(minutes=5)

Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse decompression bombs well above A4 @ 600 DPI


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed(raw: str | Path, roots: list[Path], *, what: str) -> Path:
    """Resolve symlinks and ``..`` first, then require the target to sit inside a root."""
    if raw is None or str(raw).strip() == "" or "\x00" in str(raw):
        raise HardwareError("invalid_path", f"{what} path is empty or malformed")
    try:
        resolved = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HardwareError("not_found", f"{what} does not exist") from exc
    if not resolved.is_file():
        raise HardwareError("not_a_file", f"{what} is not a regular file")
    if not any(_inside(resolved, root) for root in roots):
        raise HardwareError("outside_allowlist", f"{what} is outside the allowed directories")
    return resolved


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    report: ArtifactValidation
    raster: Image.Image | None


class _Findings:
    def __init__(self) -> None:
        self.checks: list[ArtifactCheck] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, passed: bool, detail: str) -> bool:
        self.checks.append(ArtifactCheck(name=name, passed=passed, detail=detail))
        if not passed:
            self.errors.append(f"{name}: {detail}")
        return passed


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HardwareError("manifest_missing", f"no {APPROVAL_FILENAME} next to the artifact; approve and export the candidate first")
    if path.stat().st_size > 1_000_000:
        raise HardwareError("manifest_invalid", "approval manifest is unexpectedly large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HardwareError("manifest_invalid", "approval manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise HardwareError("manifest_invalid", "approval manifest must be a JSON object")
    return payload


def _binary_raster(pixels: np.ndarray) -> np.ndarray:
    return np.where(pixels < 128, 0, 255).astype(np.uint8)


def _check_png(path: Path, f: _Findings) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
            dpi = image.info.get("dpi")
            mode = image.mode
            pixels = np.asarray(image.convert("L"))
    except Exception:
        f.check("decode", False, "PNG could not be decoded")
        return None, None
    f.check("raster_size", size == A4_300DPI_PX, f"{size[0]} x {size[1]} px (required {A4_300DPI_PX[0]} x {A4_300DPI_PX[1]})")
    if dpi is None:
        f.warnings.append("PNG has no DPI metadata; printing uses 300 DPI by contract")
    else:
        f.check("raster_dpi", all(abs(float(v) - 300.0) < 0.5 for v in dpi[:2]), f"{float(dpi[0]):.1f} x {float(dpi[1]):.1f} DPI")
    unique = np.unique(pixels)
    f.check("toner_only_pixels", set(unique.tolist()) <= {0, 255} and mode in ("L", "1"),
            "pure black/white grayscale raster" if set(unique.tolist()) <= {0, 255}
            else f"{len(unique)} grey levels found (an original scan or preview is not printable)")
    return pixels, size


def _check_pdf(path: Path, f: _Findings) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = len(reader.pages)
        page = reader.pages[0]
        width_mm = float(page.mediabox.width) * 25.4 / 72.0
        height_mm = float(page.mediabox.height) * 25.4 / 72.0
        rotation = int(page.rotation) % 360
    except Exception:
        f.check("decode", False, "PDF could not be parsed")
        return None, None
    f.check("page_count", pages == 1, f"{pages} page(s) (exactly one required)")
    size_ok = abs(width_mm - A4_MM[0]) <= PAGE_TOLERANCE_MM and abs(height_mm - A4_MM[1]) <= PAGE_TOLERANCE_MM
    f.check("page_size", size_ok, f"{width_mm:.2f} x {height_mm:.2f} mm (required 210 x 297 mm portrait)")
    f.check("page_rotation", rotation == 0, f"/Rotate {rotation}")
    if not (pages == 1 and size_ok and rotation == 0):
        return None, (width_mm, height_mm)
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        try:
            bitmap = document[0].render(scale=300 / 72.0)
            pixels = np.asarray(bitmap.to_pil().convert("L")).copy()
        finally:
            document.close()
    except Exception:
        f.check("render", False, "PDF page could not be rendered")
        return None, (width_mm, height_mm)
    rendered = (pixels.shape[1], pixels.shape[0])
    shape_ok = all(abs(a - b) <= 1 for a, b in zip(rendered, A4_300DPI_PX))
    f.check("render_size", shape_ok, f"rendered {rendered[0]} x {rendered[1]} px at 300 DPI")
    if shape_ok and rendered != A4_300DPI_PX:
        # PDF points do not map to whole 300-DPI pixels; snap the 1 px rounding back.
        pixels = np.asarray(Image.fromarray(pixels).resize(A4_300DPI_PX, Image.Resampling.NEAREST))
        f.warnings.append(f"PDF rendered at {rendered[0]} x {rendered[1]} px; snapped to 2480 x 3508 (sub-0.1 mm change). Prefer the PNG artifact.")
    gray = float(np.mean((pixels > 32) & (pixels < 224)))
    f.check("toner_only_pixels", gray <= MAX_GRAY_FRACTION_PDF,
            f"{gray * 100:.2f}% mid-grey pixels (<= {MAX_GRAY_FRACTION_PDF * 100:.1f}% allowed for a toner-only page)")
    return (pixels if shape_ok else None), (width_mm, height_mm)


def _check_manifest(manifest: dict[str, Any], artifact: Path, manifest_path: Path, sha: str,
                    config: HardwareConfig, now: datetime, f: _Findings) -> dict[str, Any] | None:
    f.check("manifest_schema", manifest.get("schema_version") == APPROVAL_SCHEMA,
            f"schema {manifest.get('schema_version')!r}")
    for key in ("run_id", "selected_candidate_id"):
        f.check(f"manifest_{key}", isinstance(manifest.get(key), str) and bool(manifest.get(key)), "present" if manifest.get(key) else "missing")
    approval = manifest.get("approval") if isinstance(manifest.get("approval"), dict) else {}
    f.check("explicit_approval", approval.get("state") == "approved", f"approval state {approval.get('state')!r}")
    approved_at = _parse_time(approval.get("approved_at"))
    if approved_at is None:
        f.check("approval_timestamp", False, "missing or unparseable approval timestamp")
    else:
        age = now - approved_at
        fresh = -CLOCK_SKEW <= age <= timedelta(hours=config.manifest_max_age_hours)
        f.check("approval_timestamp", fresh, f"approved {age.total_seconds() / 3600:.2f} h ago (limit {config.manifest_max_age_hours:g} h)")
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
    f.check("validation_accepted", validation.get("accepted") is True, f"accepted={validation.get('accepted')!r}")
    f.check("validation_not_simulated", validation.get("simulated") is False, f"simulated={validation.get('simulated')!r}")
    f.check("final_toner_only", manifest.get("final_toner_only") is True and manifest.get("contains_source_page") is False,
            "manifest declares a toner-only page without the source document"
            if manifest.get("final_toner_only") is True and manifest.get("contains_source_page") is False
            else "manifest does not declare a final toner-only page")
    f.check("input_mode_recorded", isinstance(manifest.get("input_mode"), dict), "input-mode metadata present"
            if isinstance(manifest.get("input_mode"), dict) else "input-mode metadata missing")
    requested = manifest.get("requested_print_settings")
    if not isinstance(requested, dict):
        f.check("requested_settings", False, "requested print settings missing")
    else:
        expected = PrintSettings().model_dump()
        mismatched = {k: requested.get(k) for k, v in expected.items() if requested.get(k) != v}
        f.check("requested_settings", not mismatched and requested.get("pages", 1) == 1,
                "A4 portrait, 1 copy, simplex, monochrome, 100%, no fit" if not mismatched else f"unsupported settings {mismatched}")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    entry = None
    run_dir = manifest_path.parent.parent
    for candidate in artifacts.values():
        if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
            continue
        try:
            if (run_dir / candidate["path"]).resolve() == artifact:
                entry = candidate
        except OSError:
            continue
    if not f.check("artifact_in_manifest", entry is not None, "artifact is listed in the approval manifest"
                   if entry is not None else "artifact is not one of the approved exported files"):
        return None
    f.check("sha256_matches", entry.get("sha256") == sha, "file hash matches the approved hash"
            if entry.get("sha256") == sha else "file changed after approval (SHA-256 mismatch)")
    return entry


def _margin_check(pixels: np.ndarray, margins_mm: tuple[float, float, float, float] | None, f: _Findings) -> None:
    if margins_mm is None:
        f.warnings.append("printer margins unknown; ink near the paper edge may be clipped")
        return
    height, width = pixels.shape
    left, top, right, bottom = (int(np.ceil(m * 300 / 25.4)) for m in margins_mm)
    ink = pixels == 0
    inside = ink[top:height - bottom, left:width - right].sum()
    clipped = int(ink.sum() - inside)
    f.check("printable_area", clipped == 0, "all ink inside the printer's printable area" if clipped == 0
            else f"{clipped} ink pixels fall in the {margins_mm[0]:.2f} mm non-printable margin and would be clipped")


def validate_artifact(raw_path: str | Path, config: HardwareConfig, *, now: datetime | None = None,
                      printer_margins_mm: tuple[float, float, float, float] | None = None) -> ValidatedArtifact:
    """Validate an artifact for printing. Never raises for content problems."""
    now = now or datetime.now(timezone.utc)
    f = _Findings()
    try:
        artifact = resolve_allowed(raw_path, [config.artifacts_root], what="artifact")
    except HardwareError as exc:
        return ValidatedArtifact(ArtifactValidation(valid=False, errors=[exc.message]), None)
    base = dict(artifact_path=str(artifact))
    if _inside(artifact, config.scans_dir) or _inside(artifact, config.imports_root) or "input" in artifact.relative_to(config.artifacts_root).parts:
        return ValidatedArtifact(ArtifactValidation(valid=False, errors=["original scans and run inputs are never printable"], **base), None)
    suffix = artifact.suffix.lower()
    if suffix not in ARTIFACT_EXTENSIONS:
        return ValidatedArtifact(ArtifactValidation(valid=False, errors=[f"unsupported format {suffix or '(none)'}; only approved PNG or PDF"], **base), None)
    manifest_path = artifact.parent / APPROVAL_FILENAME
    sha = file_sha256(artifact)
    base.update(manifest_path=str(manifest_path), sha256=sha, artifact_kind=suffix[1:])
    try:
        manifest = _load_manifest(manifest_path)
    except HardwareError as exc:
        return ValidatedArtifact(ArtifactValidation(valid=False, errors=[exc.message], **base), None)
    _check_manifest(manifest, artifact, manifest_path, sha, config, now, f)

    page_mm: tuple[float, float] | None = None
    pixel_size = None
    if suffix == ".png":
        pixels, pixel_size = _check_png(artifact, f)
        if pixel_size == A4_300DPI_PX:
            page_mm = A4_MM
    else:
        pixels, page_mm = _check_pdf(artifact, f)
        if pixels is not None:
            pixel_size = (pixels.shape[1], pixels.shape[0])
    coverage = None
    raster = None
    if pixels is not None and (pixels.shape[1], pixels.shape[0]) == A4_300DPI_PX:
        binary = _binary_raster(pixels)
        coverage = float(np.mean(binary == 0))
        f.check("has_ink", coverage > 0, f"{coverage * 100:.3f}% of the page receives toner")
        f.check("plausible_coverage", coverage <= MAX_INK_COVERAGE,
                f"coverage {coverage * 100:.1f}% (limit {MAX_INK_COVERAGE * 100:.0f}%; a full scan is not a toner-only layer)")
        _margin_check(binary, printer_margins_mm, f)
        raster = Image.fromarray(binary, mode="L")
        raster.info["dpi"] = (300, 300)
    elif pixels is None and not f.errors:
        f.check("raster", False, "no printable raster could be derived")
    valid = not f.errors
    report = ArtifactValidation(
        valid=valid,
        run_id=manifest.get("run_id") if isinstance(manifest.get("run_id"), str) else None,
        candidate_id=manifest.get("selected_candidate_id") if isinstance(manifest.get("selected_candidate_id"), str) else None,
        page_size_mm=page_mm,
        pixel_size=pixel_size,
        ink_coverage_percent=None if coverage is None else round(coverage * 100, 4),
        checks=f.checks,
        errors=f.errors,
        warnings=f.warnings,
        **base,
    )
    return ValidatedArtifact(report, raster if valid else None)
