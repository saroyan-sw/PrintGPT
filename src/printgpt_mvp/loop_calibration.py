"""Print-scan loop calibration: measure and correct the whole scan -> print loop, per printer.

PrintGPT scans a sheet, computes new ink in scan coordinates, and prints it
back onto the same sheet. For the ink to land in place, the combined error of
that loop must be corrected. It is measured directly, without a ruler:

* a target page of ArUco markers at known positions ``t`` (mm) is printed with
  the current correction ``C`` and scanned exactly like a drawing;
* the markers' scan positions ``o`` give the residual map ``R: t -> o``, an
  affine fit on half of the markers (the others are a holdout check);
* the loop map is ``R = L o C`` (``L`` = scanner after printer), so the next
  correction is ``C' = C o R^-1``, which makes ``L o C' = I``.

Each round prints the target with the latest correction, so the holdout error
of successive rounds shows the correction converging; once it stops improving,
what remains is the printer's sheet-to-sheet variation. A profile can be saved
only after a *corrected* round passes (holdout error below 0.5 mm), so it is
verified on paper, not just fitted. Profiles are stored per printer queue and
paper tray, so a demo can reuse them without calibrating again.

The correction is valid for the loop it was measured on: same scanner workflow
(Scanner A4, sheet placed the same way), same printer, tray and print settings
(100% / Actual size).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Final

import cv2
import numpy as np

from printgpt_alignment import DEFAULT_LAYOUT, A4CalibrationLayout
from printgpt_mvp.email_ingestion.settings import application_data_dir


PROFILE_SCHEMA: Final[str] = "printgpt-loop-calibration/1"
SESSION_SCHEMA: Final[str] = "printgpt-loop-calibration-session/1"
CALIBRATION_DIR_ENV: Final[str] = "PRINTGPT_CALIBRATION_DIR"
ACCURACY_TARGET_MM: Final[float] = 0.5
MARKER_DICTIONARY: Final[int] = cv2.aruco.DICT_5X5_100
GRID_COLUMNS: Final[int] = 4
GRID_ROWS: Final[int] = 6
MARKER_MM: Final[float] = 14.0
TARGET_MARGIN_MM: Final[float] = 14.0
_MAX_SCALE_ERROR: Final[float] = 0.03
_MAX_ROTATION_DEG: Final[float] = 2.0
_MIN_FIT_MARKERS: Final[int] = 8
_MIN_HOLDOUT_MARKERS: Final[int] = 6


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)


@dataclass(frozen=True, slots=True)
class TargetMarker:
    marker_id: int
    x_mm: float
    y_mm: float
    holdout: bool

    def pixel_box(self, layout: A4CalibrationLayout) -> tuple[int, int, int, int]:
        """Pixel edges the marker is drawn on (x0, y0, x1, y1), exclusive end."""
        sx = layout.raster_width_px / layout.page_width_mm
        sy = layout.raster_height_px / layout.page_height_mm
        return (int(round(self.x_mm * sx)), int(round(self.y_mm * sy)),
                int(round((self.x_mm + MARKER_MM) * sx)), int(round((self.y_mm + MARKER_MM) * sy)))

    def corners_mm(self, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
        """Outer corners as drawn (TL, TR, BR, BL, the ArUco order), in page mm."""
        x0, y0, x1, y1 = self.pixel_box(layout)
        sx = layout.page_width_mm / layout.raster_width_px
        sy = layout.page_height_mm / layout.raster_height_px
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float) * [sx, sy]


def target_markers(layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> list[TargetMarker]:
    """Marker grid spread over the page; alternate markers are held out for checking."""
    usable_w = layout.page_width_mm - 2 * TARGET_MARGIN_MM - MARKER_MM
    usable_h = layout.page_height_mm - 2 * TARGET_MARGIN_MM - MARKER_MM
    markers = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            marker_id = row * GRID_COLUMNS + column
            x = TARGET_MARGIN_MM + usable_w * column / (GRID_COLUMNS - 1)
            y = TARGET_MARGIN_MM + usable_h * row / (GRID_ROWS - 1)
            markers.append(TargetMarker(marker_id, x, y, holdout=(row + column) % 2 == 1))
    return markers


def render_target_mask(layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """0/255 mask (255 = toner) of the target page in canonical A4 pixels."""
    mask = np.zeros(layout.image_shape, dtype=np.uint8)
    dictionary = _dictionary()
    modules = dictionary.markerSize + 2
    for marker in target_markers(layout):
        image = cv2.aruco.generateImageMarker(dictionary, marker.marker_id, modules, borderBits=1)
        x0, y0, x1, y1 = marker.pixel_box(layout)
        scaled = cv2.resize(image, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        mask[y0:y1, x0:x1] = np.where(scaled < 128, 255, 0).astype(np.uint8)
    return mask


def write_target(output_dir: Path, correction_mm: np.ndarray | None = None,
                 layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> tuple[Path, Path]:
    """Exact-size A4 PNG and PDF of the target, with ``correction_mm`` applied like any export."""
    from printgpt_mvp.pipeline import _write_toner_outputs

    return _write_toner_outputs(render_target_mask(layout), Path(output_dir), layout,
                                None if correction_mm is None else np.asarray(correction_mm, dtype=float))


def _stats(errors: np.ndarray) -> dict[str, float]:
    return {"mean_mm": float(errors.mean()), "p95_mm": float(np.percentile(errors, 95)),
            "max_mm": float(errors.max()), "count": int(errors.size)}


def measure_round(scan_rgb: np.ndarray, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> dict[str, Any]:
    """Fit the residual map ``t -> o`` from a Scanner A4 scan of a printed target page.

    A scan that arrives upside down is turned upright first: the unique marker
    IDs make its orientation unambiguous, and drawing scans get their
    orientation confirmed by the user in the preview, so only the small residual
    belongs to the loop calibration.
    """
    measurement = _measure_upright(scan_rgb, layout)
    if abs(abs(measurement["rotation_degrees"]) - 180.0) < 10.0:
        measurement = _measure_upright(np.ascontiguousarray(np.rot90(scan_rgb, 2)), layout)
        measurement["scan_turned_degrees"] = 180
    else:
        measurement["scan_turned_degrees"] = 0
    return measurement


def _measure_upright(scan_rgb: np.ndarray, layout: A4CalibrationLayout) -> dict[str, Any]:
    gray = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2GRAY) if scan_rgb.ndim == 3 else scan_rgb
    detector = cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    scale = np.array([layout.page_width_mm / layout.raster_width_px, layout.page_height_mm / layout.raster_height_px])
    by_id = {marker.marker_id: marker for marker in target_markers(layout)}
    # ArUco corners use pixel-centre coordinates; page coordinates use pixel edges.
    found = {int(i): (c.reshape(4, 2) + 0.5) * scale
             for c, i in zip(corners or (), [] if ids is None else ids.ravel()) if int(i) in by_id}
    fit_ids = [i for i in found if not by_id[i].holdout]
    holdout_ids = [i for i in found if by_id[i].holdout]
    if len(fit_ids) < _MIN_FIT_MARKERS or len(holdout_ids) < _MIN_HOLDOUT_MARKERS:
        raise ValueError(
            f"found {len(found)} of {len(by_id)} calibration markers; scan the whole printed target page "
            "in Scanner A4 mode, sharp and unfolded"
        )

    def pairs(marker_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        return (np.vstack([by_id[i].corners_mm(layout) for i in marker_ids]),
                np.vstack([found[i] for i in marker_ids]))

    target_fit, observed_fit = pairs(fit_ids)
    coefficients, *_ = np.linalg.lstsq(np.c_[target_fit, np.ones(len(target_fit))], observed_fit, rcond=None)
    residual_map = np.eye(3)
    residual_map[:2] = coefficients.T
    target_hold, observed_hold = pairs(holdout_ids)
    predicted = np.c_[target_hold, np.ones(len(target_hold))] @ residual_map[:2].T
    centre = np.array([layout.page_width_mm / 2, layout.page_height_mm / 2])
    offset = residual_map[:2, :2] @ centre + residual_map[:2, 2] - centre
    # No fitting involved: how far this printed sheet actually is from the target, over every marker.
    target_all, observed_all = pairs(sorted(found))
    printed_errors = np.linalg.norm(observed_all - target_all, axis=1)
    scales = np.linalg.norm(residual_map[:2, :2], axis=0)
    rotation = float(np.degrees(np.arctan2(residual_map[1, 0], residual_map[0, 0])))
    problems = []
    if np.any(np.abs(scales - 1.0) > _MAX_SCALE_ERROR):
        problems.append(f"the page was scaled by {scales.mean():.3f}: print at 100% / Actual size, "
                        "without Fit to page or Shrink to printable area")
    if abs(rotation) > _MAX_ROTATION_DEG:
        problems.append(f"the page is rotated {rotation:.1f} degrees: place the sheet straight, as for drawings")
    return {
        "markers_found": len(found),
        "residual_map_mm": residual_map.tolist(),
        "offset_mm_at_centre": offset.tolist(),
        "axis_scale": scales.tolist(),
        "rotation_degrees": rotation,
        "printed_error": _stats(printed_errors),
        "holdout_after_fit": _stats(np.linalg.norm(predicted - observed_hold, axis=1)),
        "problems": problems,
    }


def calibration_dir() -> Path:
    configured = os.environ.get(CALIBRATION_DIR_ENV, "").strip()
    return Path(configured) if configured else application_data_dir() / "calibration"


def profile_key(printer_queue: str, tray: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{printer_queue}--{tray}").strip("-").lower()


class CalibrationSession:
    """Rounds of print -> scan -> measure for one printer and tray, stored as JSON."""

    def __init__(self, directory: Path, data: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.data = data

    @classmethod
    def start(cls, root: Path, printer_queue: str, tray: str, now: datetime | None = None) -> CalibrationSession:
        now = now or datetime.now(UTC)
        directory = Path(root) / f"{profile_key(printer_queue, tray)}-{now.strftime('%Y%m%dT%H%M%SZ')}"
        session = cls(directory, {"schema": SESSION_SCHEMA, "printer_queue": printer_queue, "tray": tray,
                                  "started_at": now.isoformat(timespec="seconds"),
                                  "correction_mm": np.eye(3).tolist(), "rounds": []})
        session.save()
        return session

    @classmethod
    def load(cls, directory: Path) -> CalibrationSession:
        data = json.loads((Path(directory) / "session.json").read_text(encoding="utf-8"))
        if data.get("schema") != SESSION_SCHEMA:
            raise ValueError(f"{directory} is not a calibration session")
        return cls(directory, data)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "session.json").write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    @property
    def rounds(self) -> list[dict[str, Any]]:
        return self.data["rounds"]

    @property
    def correction(self) -> np.ndarray:
        return np.asarray(self.data["correction_mm"], dtype=float)

    @property
    def next_round(self) -> int:
        return len(self.rounds) + 1

    def target_for_next_round(self, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> Path:
        """PDF to print for the next round, with the current correction applied."""
        _, pdf = write_target(self.directory / f"round-{self.next_round:02d}", self.correction, layout)
        return pdf

    def add_scan(self, scan_rgb: np.ndarray, scan_name: str, layout: A4CalibrationLayout = DEFAULT_LAYOUT,
                 now: datetime | None = None) -> dict[str, Any]:
        """Measure a scan of the latest printed round and update the correction.

        A scan that was already measured is refused: measuring it again would
        apply its correction twice and overshoot.
        """
        fingerprint = sha256(np.ascontiguousarray(scan_rgb).tobytes()).hexdigest()
        for earlier in self.rounds:
            if earlier.get("scan_sha256") == fingerprint:
                raise ValueError(
                    f"this scan was already measured as round {earlier['round']}; print the round "
                    f"{self.next_round} page, scan it, and measure that new scan"
                )
        measurement = measure_round(scan_rgb, layout)
        printed_with = self.correction
        result = {"round": self.next_round, "scan": scan_name, "scan_sha256": fingerprint,
                  "measured_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
                  "printed_with_correction": printed_with.tolist(), "corrected": len(self.rounds) > 0,
                  **measurement}
        if not measurement["problems"]:
            residual = np.asarray(measurement["residual_map_mm"], dtype=float)
            self.data["correction_mm"] = (printed_with @ np.linalg.inv(residual)).tolist()
        self.rounds.append(result)
        self.save()
        return result

    def verified_round(self) -> dict[str, Any] | None:
        """The latest round printed *with* a correction whose real error is within the target."""
        for result in reversed(self.rounds):
            if result["corrected"] and not result["problems"] \
                    and result["printed_error"]["max_mm"] < ACCURACY_TARGET_MM:
                return result
        return None

    def profile(self, now: datetime | None = None) -> dict[str, Any]:
        verified = self.verified_round()
        if verified is None:
            raise ValueError("no corrected round has been printed and measured within 0.5 mm yet")
        usable = [r for r in self.rounds if not r["problems"]]
        # Loop map seen by each round, independent of the correction it was printed with.
        loops = [np.asarray(r["residual_map_mm"]) @ np.linalg.inv(np.asarray(r["printed_with_correction"]))
                 for r in usable]
        centre = np.array([DEFAULT_LAYOUT.page_width_mm / 2, DEFAULT_LAYOUT.page_height_mm / 2, 1.0])
        offsets = np.array([(loop @ centre)[:2] - centre[:2] for loop in loops])
        return {
            "schema": PROFILE_SCHEMA,
            "status": "measured",
            "printer_queue": self.data["printer_queue"],
            "tray": self.data["tray"],
            "measured_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
            "session": str(self.directory),
            "rounds": len(self.rounds),
            # The correction that produced the verified print, not a later, untested refinement.
            "correction_mm": verified["printed_with_correction"],
            "verified_round": verified["round"],
            "verified_error": verified["printed_error"],
            "accuracy_bound_mm": verified["printed_error"]["max_mm"],
            "sheet_to_sheet_offset_spread_mm": float(np.linalg.norm(offsets.std(axis=0))) if len(offsets) > 1 else None,
            "valid_for": "same scanner workflow (Scanner A4), printer, tray, and 100% / Actual size printing",
        }


def save_profile(profile: dict[str, Any], root: Path | None = None) -> Path:
    root = root or calibration_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{profile_key(profile['printer_queue'], profile['tray'])}.json"
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return path


def load_profile(printer_queue: str, tray: str, root: Path | None = None) -> dict[str, Any] | None:
    """The saved, verified profile for this printer and tray, or ``None``."""
    path = (root or calibration_dir()) / f"{profile_key(printer_queue, tray)}.json"
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "measured":
        return None
    if not profile.get("accuracy_bound_mm", ACCURACY_TARGET_MM) < ACCURACY_TARGET_MM:
        return None
    return profile
