"""Deterministic synthetic A4 capture generation with exact homographies."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import cv2
import numpy as np

from .calibration_page import render_calibration_image
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT


def _read_scan(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image if image is not None and image.size else None


def _page_quad(
    rng: np.random.Generator,
    canvas_size: tuple[int, int],
    layout: A4CalibrationLayout,
    severity: float,
    crop_mode: str | None,
) -> np.ndarray:
    canvas_width, canvas_height = canvas_size
    height = rng.uniform(0.66, 0.84) * canvas_height
    width = height * layout.page_width_mm / layout.page_height_mm
    center = np.asarray(
        [canvas_width / 2 + rng.uniform(-0.04, 0.04) * canvas_width, canvas_height / 2 + rng.uniform(-0.03, 0.03) * canvas_height],
        dtype=np.float32,
    )
    angle = np.deg2rad(rng.uniform(-12.0, 12.0) * severity)
    local = np.asarray(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float32,
    )
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    quad = local @ rotation.T + center
    jitter = np.asarray(
        [
            [rng.uniform(-1, 1), rng.uniform(-1, 1)],
            [rng.uniform(-1, 1), rng.uniform(-1, 1)],
            [rng.uniform(-1, 1), rng.uniform(-1, 1)],
            [rng.uniform(-1, 1), rng.uniform(-1, 1)],
        ],
        dtype=np.float32,
    )
    quad += jitter * np.asarray([width * 0.055, height * 0.045], dtype=np.float32) * severity
    if crop_mode == "left":
        quad[:, 0] -= float(np.min(quad[:, 0]) + width * 0.13)
    elif crop_mode == "right":
        quad[:, 0] += float(canvas_width - np.max(quad[:, 0]) + width * 0.13)
    elif crop_mode == "top":
        quad[:, 1] -= float(np.min(quad[:, 1]) + height * 0.13)
    elif crop_mode == "bottom":
        quad[:, 1] += float(canvas_height - np.max(quad[:, 1]) + height * 0.13)
    return quad.astype(np.float32)


def _background(rng: np.random.Generator, canvas_size: tuple[int, int]) -> np.ndarray:
    width, height = canvas_size
    base = rng.integers(165, 225)
    x_gradient = np.linspace(rng.uniform(-18, 8), rng.uniform(-8, 18), width, dtype=np.float32)
    y_gradient = np.linspace(rng.uniform(-12, 8), rng.uniform(-8, 12), height, dtype=np.float32)[:, None]
    image = np.full((height, width), float(base), dtype=np.float32)
    image += x_gradient[None, :] + y_gradient
    return np.clip(image, 0, 255).astype(np.uint8)


def _apply_photometric_effects(
    image: np.ndarray,
    rng: np.random.Generator,
    blur_sigma: float,
    noise_sigma: float,
    shadow_strength: float,
) -> np.ndarray:
    result = image.astype(np.float32)
    if shadow_strength > 0:
        height, width = result.shape
        center = rng.uniform(0.2, 0.8, size=2) * np.asarray([width, height])
        yy, xx = np.mgrid[:height, :width]
        distance = np.sqrt(((xx - center[0]) / width) ** 2 + ((yy - center[1]) / height) ** 2)
        shadow = 1.0 - shadow_strength * np.clip(1.0 - distance / 0.8, 0.0, 1.0)
        result *= shadow
    if noise_sigma > 0:
        result += rng.normal(0.0, noise_sigma, result.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        kernel = max(3, int(round(blur_sigma * 6)) | 1)
        result = cv2.GaussianBlur(result, (kernel, kernel), blur_sigma)
    return result


def generate_synthetic_benchmark(
    output_dir: str | Path,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    count: int = 30,
    seed: int = 20261014,
    scan_paths: list[str | Path] | None = None,
    canvas_size: tuple[int, int] = (1920, 2600),
) -> Path:
    """Generate cases and return the written manifest path."""
    if count <= 0:
        raise ValueError("count must be positive")
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    scans = [Path(path) for path in (scan_paths or []) if Path(path).is_file()]
    invalid_count = 8 if count >= 30 else count // 4
    valid_count = count - invalid_count
    master_rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for index in range(count):
        case_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(case_seed)
        is_valid = index < valid_count
        invalid_index = index - valid_count
        crop_mode = None
        invalid_kind = None
        if not is_valid:
            if invalid_index < 4:
                crop_mode = ("left", "right", "top", "bottom")[invalid_index]
                invalid_kind = f"crop_{crop_mode}"
            elif invalid_index < 6:
                invalid_kind = "severe_degradation"
            else:
                invalid_kind = "marker_occlusion"

        if index < 5:
            category, severity = "affine", 0.25
        elif index < 13:
            category, severity = "perspective", 0.65
        elif index < 18:
            category, severity = "photometric", 0.45
        else:
            category, severity = "compound", 0.90

        scan_path = scans[index % len(scans)] if scans and index < max(15, len(scans)) else None
        scan = _read_scan(scan_path) if scan_path is not None else None
        canonical = render_calibration_image(layout, background=scan)
        quad = _page_quad(rng, canvas_size, layout, severity, crop_mode)
        source_corners = np.asarray(layout.page_corners_px, dtype=np.float32)
        canonical_to_source = cv2.getPerspectiveTransform(source_corners, quad)
        canvas = _background(rng, canvas_size)
        warped_page = cv2.warpPerspective(canonical, canonical_to_source, canvas_size, borderValue=255)
        page_mask = cv2.warpPerspective(
            np.full(layout.image_shape, 255, dtype=np.uint8),
            canonical_to_source,
            canvas_size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        capture = np.where(page_mask > 0, warped_page, canvas).astype(np.uint8)

        if invalid_kind == "marker_occlusion":
            marker_id = 10 if invalid_index == 6 else 12
            marker = layout.marker_by_id(marker_id)
            corners = np.asarray(marker.corners_px(layout.dpi), dtype=np.float32)
            observed = cv2.perspectiveTransform(corners[None, :, :], canonical_to_source)[0]
            cv2.fillConvexPoly(capture, np.rint(observed).astype(np.int32), int(np.median(canvas)))

        blur_sigma = float(rng.uniform(0.0, 1.4 if is_valid else 2.0))
        noise_sigma = float(rng.uniform(0.0, 8.0 if is_valid else 12.0))
        shadow_strength = float(rng.uniform(0.0, 0.22 if is_valid else 0.32))
        if category == "affine":
            blur_sigma *= 0.35
            noise_sigma *= 0.35
            shadow_strength *= 0.35
        if invalid_kind == "severe_degradation":
            blur_sigma = float(rng.uniform(4.0, 5.5))
            noise_sigma = float(rng.uniform(18.0, 28.0))
            shadow_strength = float(rng.uniform(0.25, 0.42))
        capture = _apply_photometric_effects(capture, rng, blur_sigma, noise_sigma, shadow_strength)

        filename = f"case_{index + 1:02d}.jpg"
        image_path = images_dir / filename
        quality = int(rng.integers(82, 96)) if is_valid else int(rng.integers(65, 90))
        if not cv2.imwrite(str(image_path), capture, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError(f"failed to write {image_path}")
        digest = sha256(image_path.read_bytes()).hexdigest()
        records.append(
            {
                "case_id": f"case_{index + 1:02d}",
                "image": f"images/{filename}",
                "sha256": digest,
                "seed": case_seed,
                "category": category,
                "expected_accept": is_valid,
                "invalid_kind": invalid_kind,
                "scan_background": scan_path.name if scan_path else None,
                "blur_sigma": blur_sigma,
                "noise_sigma": noise_sigma,
                "shadow_strength": shadow_strength,
                "jpeg_quality": quality,
                "canonical_to_source": canonical_to_source.tolist(),
                "source_quad": quad.tolist(),
            }
        )

    manifest = {
        "schema_version": 1,
        "layout_id": layout.layout_id,
        "dpi": layout.dpi,
        "canonical_raster_size": list(layout.raster_size_px),
        "canvas_size": list(canvas_size),
        "seed": seed,
        "case_count": count,
        "expected_valid_count": valid_count,
        "expected_invalid_count": invalid_count,
        "cases": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
