"""Prepare a small SmartDoc-derived camera benchmark directly from its archive."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
import json
import tarfile

import cv2
import numpy as np

from .calibration_page import render_calibration_image
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT


@dataclass(frozen=True, slots=True)
class AnnotatedFrame:
    video_stem: str
    frame_index: int
    quad: tuple[tuple[float, float], ...]


def _parse_annotations(archive: tarfile.TarFile) -> dict[str, list[AnnotatedFrame]]:
    groups: dict[str, list[AnnotatedFrame]] = {}
    members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".gt.xml")]
    for member in members:
        stream = archive.extractfile(member)
        if stream is None:
            continue
        root = ET.fromstring(stream.read())
        source_name = root.findtext("source_sample_file") or PurePosixPath(member.name).name.replace(".gt.xml", ".avi")
        video_stem = PurePosixPath(source_name).stem
        frames: list[AnnotatedFrame] = []
        for frame in root.findall(".//frame"):
            if frame.attrib.get("rejected", "false").lower() == "true":
                continue
            points = {point.attrib["name"]: (float(point.attrib["x"]), float(point.attrib["y"])) for point in frame.findall("point")}
            if not {"tl", "tr", "br", "bl"}.issubset(points):
                continue
            frames.append(
                AnnotatedFrame(
                    video_stem=video_stem,
                    frame_index=int(frame.attrib["index"]),
                    quad=(points["tl"], points["tr"], points["br"], points["bl"]),
                )
            )
        if frames:
            groups[video_stem] = frames
    return groups


def _features(frame: AnnotatedFrame, width: int = 1920, height: int = 1080) -> np.ndarray:
    quad = np.asarray(frame.quad, dtype=np.float64)
    area = abs(float(cv2.contourArea(quad.astype(np.float32)))) / (width * height)
    center = quad.mean(axis=0) / np.asarray([width, height])
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    angle = np.arctan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0]) / np.pi
    perspective_x = abs(top - bottom) / max(top, bottom, 1e-9)
    perspective_y = abs(left - right) / max(left, right, 1e-9)
    return np.asarray([area, center[0], center[1], angle, perspective_x, perspective_y], dtype=np.float64)


def _select_diverse(frames: list[AnnotatedFrame], count: int) -> list[AnnotatedFrame]:
    if count >= len(frames):
        return frames
    matrix = np.asarray([_features(frame) for frame in frames], dtype=np.float64)
    standard_deviation = matrix.std(axis=0)
    standard_deviation[standard_deviation < 1e-9] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / standard_deviation
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(normalized[:, None, :] - normalized[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return [frames[index] for index in sorted(selected, key=lambda index: frames[index].frame_index)]


def _allocation(names: list[str], total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(names))
    return {name: base + (1 if index < remainder else 0) for index, name in enumerate(names)}


def prepare_smartdoc_benchmark(
    archive_path: str | Path,
    output_dir: str | Path,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    count: int = 10,
) -> Path:
    """Create exactly-ground-truthed marker captures from real SmartDoc scenes."""
    archive_path = Path(archive_path)
    if count <= 0:
        raise ValueError("count must be positive")
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    with tarfile.open(archive_path, "r:gz") as archive:
        groups = _parse_annotations(archive)
        if not groups:
            raise ValueError("SmartDoc archive contains no usable ground-truth annotations")
        names = sorted(groups)
        allocation = _allocation(names, count)
        selected = {name: _select_diverse(groups[name], allocation[name]) for name in names}
        video_members = {
            PurePosixPath(member.name).stem: member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".avi")
        }
        missing_videos = sorted(set(selected) - set(video_members))
        if missing_videos:
            raise ValueError(f"SmartDoc videos missing from archive: {missing_videos}")

        with TemporaryDirectory(prefix="printgpt_smartdoc_") as temporary_directory:
            temporary = Path(temporary_directory)
            case_number = 0
            for video_stem in names:
                member = video_members[video_stem]
                video_path = temporary / f"{video_stem}.avi"
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f"could not read {member.name}")
                with source, video_path.open("wb") as destination:
                    destination.write(source.read())
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise OSError(f"could not open {video_stem}.avi")
                for annotation in selected[video_stem]:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, annotation.frame_index - 1))
                    ok, frame = capture.read()
                    if not ok:
                        raise OSError(f"could not read frame {annotation.frame_index} from {video_stem}")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    quad = np.asarray(annotation.quad, dtype=np.float32)
                    canonical_corners = np.asarray(layout.page_corners_px, dtype=np.float32)
                    source_to_canonical = cv2.getPerspectiveTransform(quad, canonical_corners)
                    canonical_to_source = np.linalg.inv(source_to_canonical)
                    rectified_document = cv2.warpPerspective(gray, source_to_canonical, layout.raster_size_px, borderValue=255)
                    marked_page = render_calibration_image(layout, background=rectified_document)
                    frame_size = (gray.shape[1], gray.shape[0])
                    projected = cv2.warpPerspective(marked_page, canonical_to_source, frame_size, borderValue=255)
                    mask = cv2.warpPerspective(
                        np.full(layout.image_shape, 255, dtype=np.uint8),
                        canonical_to_source,
                        frame_size,
                        flags=cv2.INTER_NEAREST,
                        borderValue=0,
                    )
                    result = np.where(mask > 0, projected, gray).astype(np.uint8)
                    case_number += 1
                    filename = f"smartdoc_{case_number:02d}_{video_stem}_f{annotation.frame_index:04d}.jpg"
                    image_path = images_dir / filename
                    if not cv2.imwrite(str(image_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        raise OSError(f"failed to write {image_path}")
                    records.append(
                        {
                            "case_id": f"smartdoc_{case_number:02d}",
                            "image": f"images/{filename}",
                            "sha256": sha256(image_path.read_bytes()).hexdigest(),
                            "category": "smartdoc_camera",
                            "expected_accept": True,
                            "invalid_kind": None,
                            "source_video": video_stem,
                            "source_frame_index": annotation.frame_index,
                            "canonical_to_source": canonical_to_source.tolist(),
                            "source_quad": quad.tolist(),
                        }
                    )
                capture.release()

    manifest = {
        "schema_version": 1,
        "layout_id": layout.layout_id,
        "dpi": layout.dpi,
        "canonical_raster_size": list(layout.raster_size_px),
        "source_dataset": "ICDAR 2015 SmartDoc Challenge 1 sample",
        "source_archive_filename": archive_path.name,
        "case_count": len(records),
        "expected_valid_count": len(records),
        "expected_invalid_count": 0,
        "cases": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
