"""Free manual Gemini tests: prepare app inputs from a scan, then score the app's output offline.

    python scripts/manual_gemini_test.py prepare <scan.pdf> --region X Y W H --out <dir> [--prompt TEXT]
    python scripts/manual_gemini_test.py score <dir> <gemini_output.png> [--name A]

``prepare`` writes exactly what a live run would send for each candidate: the
page crop, its region guidance, a context thumbnail, and the two variant
prompts. Paste those into the Gemini app. ``score`` runs a downloaded app output
through the real pipeline (crop paste-back, alignment, registration, Track A,
sanitized preview) with a mock transport and a placeholder critic: no network,
no cost. Nothing printable is created.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

# Use this checkout's sources even when the shared venv has another checkout installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402

from printgpt_alignment import DEFAULT_LAYOUT  # noqa: E402
from printgpt_generation.framing import expand, mask_bounds  # noqa: E402
from printgpt_generation.gemini import (  # noqa: E402
    _CONTEXT_NOTE,
    GeminiSettings,
    context_thumbnail,
    create_gemini_client,
    frame_for_call,
)
from printgpt_generation.guidance import FORBIDDEN_TINT_RGB, render_region_guidance  # noqa: E402
from printgpt_generation.models import AttemptKind  # noqa: E402
from printgpt_generation.prompts import build_generation_prompt, resolve_user_prompt  # noqa: E402
from printgpt_generation.providers import GenerationCall  # noqa: E402
from printgpt_mvp.input_normalization import InputMode, normalize_document_input  # noqa: E402
from printgpt_mvp.pipeline import rectangular_allowed_region, run_offline_document  # noqa: E402

_PLACEHOLDER_CRITIC = {
    "score": 5, "boundary_continuity_score": 5, "style_match_score": 5, "geometry_plausibility_score": 5,
    "problems": ["placeholder critic: manual scoring checks geometry only"],
    "revision_instruction": "", "recommendation": "accept",
}


def _save(path: Path, image: np.ndarray) -> None:
    pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"failed to write {path}")


def _offline_run(scan: Path, region: list[float], out: Path, prompt: str, client=None):
    x, y, width, height = region
    normalization = normalize_document_input(scan, input_mode=InputMode.SCANNER_A4, layout=DEFAULT_LAYOUT)
    return run_offline_document(
        scan,
        rectangular_allowed_region(x_mm=x, y_mm=y, width_mm=width, height_mm=height, layout=DEFAULT_LAYOUT),
        out,
        user_prompt=prompt,
        layout=DEFAULT_LAYOUT,
        input_mode=InputMode.SCANNER_A4,
        normalization_result=normalization,
        normalization_confirmed=True,
        backend="gemini" if client is not None else "offline",
        gemini_client=client,
        gemini_settings=GeminiSettings(image_model="gemini-app-manual", critic_model="placeholder")
        if client is not None else None,
    )


def prepare(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    # The offline run builds the real Track A masks; only its page and masks are used here.
    run = _offline_run(args.scan, args.region, out / "_masks_run", args.prompt)
    page = run.canonical_page_rgb
    printable = run.masks.printable_region_mask
    guidance = render_region_guidance(page, printable)
    call = GenerationCall("manual", 0, 0, AttemptKind.INITIAL, "", 0, page, guidance, printable)
    frame = frame_for_call(call, GeminiSettings())
    forbidden = tuple(int(value) for value in FORBIDDEN_TINT_RGB)
    _save(out / "1_page_crop.png", frame.prepare(page))
    _save(out / "2_guidance_crop.png", frame.prepare(guidance, forbidden))
    _save(out / "3_context_thumbnail.png", context_thumbnail(page, frame.box, 1024))
    resolved = resolve_user_prompt(args.prompt)
    for index, name in enumerate("AB"):
        text = f"{build_generation_prompt(resolved, index)}\n\n{_CONTEXT_NOTE}\n"
        (out / f"prompt_{name}.txt").write_text(text, encoding="utf-8")
    manifest = {
        "scan": str(args.scan.resolve()), "region_mm": args.region, "prompt": args.prompt,
        "aspect": frame.aspect, "crop_box_px": list(frame.box.as_tuple()),
    }
    (out / "manual_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mm = 25.4 / DEFAULT_LAYOUT.dpi
    x, y, width, height = (round(value * mm, 1) for value in frame.box.as_tuple())
    print(f"crop {frame.aspect}: x {x}-{x + width} mm, y {y}-{y + height} mm")
    print(f"upload in order: 1_page_crop.png, 2_guidance_crop.png, 3_context_thumbnail.png; then paste prompt_A.txt")
    print(f"files: {out.resolve()}")
    return 0


def score(args: argparse.Namespace) -> int:
    manifest = json.loads((args.dir / "manual_test.json").read_text(encoding="utf-8"))
    pixels = cv2.imread(str(args.output), cv2.IMREAD_COLOR)
    if pixels is None:
        print(f"cannot read {args.output}")
        return 2
    ok, encoded = cv2.imencode(".png", pixels)
    data = base64.b64encode(encoded.tobytes()).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["response_format"]["type"] == "image":
            content = [{"type": "image", "mime_type": "image/png", "data": data}]
        else:
            content = [{"type": "text", "text": json.dumps(_PLACEHOLDER_CRITIC)}]
        return httpx.Response(200, json={"id": "x", "status": "completed",
                                         "steps": [{"type": "model_output", "content": content}]})

    client = create_gemini_client("offline", httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = args.dir / f"score_{args.name}"
    run = _offline_run(Path(manifest["scan"]), manifest["region_mm"], out / "_run", manifest["prompt"], client)
    attempt = run.generation.attempts[0]
    record = attempt.normalization
    result: dict[str, object] = {"output": str(args.output), "status": attempt.status.value}
    if attempt.generation.failure is not None:
        result["failure"] = attempt.generation.failure.message
    if record is not None:
        result["failure_reasons"] = list(record.failure_reasons)
        if record.alignment is not None:
            alignment = record.alignment
            result["alignment"] = {"applied": alignment.applied, "reason": alignment.reason,
                                   "shift_mm": [round(v, 2) for v in alignment.centre_shift_mm],
                                   "scale": round(alignment.scale, 4), "rotation_deg": round(alignment.rotation_deg, 2)}
        if record.registration is not None:
            result["residual_shift_mm"] = round(record.registration.estimated_shift_mm, 3)
            result["original_ink_kept"] = round(record.registration.source_retention, 4)
        if record.extraction is not None:
            result["new_ink_pixels"] = record.extraction.new_ink_pixels
            result["dropped_coloured_pixels"] = record.extraction.discarded_coloured_pixels
    if attempt.validation is not None:
        result["track_a_accepted"] = attempt.validation.accepted
        result["track_a_reasons"] = list(attempt.validation.reasons)
        result["track_a_removed_fraction"] = attempt.validation.metrics.get("removed_fraction")
    region = mask_bounds(run.masks.allowed_region_mask)
    for name, image in (("sanitized_preview", attempt.preview_rgb),
                        ("aligned_candidate", attempt.normalized.candidate_rgb if attempt.normalized else None)):
        if image is not None and region is not None:
            close_up = expand(region, 60 * DEFAULT_LAYOUT.dpi / 25.4, image.shape).crop(image)
            _save(out / f"{name}.png", close_up)
    (out / "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"images: {out.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="write Gemini app inputs for one scan")
    prep.add_argument("scan", type=Path)
    prep.add_argument("--region", nargs=4, type=float, metavar=("X", "Y", "W", "H"), required=True)
    prep.add_argument("--prompt", default="")
    prep.add_argument("--out", type=Path, required=True)
    sc = commands.add_parser("score", help="check one Gemini app output offline")
    sc.add_argument("dir", type=Path, help="folder written by prepare")
    sc.add_argument("output", type=Path, help="image downloaded from the Gemini app")
    sc.add_argument("--name", default="A")
    args = parser.parse_args(argv)
    return prepare(args) if args.command == "prepare" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
