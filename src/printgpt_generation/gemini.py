"""Optional Gemini adapters built on the ``google-genai`` Interactions API.

Verified against ``google-genai`` 2.31 (``client.interactions.create``); both
default model IDs are in that SDK's model list. Model IDs are configuration,
not code. Importing this module never creates a client or reads credentials;
call ``create_gemini_client`` explicitly (it reads ``GEMINI_API_KEY`` via the
SDK unless a key is passed).

Framing. Gemini image output supports fixed aspect ratios, none of which is A4
(1:1.414). With the default ``crop`` framing the generator sends only a window
around the printable region, grown with enough of the existing drawing for
context and then to the closest supported aspect ratio on the real page. A
small thumbnail of the whole page, with the window outlined, gives global
context but is never returned. The output is declared as that window through
``GeneratedImage.canvas_box``, so registration is checked inside the window.
``full_page`` framing sends the whole page padded on the right to 3:4.
Whatever the page cannot supply is padded (white on the page, forbidden tint in
the guidance) and cropped off the output again.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Final

import cv2
import httpx
import numpy as np
from google import genai
from google.genai import types

# The Interactions API raises these instead of ``google.genai.errors``; 2.31 only
# exposes them from this module, so the SDK version is pinned in requirements.
from google.genai._gaos.lib.compat_errors import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
)

from .critic import CriticReport
from .framing import AspectFrame, PixelBox, best_aspect_frame, expand, fit_aspect, full_box, mask_bounds
from .guidance import FORBIDDEN_TINT_RGB
from .models import GeneratedImage, RGBImage, UsageRecord
from .providers import (
    CritiqueCall,
    GenerationCall,
    ProviderPermanentError,
    ProviderTimeoutError,
    ProviderTransientError,
    RawCritique,
)


DEFAULT_IMAGE_MODEL: Final[str] = "gemini-nano-banana-2.1"
DEFAULT_CRITIC_MODEL: Final[str] = "gemini-3.8-flash"
FULL_PAGE_ASPECT: Final[str] = "3:4"
# Aspect ratios accepted by ``ImageResponseFormat`` in google-genai 2.31; the
# extreme 1:4, 4:1, 1:8 and 8:1 ratios are left out until verified live.
CROP_ASPECTS: Final[tuple[str, ...]] = ("1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "21:9")
# ``ImageResponseFormatMimeType`` in google-genai 2.31 allows only JPEG output.
OUTPUT_MIME_TYPE: Final[str] = "image/jpeg"
# Canonical rasters always span the A4 page width.
_A4_WIDTH_MM: Final[float] = 210.0
_CONTEXT_OUTLINE_RGB: Final[tuple[int, int, int]] = (0, 90, 255)
_CONTEXT_NOTE: Final[str] = (
    "Context only, do not return or copy it: a reduced view of the whole page. The blue "
    "rectangle marks the area shown in the images above. Use it to understand what the "
    "drawing is, then return only that area."
)
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
# A status code the API never returns; with one attempt it disables SDK-internal
# retries so ``RetryPolicy`` remains the only, bounded retry mechanism.
_NO_SDK_RETRY: Final[types.HttpRetryOptions] = types.HttpRetryOptions(attempts=1, http_status_codes=[599])


@dataclass(frozen=True, slots=True)
class GeminiSettings:
    image_model: str = DEFAULT_IMAGE_MODEL
    critic_model: str = DEFAULT_CRITIC_MODEL
    image_size: str = "2K"
    timeout_s: float = 120.0
    output_aspect_tolerance: float = 0.02
    framing: str = "crop"
    context_margin_mm: float = 25.0
    # The crop grows until it holds this much existing ink, so registration can be verified.
    min_context_ink_mm2: float = 20.0
    crop_aspects: tuple[str, ...] = CROP_ASPECTS
    thumbnail_max_side_px: int = 1024

    def __post_init__(self) -> None:
        if not self.image_model or not self.critic_model:
            raise ValueError("model IDs must be non-empty")
        if self.image_size not in ("512", "1K", "2K", "4K"):
            raise ValueError("image_size must be one of 512, 1K, 2K, 4K")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.framing not in ("crop", "full_page"):
            raise ValueError("framing must be 'crop' or 'full_page'")
        if self.context_margin_mm < 0 or self.min_context_ink_mm2 < 0:
            raise ValueError("context settings cannot be negative")
        if not self.crop_aspects or any(aspect not in CROP_ASPECTS for aspect in self.crop_aspects):
            raise ValueError(f"crop_aspects must be a non-empty subset of {CROP_ASPECTS}")
        if self.thumbnail_max_side_px < 64:
            raise ValueError("thumbnail_max_side_px must be at least 64")


def create_gemini_client(api_key: str | None = None, *, httpx_client: httpx.Client | None = None) -> genai.Client:
    """Create a client with SDK retries disabled.

    Without ``api_key`` the SDK reads ``GEMINI_API_KEY``. ``httpx_client`` lets
    tests substitute a mock transport.
    """
    options = types.HttpOptions(retry_options=_NO_SDK_RETRY, httpx_client=httpx_client)
    if api_key:
        return genai.Client(api_key=api_key, http_options=options)
    return genai.Client(http_options=options)


@dataclass(frozen=True, slots=True)
class CanvasFrame:
    """One page window, padded right/bottom to the requested aspect ratio."""

    frame: AspectFrame
    page_shape: tuple[int, int]

    @property
    def box(self) -> PixelBox:
        return self.frame.box

    @property
    def aspect(self) -> str:
        return self.frame.aspect

    @property
    def cropped(self) -> bool:
        return not self.box.is_full(self.page_shape)

    def prepare(self, image: RGBImage, fill: tuple[int, int, int] = (255, 255, 255)) -> RGBImage:
        pad_right, pad_bottom = self.frame.padding
        window = np.ascontiguousarray(self.box.crop(image))
        if not pad_right and not pad_bottom:
            return window
        return cv2.copyMakeBorder(window, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=fill)

    def restore(self, output: RGBImage, tolerance: float) -> RGBImage:
        height, width = output.shape[:2]
        expected = self.frame.canvas_width / self.frame.canvas_height
        if abs((width / height) / expected - 1.0) > tolerance:
            raise ProviderPermanentError(f"Gemini returned {width}x{height}, not the requested {self.aspect}")
        size = (self.frame.canvas_width, self.frame.canvas_height)
        resized = cv2.resize(output, size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized[: self.box.height, : self.box.width])


def frame_for_call(call: GenerationCall, settings: GeminiSettings) -> CanvasFrame:
    """Choose the window sent to Gemini; deterministic for a given page and region."""
    shape = (int(call.source_page.shape[0]), int(call.source_page.shape[1]))
    region = mask_bounds(call.printable_region_mask)
    if settings.framing == "full_page" or region is None:
        return CanvasFrame(fit_aspect(full_box(shape), FULL_PAGE_ASPECT, shape), shape)
    px_per_mm = shape[1] / _A4_WIDTH_MM
    ink = cv2.cvtColor(call.source_page, cv2.COLOR_RGB2GRAY) < 128
    min_ink = settings.min_context_ink_mm2 * px_per_mm**2
    margin = settings.context_margin_mm * px_per_mm
    frame = best_aspect_frame(expand(region, margin, shape), settings.crop_aspects, shape)
    for _ in range(6):
        if frame.box.is_full(shape) or np.count_nonzero(frame.box.crop(ink)) >= min_ink:
            break
        margin = margin * 1.6 + px_per_mm
        frame = best_aspect_frame(expand(region, margin, shape), settings.crop_aspects, shape)
    return CanvasFrame(frame, shape)


def context_thumbnail(page: RGBImage, box: PixelBox, max_side: int) -> RGBImage:
    """Reduced whole page with ``box`` outlined; context for the model only."""
    scale = min(1.0, max_side / max(page.shape[:2]))
    size = (max(1, int(round(page.shape[1] * scale))), max(1, int(round(page.shape[0] * scale))))
    thumbnail = cv2.resize(page, size, interpolation=cv2.INTER_AREA)
    top_left = (int(box.x * scale), int(box.y * scale))
    bottom_right = (int((box.right - 1) * scale), int((box.bottom - 1) * scale))
    cv2.rectangle(thumbnail, top_left, bottom_right, _CONTEXT_OUTLINE_RGB, max(2, round(max(size) / 400)))
    return thumbnail


def _png_part(image: RGBImage) -> dict[str, str]:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    return {"type": "image", "data": base64.b64encode(encoded.tobytes()).decode("ascii"), "mime_type": "image/png"}


def _text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _decode_image(data: str) -> RGBImage:
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ProviderPermanentError("Gemini returned undecodable image data")
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def _usage(interaction: object) -> UsageRecord:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return UsageRecord()
    # Cost stays unknown: pricing is not verified here and must not be guessed.
    return UsageRecord(input_tokens=usage.total_input_tokens, output_tokens=usage.total_output_tokens)


def _create_interaction(client: genai.Client, timeout_s: float, **body: object) -> object:
    """Call the API and translate SDK/transport errors into provider errors."""
    try:
        interaction = client.interactions.create(timeout=timeout_s, **body)
    except APITimeoutError as error:
        raise ProviderTimeoutError(f"Gemini request exceeded {timeout_s:.0f} s") from error
    except APIConnectionError as error:
        raise ProviderTransientError(f"Gemini connection error: {error}") from error
    except APIStatusError as error:
        message = f"Gemini request failed ({error.status_code}): {error.message}"
        if error.status_code in _RETRYABLE_STATUSES:
            raise ProviderTransientError(message) from error
        raise ProviderPermanentError(message) from error
    except APIError as error:
        raise ProviderPermanentError(f"Gemini API error: {error.message}") from error
    status = getattr(interaction, "status", None)
    if status != "completed":
        raise ProviderPermanentError(f"Gemini interaction ended with status {status!r}")
    return interaction


class GeminiImageGenerator:
    """``CandidateGenerator`` backed by a Gemini image model."""

    def __init__(self, client: genai.Client, settings: GeminiSettings = GeminiSettings()) -> None:
        self._client = client
        self._settings = settings

    @property
    def model_id(self) -> str:
        return self._settings.image_model

    def generate(self, call: GenerationCall) -> GeneratedImage:
        frame = frame_for_call(call, self._settings)
        forbidden_padding = tuple(int(value) for value in FORBIDDEN_TINT_RGB)
        parts = [
            _text_part(call.prompt),
            _png_part(frame.prepare(call.source_page)),
            _png_part(frame.prepare(call.guidance_image, forbidden_padding)),
        ]
        if call.previous_candidate is not None:
            parts.append(_text_part("Previous attempt to revise:"))
            parts.append(_png_part(frame.prepare(call.previous_candidate)))
        if frame.cropped:
            parts.append(_text_part(_CONTEXT_NOTE))
            thumbnail = context_thumbnail(call.source_page, frame.box, self._settings.thumbnail_max_side_px)
            parts.append(_png_part(thumbnail))
        interaction = _create_interaction(
            self._client,
            self._settings.timeout_s,
            model=self._settings.image_model,
            input=parts,
            generation_config={"seed": int(call.seed)},
            response_format={
                "type": "image",
                "aspect_ratio": frame.aspect,
                "image_size": self._settings.image_size,
                "mime_type": OUTPUT_MIME_TYPE,
            },
        )
        image = getattr(interaction, "output_image", None)
        if image is None or not image.data:
            raise ProviderPermanentError("Gemini returned no image")
        output = frame.restore(_decode_image(image.data), self._settings.output_aspect_tolerance)
        return GeneratedImage(
            image=output,
            model_id=self.model_id,
            usage=_usage(interaction),
            canvas_box=frame.box.as_tuple() if frame.cropped else None,
        )


class GeminiVisualCritic:
    """``VisualCritic`` that requests JSON constrained to ``CriticReport``'s schema."""

    def __init__(self, client: genai.Client, settings: GeminiSettings = GeminiSettings()) -> None:
        self._client = client
        self._settings = settings

    @property
    def model_id(self) -> str:
        return self._settings.critic_model

    def critique(self, call: CritiqueCall) -> RawCritique:
        images = [call.original_page, call.guidance_image, call.sanitized_preview]
        if call.region_preview is not None:
            images.append(call.region_preview)
        interaction = _create_interaction(
            self._client,
            self._settings.timeout_s,
            model=self._settings.critic_model,
            input=[_text_part(call.prompt), *(_png_part(image) for image in images)],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema_": CriticReport.model_json_schema(),
            },
        )
        # Schema-constrained output is still parsed and validated as untrusted text.
        return RawCritique(getattr(interaction, "output_text", None) or "", self.model_id, _usage(interaction))
