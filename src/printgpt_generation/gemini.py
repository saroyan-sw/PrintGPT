"""Optional Gemini adapters built on the ``google-genai`` Interactions API.

Verified against ``google-genai`` 2.31 (``client.interactions.create``) and
the Gemini model list of October 2026. Model IDs are configuration, not code.
Importing this module never creates a client or reads credentials; call
``create_gemini_client`` explicitly (it reads ``GEMINI_API_KEY`` via the SDK).

Gemini image output supports fixed aspect ratios, none of which is A4
(1:1.414). The generator therefore pads every page image on the right with
white to 3:4, marks the padding as forbidden in the guidance, requests 3:4
output, rescales the result to the padded canvas, and crops the padding off.
The returned image has A4 aspect, so normal registration checks still decide
whether the original content stayed in place.
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
_REQUEST_ASPECT: Final[str] = "3:4"
_REQUEST_ASPECT_VALUE: Final[float] = 3.0 / 4.0
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

    def __post_init__(self) -> None:
        if not self.image_model or not self.critic_model:
            raise ValueError("model IDs must be non-empty")
        if self.image_size not in ("512", "1K", "2K", "4K"):
            raise ValueError("image_size must be one of 512, 1K, 2K, 4K")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


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
class PaddedCanvas:
    """Right-padding that turns an A4 raster into the requested aspect ratio."""

    source_width: int
    source_height: int
    padded_width: int

    @classmethod
    def for_shape(cls, height: int, width: int) -> PaddedCanvas:
        padded_width = max(width, int(round(height * _REQUEST_ASPECT_VALUE)))
        return cls(width, height, padded_width)

    def pad(self, image: RGBImage, fill: tuple[int, int, int] = (255, 255, 255)) -> RGBImage:
        padding = self.padded_width - self.source_width
        return cv2.copyMakeBorder(image, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=fill)

    def unpad(self, output: RGBImage, tolerance: float) -> RGBImage:
        height, width = output.shape[:2]
        expected = self.padded_width / self.source_height
        if abs((width / height) / expected - 1.0) > tolerance:
            raise ProviderPermanentError(f"Gemini returned {width}x{height}, not the requested {_REQUEST_ASPECT}")
        resized = cv2.resize(output, (self.padded_width, self.source_height), interpolation=cv2.INTER_AREA)
        return resized[:, : self.source_width]


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
        height, width = call.source_page.shape[:2]
        canvas = PaddedCanvas.for_shape(height, width)
        forbidden_padding = tuple(int(value) for value in FORBIDDEN_TINT_RGB)
        parts = [
            _text_part(call.prompt),
            _png_part(canvas.pad(call.source_page)),
            _png_part(canvas.pad(call.guidance_image, forbidden_padding)),
        ]
        if call.previous_candidate is not None:
            parts.append(_text_part("Previous attempt to revise:"))
            parts.append(_png_part(canvas.pad(call.previous_candidate)))
        interaction = _create_interaction(
            self._client,
            self._settings.timeout_s,
            model=self._settings.image_model,
            input=parts,
            response_format={
                "type": "image",
                "aspect_ratio": _REQUEST_ASPECT,
                "image_size": self._settings.image_size,
                "mime_type": "image/png",
            },
        )
        image = getattr(interaction, "output_image", None)
        if image is None or not image.data:
            raise ProviderPermanentError("Gemini returned no image")
        output = canvas.unpad(_decode_image(image.data), self._settings.output_aspect_tolerance)
        return GeneratedImage(image=output, model_id=self.model_id, usage=_usage(interaction))


class GeminiVisualCritic:
    """``VisualCritic`` that requests JSON constrained to ``CriticReport``'s schema."""

    def __init__(self, client: genai.Client, settings: GeminiSettings = GeminiSettings()) -> None:
        self._client = client
        self._settings = settings

    @property
    def model_id(self) -> str:
        return self._settings.critic_model

    def critique(self, call: CritiqueCall) -> RawCritique:
        interaction = _create_interaction(
            self._client,
            self._settings.timeout_s,
            model=self._settings.critic_model,
            input=[
                _text_part(call.prompt),
                _png_part(call.original_page),
                _png_part(call.guidance_image),
                _png_part(call.sanitized_preview),
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema_": CriticReport.model_json_schema(),
            },
        )
        # Schema-constrained output is still parsed and validated as untrusted text.
        return RawCritique(getattr(interaction, "output_text", None) or "", self.model_id, _usage(interaction))
