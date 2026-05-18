"""Unified multimodal content normalization.

This module takes any of:

* a plain ``str`` containing legacy placeholders such as
  ``[image,url=...] / [image,file=...] / [image,base64=...]``,
  ``[audio,url=...] / [audio,file=...] / [audio,base64=...]``,
  ``[video,url=... ,fps=2] / [video,file=...] / [video,base64=...]``,
  ``[gif,url=...] / [gif,file=...] / [gif,base64=...]``;
* a list of OpenAI-style content parts
  ``[{"type":"text", "text":"..."}, {"type":"image_url", "image_url":{"url":"..."}}, ...]``;
* a mix of the two (e.g. an OpenAI list whose ``text`` entries still embed
  legacy placeholders);

and produces a flat, ordered list of :class:`ContentPart` objects, then turns
that list into the standard OpenAI ``content`` array expected by the upstream
provider. Order is preserved relative to where each piece appears in the
original input -- this is essential for vision-language models where the
position of an image relative to the surrounding text changes interpretation.

GIFs are special: they are expanded locally into a sequence of image frames
(via Pillow) because vLLM does not currently treat GIFs as videos. All other
media types are passed through as references (``file:`` / ``http(s):`` /
``data:`` URIs), letting vLLM's media pipeline do the heavy lifting on the
server side.

When ``prefetch_files=True`` or the provider config has ``prefetch_media=true``,
**HTTP(S) URLs for all media types (image/audio/video) are downloaded locally
and inlined as ``data:`` URIs** before being sent to the upstream. This solves
the problem of vLLM (or other providers) being unable to fetch protected URLs
(e.g. CDN-signed URLs with temporary keys).
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from PIL import Image, ImageSequence  # type: ignore
except Exception:  # pragma: no cover -- pillow listed in requirements
    Image = None  # type: ignore
    ImageSequence = None  # type: ignore

LOG = logging.getLogger(__name__)

# Placeholder pattern: [type,arg=value,arg=value,...]
# Where type is one of image/audio/video/gif. The value of each arg can be a
# url containing ``,`` or ``]`` characters, so we use a tolerant pattern and
# then split args manually.
_PLACEHOLDER_RE = re.compile(
    r"\[(image|audio|video|gif),(?P<body>[^\[\]]*)\]",
    re.IGNORECASE,
)

# Default frame budget when expanding GIFs. Big GIFs would otherwise blow up
# the prompt; this is the same kind of cap vLLM applies for videos.
DEFAULT_GIF_MAX_FRAMES = 16

# User-Agent for URL prefetch requests (some CDNs require it).
_PREFETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# URL prefetch helper
# ---------------------------------------------------------------------------


def _prefetch_url(url: str, timeout: float = 15.0) -> Optional[Tuple[bytes, str]]:
    """Download a URL and return ``(raw_bytes, mime_type)``.

    Returns ``None`` on any failure (network error, timeout, empty body).
    The caller silently falls back to the original URL when prefetch fails.
    """
    try:
        import requests  # lazy import, already in requirements.txt
    except ImportError:
        LOG.warning("requests not available, cannot prefetch %s", url)
        return None
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _PREFETCH_USER_AGENT},
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()
        data = resp.content
        if not data:
            return None
        mime = resp.headers.get("Content-Type", "") or _guess_mime_from_url(url)
        return data, mime
    except Exception as e:
        LOG.debug("Failed to prefetch %s: %r", url, e)
        return None


def _guess_mime_from_url(url: str) -> str:
    """Guess MIME type from the URL's extension."""
    path = url.split("?")[0].split("#")[0]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContentPart:
    """One ordered piece of a normalized message body.

    ``kind`` is one of: ``text``, ``image``, ``audio``, ``video``.
    For media kinds, ``ref`` carries one of:

      * ``{"source": "url",   "url":  "http(s)://..."}``
      * ``{"source": "file",  "path": "/abs/or/rel/path"}``
      * ``{"source": "base64","data": "<b64>", "mime": "image/png"}``

    Extra knobs (such as ``fps`` for videos) are kept in ``options``.
    """

    kind: str
    text: Optional[str] = None
    ref: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Placeholder parsing
# ---------------------------------------------------------------------------


def _parse_placeholder_body(body: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse the comma-separated ``key=value`` body of a placeholder.

    Returns a tuple ``(ref, options)`` where ``ref`` contains exactly one of
    ``url`` / ``file`` / ``base64``, and ``options`` is everything else
    (numeric values are coerced when possible).
    """
    # We greedily split on the FIRST '=' of each segment so that the value may
    # itself contain '='. The first segment determines the source kind.
    # However the value may also contain ',' (e.g. base64 padding). To keep
    # things robust we adopt the convention: the source value is whatever
    # follows the first '=' up to the NEXT recognised ``,key=`` boundary, or
    # the end of the body.
    keys = ("url", "file", "base64", "fps", "max_frames", "format", "mime")
    # Find the start indices of every recognised key.
    boundaries: List[Tuple[int, str]] = []
    for key in keys:
        for m in re.finditer(rf"(^|,){key}=", body):
            start = m.start() + (0 if m.start() == 0 else 1)
            boundaries.append((start, key))
    boundaries.sort(key=lambda x: x[0])
    if not boundaries:
        return {}, {}

    pieces: Dict[str, str] = {}
    for i, (start, key) in enumerate(boundaries):
        end = boundaries[i + 1][0] - 1 if i + 1 < len(boundaries) else len(body)
        if end < 0:
            end = len(body)
        # ``body[start:end]`` looks like ``key=value`` (or ``key=value,`` if it
        # was followed by another boundary; the ``-1`` above already stripped
        # the trailing ``,``).
        seg = body[start:end]
        if "=" in seg:
            _, _, val = seg.partition("=")
            pieces[key] = val.strip()

    ref: Dict[str, Any] = {}
    if "url" in pieces:
        ref = {"source": "url", "url": pieces["url"]}
    elif "file" in pieces:
        ref = {"source": "file", "path": pieces["file"]}
    elif "base64" in pieces:
        ref = {
            "source": "base64",
            "data": pieces["base64"],
            "mime": pieces.get("mime", ""),
        }

    options: Dict[str, Any] = {}
    if "fps" in pieces:
        try:
            options["fps"] = float(pieces["fps"])
        except ValueError:
            options["fps"] = pieces["fps"]
    if "max_frames" in pieces:
        try:
            options["max_frames"] = int(pieces["max_frames"])
        except ValueError:
            pass
    if "format" in pieces:
        options["format"] = pieces["format"]
    if "mime" in pieces and "base64" not in pieces:
        options["mime"] = pieces["mime"]
    return ref, options


def _split_text_with_placeholders(text: str) -> List[ContentPart]:
    """Split a string into an ordered list of text / media parts."""
    if not text:
        return []
    parts: List[ContentPart] = []
    last_end = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        if m.start() > last_end:
            parts.append(ContentPart(kind="text", text=text[last_end : m.start()]))
        kind = m.group(1).lower()
        ref, options = _parse_placeholder_body(m.group("body"))
        if not ref:
            # Unrecognised body -- keep it as literal text so we never silently
            # drop user content.
            parts.append(ContentPart(kind="text", text=m.group(0)))
        else:
            # Map ``gif`` to the special expansion kind. The actual frame
            # extraction happens later in :func:`expand_gif_parts`.
            parts.append(
                ContentPart(
                    kind="gif" if kind == "gif" else kind,
                    ref=ref,
                    options=options,
                )
            )
        last_end = m.end()
    if last_end < len(text):
        parts.append(ContentPart(kind="text", text=text[last_end:]))
    return parts


# ---------------------------------------------------------------------------
# OpenAI-array parsing
# ---------------------------------------------------------------------------


def _ref_from_openai_url(url: str) -> Dict[str, Any]:
    if url.startswith("data:"):
        # data:<mime>;base64,<payload>
        head, _, payload = url[len("data:") :].partition(",")
        mime, _, encoding = head.partition(";")
        if encoding == "base64":
            return {"source": "base64", "data": payload, "mime": mime}
        # Non-base64 data URL -- treat the entire URL as opaque.
        return {"source": "url", "url": url}
    if url.startswith(("http://", "https://", "file://")):
        return {"source": "url", "url": url}
    # Bare path -- treat as file.
    return {"source": "file", "path": url}


def _parse_openai_part(part: Dict[str, Any]) -> List[ContentPart]:
    """Convert one OpenAI content part into our internal representation."""
    ptype = part.get("type")
    if ptype == "text":
        # Even within an OpenAI array, the text body may still embed legacy
        # placeholders -- expand them here so order is preserved.
        return _split_text_with_placeholders(part.get("text", "") or "")
    if ptype in ("image_url", "image"):
        url = ""
        if "image_url" in part:
            iu = part["image_url"]
            url = iu["url"] if isinstance(iu, dict) else str(iu)
        elif "image" in part:
            url = str(part["image"])
        if not url:
            return []
        return [ContentPart(kind="image", ref=_ref_from_openai_url(url))]
    if ptype == "input_audio":
        ia = part.get("input_audio", {}) or {}
        data = ia.get("data", "")
        fmt = ia.get("format", "")
        if not data:
            return []
        mime = f"audio/{fmt}" if fmt else "audio/wav"
        return [
            ContentPart(
                kind="audio",
                ref={"source": "base64", "data": data, "mime": mime},
                options={"format": fmt} if fmt else {},
            )
        ]
    if ptype in ("audio_url", "audio"):
        url = part.get("audio_url") or part.get("audio") or ""
        if isinstance(url, dict):
            url = url.get("url", "")
        if not url:
            return []
        return [ContentPart(kind="audio", ref=_ref_from_openai_url(str(url)))]
    if ptype == "video_url":
        vu = part.get("video_url", {}) or {}
        url = vu.get("url") if isinstance(vu, dict) else str(vu)
        if not url:
            return []
        opts: Dict[str, Any] = {}
        if isinstance(vu, dict):
            for k in ("fps", "max_frames"):
                if k in vu:
                    opts[k] = vu[k]
        return [ContentPart(kind="video", ref=_ref_from_openai_url(url), options=opts)]
    if ptype == "video":
        # Either a single URL/path, or a list of frame URLs (Qwen3-VL format).
        val = part.get("video")
        if isinstance(val, list):
            out: List[ContentPart] = []
            for u in val:
                if not u:
                    continue
                out.append(ContentPart(kind="image", ref=_ref_from_openai_url(str(u))))
            return out
        if isinstance(val, str):
            return [ContentPart(kind="video", ref=_ref_from_openai_url(val))]
        return []
    # Unknown types are dropped silently rather than aborting the request.
    return []


# ---------------------------------------------------------------------------
# Public API: normalize
# ---------------------------------------------------------------------------


def normalize_content(content: Any) -> List[ContentPart]:
    """Normalize a message ``content`` into an ordered :class:`ContentPart` list.

    Accepts ``None`` / ``""`` (returns an empty list), a plain string, an
    OpenAI-style list of parts, or a mix.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return _split_text_with_placeholders(content)
    if isinstance(content, list):
        out: List[ContentPart] = []
        for part in content:
            if isinstance(part, str):
                out.extend(_split_text_with_placeholders(part))
            elif isinstance(part, dict):
                out.extend(_parse_openai_part(part))
        return out
    # Fallback: coerce to string.
    return _split_text_with_placeholders(str(content))


# ---------------------------------------------------------------------------
# GIF expansion (the only piece we must process locally because vLLM does not
# treat .gif as a video container).
# ---------------------------------------------------------------------------


def _load_bytes_from_ref(ref: Dict[str, Any]) -> Optional[bytes]:
    """Best-effort loader for a media ``ref`` -- returns ``None`` on failure."""
    if not ref:
        return None
    source = ref.get("source")
    if source == "base64":
        try:
            return base64.b64decode(ref.get("data", ""))
        except Exception:
            return None
    if source == "file":
        path = ref.get("path", "")
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None
    if source == "url":
        url = ref.get("url", "")
        if not url:
            return None
        # Local file:// URL is OK to read directly.
        if url.startswith("file://"):
            path = url[len("file://") :]
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        # For http(s) URLs we deliberately do not fetch -- caller decides via
        # ``prefetch_media``.
        return None
    return None


def _expand_gif(part: ContentPart) -> List[ContentPart]:
    """Expand one GIF :class:`ContentPart` into a sequence of image parts."""
    if Image is None or ImageSequence is None:
        # Pillow unavailable -- treat the GIF as a regular image reference.
        return [ContentPart(kind="image", ref=part.ref, options=part.options)]
    data = _load_bytes_from_ref(part.ref or {})
    if data is None:
        # For URLs we do not fetch; just hand the GIF to the model as-is.
        return [ContentPart(kind="image", ref=part.ref, options=part.options)]
    try:
        img = Image.open(io.BytesIO(data))
        frames: List[Image.Image] = []
        for f in ImageSequence.Iterator(img):
            frames.append(f.convert("RGBA").copy())
    except Exception:
        return [ContentPart(kind="image", ref=part.ref, options=part.options)]

    if not frames:
        return [ContentPart(kind="image", ref=part.ref, options=part.options)]

    max_frames = int(part.options.get("max_frames", DEFAULT_GIF_MAX_FRAMES) or DEFAULT_GIF_MAX_FRAMES)
    fps = part.options.get("fps")
    if fps:
        try:
            fps_f = float(fps)
            duration_ms = int(img.info.get("duration", 100)) or 100
            stride = max(1, int(round(1000.0 / fps_f / duration_ms)))
            frames = frames[::stride]
        except Exception:
            pass
    if len(frames) > max_frames:
        # Uniformly subsample down to ``max_frames``.
        step = len(frames) / float(max_frames)
        sampled = [frames[int(i * step)] for i in range(max_frames)]
        frames = sampled

    out: List[ContentPart] = []
    for frame in frames:
        buf = io.BytesIO()
        frame.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(
            ContentPart(
                kind="image",
                ref={"source": "base64", "data": b64, "mime": "image/png"},
            )
        )
    return out


def expand_gif_parts(parts: Iterable[ContentPart]) -> List[ContentPart]:
    """Expand every ``gif`` part to ``image`` frames; keep others unchanged."""
    out: List[ContentPart] = []
    for p in parts:
        if p.kind == "gif":
            out.extend(_expand_gif(p))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Serialization to OpenAI content array
# ---------------------------------------------------------------------------


def _guess_mime_from_path(path: str, fallback: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or fallback


def _ref_to_openai_url(
    ref: Dict[str, Any],
    *,
    fallback_mime: str,
    prefetch: bool = False,
) -> str:
    """Turn a ``ref`` into the URL string used by OpenAI-style content parts.

    When ``prefetch`` is true, ``file`` refs are inlined as base64 data URLs,
    and **HTTP(S) URLs are downloaded and inlined** so that the upstream
    provider does not need to fetch them itself (solving CDN auth issues).
    ``url`` refs are NEVER fetched here -- the caller may pre-process them.
    """
    source = ref.get("source")
    if source == "url":
        url = str(ref.get("url", ""))
        if prefetch and url.startswith(("http://", "https://")):
            result = _prefetch_url(url)
            if result is not None:
                data, mime = result
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{b64}"
            # Prefetch failed -- fall back to the original URL so the request
            # doesn't completely fail; vLLM may still be able to fetch it.
            LOG.debug("Prefetch failed for %s, falling back to raw URL", url)
        return url
    if source == "file":
        path = ref.get("path", "")
        if prefetch and path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                mime = _guess_mime_from_path(path, fallback_mime)
                b64 = base64.b64encode(raw).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except OSError:
                pass
        # Conform to RFC 8089 (file://) so the remote server can read it.
        if path.startswith("/"):
            return f"file://{path}"
        return f"file://{os.path.abspath(path)}"
    if source == "base64":
        mime = ref.get("mime") or fallback_mime
        return f"data:{mime};base64,{ref.get('data','')}"
    return ""


def to_openai_content(
    parts: Iterable[ContentPart],
    *,
    prefetch_files: bool = False,
) -> Union[str, List[Dict[str, Any]]]:
    """Serialize an ordered :class:`ContentPart` list to an OpenAI content payload.

    If the list contains only text parts, returns a single concatenated string
    (compatible with text-only providers). Otherwise returns the canonical
    OpenAI list-of-parts form.
    """
    parts = list(parts)
    if not parts:
        return ""
    if all(p.kind == "text" for p in parts):
        return "".join(p.text or "" for p in parts)

    out: List[Dict[str, Any]] = []
    for p in parts:
        if p.kind == "text":
            out.append({"type": "text", "text": p.text or ""})
        elif p.kind == "image":
            url = _ref_to_openai_url(
                p.ref or {}, fallback_mime="image/png", prefetch=prefetch_files
            )
            out.append({"type": "image_url", "image_url": {"url": url}})
        elif p.kind == "audio":
            ref = p.ref or {}
            if ref.get("source") == "base64":
                mime = ref.get("mime") or "audio/wav"
                fmt = mime.split("/", 1)[-1] if "/" in mime else (p.options.get("format") or "wav")
                out.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": ref.get("data", ""), "format": fmt},
                    }
                )
            else:
                url = _ref_to_openai_url(
                    ref, fallback_mime="audio/wav", prefetch=prefetch_files
                )
                out.append({"type": "audio_url", "audio_url": {"url": url}})
        elif p.kind == "video":
            ref = p.ref or {}
            url = _ref_to_openai_url(
                ref, fallback_mime="video/mp4", prefetch=prefetch_files
            )
            video_part: Dict[str, Any] = {"type": "video_url", "video_url": {"url": url}}
            for k in ("fps", "max_frames"):
                if k in p.options:
                    video_part["video_url"][k] = p.options[k]
            out.append(video_part)
        # ``gif`` should have been expanded already; if not, treat as image.
        elif p.kind == "gif":
            url = _ref_to_openai_url(
                p.ref or {}, fallback_mime="image/gif", prefetch=prefetch_files
            )
            out.append({"type": "image_url", "image_url": {"url": url}})
    return out


def normalize_message_content(
    content: Any,
    *,
    expand_gifs: bool = True,
    prefetch_files: bool = False,
) -> Union[str, List[Dict[str, Any]]]:
    """End-to-end helper: normalize -> expand GIFs -> serialize.

    This is the function chat code should call. Returns either a plain
    string (when no multimedia is present) or an OpenAI content array.
    """
    parts = normalize_content(content)
    if expand_gifs:
        parts = expand_gif_parts(parts)
    return to_openai_content(parts, prefetch_files=prefetch_files)


def has_media(parts_or_content: Any) -> bool:
    """Whether the normalized content contains any non-text part."""
    if isinstance(parts_or_content, list) and parts_or_content and isinstance(
        parts_or_content[0], ContentPart
    ):
        return any(p.kind != "text" for p in parts_or_content)
    parts = normalize_content(parts_or_content)
    return any(p.kind != "text" for p in parts)