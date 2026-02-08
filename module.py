from __future__ import annotations

from typing import List, Optional

try:
    from googletrans import Translator
except Exception:  # pragma: no cover - optional dependency in tests
    Translator = None


DEFAULT_MAX_LENGTH = 15000
BOUNDARY_CHARS = (
    "\n",
    ".",
    "!",
    "?",
    ";",
    ":",
    ",",
    " ",
    "。",
    "！",
    "？",
    "、",
    "，",
    "；",
    "：",
)


def split_text_by_length(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> List[str]:
    """Split text into chunks no longer than max_length characters.

    Prefers to split on punctuation/whitespace when possible, but always
    falls back to a hard split to support texts without punctuation.
    """
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if text == "":
        return [""]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        cut = _find_split_index(remaining, max_length)
        if cut <= 0:
            cut = max_length

        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks


class UnlimitedTranslator:
    def __init__(
        self,
        text_: str,
        src: Optional[str] = None,
        dest: str = "en",
        max_length: int = DEFAULT_MAX_LENGTH,
        translator: Optional[object] = None,
    ) -> None:
        if text_ == "":
            self.translated_text = ""
            return

        if translator is None:
            if Translator is None:
                raise ImportError(
                    "googletrans is required. Install it or pass a custom translator."
                )
            translator = Translator()

        self.translated_text = ""
        self.translator = translator
        self.max_length = max_length

        parts = split_text_by_length(text_, max_length=max_length)
        translated_parts = []
        for part in parts:
            if src is None:
                result = self.translator.translate(part, dest=dest)
            else:
                result = self.translator.translate(part, dest=dest, src=src)
            translated_parts.append(_extract_text(result))

        self.translated_text = "".join(translated_parts)


def _find_split_index(text: str, max_length: int) -> int:
    window = text[:max_length]
    split_at = -1
    for boundary in BOUNDARY_CHARS:
        idx = window.rfind(boundary)
        if idx > split_at:
            split_at = idx

    if split_at < 1:
        return max_length
    return split_at + 1


def _extract_text(result: object) -> str:
    if hasattr(result, "text"):
        return getattr(result, "text")
    return str(result)
