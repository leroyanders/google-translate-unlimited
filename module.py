from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

MODEL_NAME = "models/llama-3-8b"
DEFAULT_CONTEXT_WINDOW = 4096
TOKENIZER_MAX_LENGTH_SENTINEL = 1_000_000
DEFAULT_INPUT_TOKEN_RATIO = 0.45
DEFAULT_CHUNK_SAFETY_TOKENS = 16


def _encode(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _decode(tokenizer, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _get_context_window(tokenizer, model) -> int:
    candidates = []
    if model is not None and hasattr(model, "config"):
        for attr in ("max_position_embeddings", "max_length", "n_positions", "max_seq_len", "seq_length"):
            value = getattr(model.config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
    if tokenizer is not None:
        value = getattr(tokenizer, "model_max_length", None)
        if isinstance(value, int) and 0 < value < TOKENIZER_MAX_LENGTH_SENTINEL:
            candidates.append(value)
    return min(candidates) if candidates else DEFAULT_CONTEXT_WINDOW


def split_text_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    if max_tokens <= 0:
        return [text]
    if text == "":
        return [""]

    segments = re.findall(r"\s+|[^\s]+", text, flags=re.UNICODE)
    chunks: List[str] = []
    current = ""
    current_tokens = 0

    for segment in segments:
        segment_tokens = _encode(tokenizer, segment)
        segment_len = len(segment_tokens)

        if segment_len > max_tokens:
            if current:
                chunks.append(current)
                current = ""
                current_tokens = 0
            for i in range(0, segment_len, max_tokens):
                part_ids = segment_tokens[i:i + max_tokens]
                chunks.append(_decode(tokenizer, part_ids))
            continue

        if current_tokens + segment_len <= max_tokens:
            current += segment
            current_tokens += segment_len
        else:
            if current:
                chunks.append(current)
            current = segment
            current_tokens = segment_len

    if current:
        chunks.append(current)

    return chunks


def generate_prompt(text: str, lng: str) -> str:
    return f"""
**Objective:** Translate the text into {lng}, preserving the original tone, style, and context.

**Translation Steps:**
1. **Identify Language and Emotional Tone:** Determine the source language and assess its emotional tone or informal style. Use English as an intermediary for better understanding if translating from Asian languages.
2. **Comprehensive Translation:** Translate all non-{lng} text fully into the target language, ensuring the translation captures the original message and any emotional nuances (e.g., urgency, irony).
3. **Retain Informal Elements:** Preserve the original emojis and symbols to maintain the text's expressive and informal nature.
4. **Proofread for Accuracy and Flow:** Review to ensure grammatical correctness and that the translated text reads naturally and contextually in {lng}.
5. **Ensure Completeness and Correct Formatting:** Convert dates, times, and numbers to adhere to {lng}'s conventional formats.

**Instructions:**
- Begin the translation output with flag emojis indicating source ➝ target language.
- Maintain original tags (e.g., <url>) and formatting within the translation.
- Output only the translated text, excluding any additional comments or notes.
- Translate all sections not originally in {lng}, ensuring no text remains untranslated if it's not in the target language.
- Format dates and numbers according to the target language's conventions, but keep the original form in parentheses.
- Translate everything and transliterate untranslatable items, placing the original in parentheses. Example: Эппл(Apple), Монхан(モンハン)

---

**Text to Translate:**
{text}
""".strip()


class PromptTranslator:
    """Translate text using Meta-LLaMA-3-8B-Instruct on M1 GPU (MPS)."""

    def __init__(
        self,
        text: str,
        dest: str = "en",
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        input_token_ratio: float = DEFAULT_INPUT_TOKEN_RATIO,
        chunk_safety_tokens: int = DEFAULT_CHUNK_SAFETY_TOKENS,
        tokenizer=None,
        model=None,
        auto_translate: bool = True,
    ) -> None:
        self.text = text
        self.dest = dest
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.input_token_ratio = input_token_ratio
        self.chunk_safety_tokens = chunk_safety_tokens

        # Автоматический выбор устройства с приоритетом MPS
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = model.to(self.device) if hasattr(model, "to") else model

        self._context_window = _get_context_window(self.tokenizer, self.model)
        self._prompt_overhead_tokens = len(_encode(self.tokenizer, generate_prompt("", self.dest)))
        self._max_chunk_tokens = self._compute_max_chunk_tokens()

        self.translated_text = self._translate_text() if auto_translate else ""

    def _compute_max_chunk_tokens(self) -> int:
        available = self._context_window - self._prompt_overhead_tokens - self.chunk_safety_tokens
        if available <= 1:
            return 1
        max_input_tokens = max(1, int(available * self.input_token_ratio))
        if max_input_tokens >= available:
            max_input_tokens = available - 1
        return max_input_tokens

    def _get_max_new_tokens(self, prompt_tokens: int) -> int:
        if self._context_window <= prompt_tokens:
            return 1
        available = self._context_window - prompt_tokens
        if self.max_new_tokens is None:
            return available
        return max(1, min(self.max_new_tokens, available))

    def _split_text(self, text: str) -> List[str]:
        return split_text_by_tokens(text, self.tokenizer, self._max_chunk_tokens)

    def _translate_text(self) -> str:
        chunks = self._split_text(self.text)
        translated_chunks = [self._translate_chunk(chunk) for chunk in chunks]
        return "".join(translated_chunks)

    def _translate_chunk(self, chunk: str) -> str:
        prompt = generate_prompt(chunk, self.dest)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )

        prompt_tokens = inputs["input_ids"].size(1)
        if prompt_tokens >= self._context_window:
            if self._max_chunk_tokens <= 1:
                raise ValueError("Prompt is too long for the model context window.")
            smaller_max = max(1, self._max_chunk_tokens // 2)
            sub_chunks = split_text_by_tokens(chunk, self.tokenizer, smaller_max)
            return "".join(self._translate_chunk(sub_chunk) for sub_chunk in sub_chunks)

        max_new_tokens = self._get_max_new_tokens(prompt_tokens)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs.to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0][prompt_tokens:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
