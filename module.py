from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ⛔️ Снимаем ограничение по MPS-памяти (риск краша, но работает)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

MODEL_NAME = "models/llama-3-8b"


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
        max_new_tokens: int = 256,  # ↓ уменьшено
    ) -> None:
        self.text = text
        self.dest = dest
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Автоматический выбор устройства с приоритетом MPS
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # ↓ float16 для экономии памяти
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.translated_text = self._translate_text()

    def _translate_text(self) -> str:
        prompt = generate_prompt(self.text, self.dest)

        # ↓ Урезаем входной размер до 2048 и не переносим на GPU
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs.to(self.device),  # переносим в момент генерации
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0][inputs["input_ids"].size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
