from typing import Optional, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch



# Default chat-oriented model used for prompt-based translation. The
# lightweight `openai-community/openai-gpt` model is used by default
# to avoid the heavy NLLB dependency when translating via prompts.
CHAT_MODEL = "openai-community/openai-gpt"

# Prompt template for the chat model
PROMPT_TEMPLATE = """
**Objective:** Translate the text into {lng}, preserving the original tone, style, and context.

**Translation Steps:**
1. **Identify Language and Emotional Tone:** Determine the source language and assess its emotional tone or informal style. Use English as an intermediary for better understanding if translating from Asian languages.
2. **Comprehensive Translation:** Translate all non-{lng} text fully into the target language, ensuring the translation captures the original message and any emotional nuances (e.g., urgency, irony).
3. **Retain Informal Elements:** Preserve the original emojis and symbols to maintain the text's expressive and informal nature.
4. **Proofread for Accuracy and Flow:** Review to ensure grammatical correctness and that the translated text reads naturally and contextually in {lng}.
5. **Ensure Completeness and Correct Formatting:** Convert dates, times, and numbers to adhere to {lng}'s conventional formats.

**Instructions:**
- Begin the translation output with flag emojis indicating source ➝ target languages, followed by a line break.
- Maintain original tags (e.g., <url>) and formatting within the translation.
- Output only the translated text, excluding any additional comments or notes.
- Translate all sections not originally in {lng}, ensuring no text remains untranslated if it's not in the target language.
- Format dates and numbers according to the target language's conventions, but keep the original form in parentheses.
- Translate everything and transliterate untranslatable items, placing the original in parentheses. Example: Эппл(Apple), Монхан(モンハン)

**Format Example:**
🇺🇸➝🇺🇦
{text}
"""



class PromptTranslator:
    """Translate text using a chat-oriented model and a prompt."""

    def __init__(
        self,
        text: str,
        dest: str = "en",
        model_name: str = CHAT_MODEL,
        prompt_template: str = PROMPT_TEMPLATE,
        device: Optional[str] = None,
    ) -> None:
        self.text = text
        self.dest = dest
        self.prompt_template = prompt_template
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        pipe_device = 0 if self.device.type != "cpu" else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipe_device,
        )
        self.translated_text = self._translate_text()

    def _translate_text(self) -> str:
        prompt = self.prompt_template.format(lng=self.dest, text=self.text)
        output = self.generator(prompt, max_new_tokens=512, do_sample=False)[0][
            "generated_text"
        ]
        return output[len(prompt) :].strip()
