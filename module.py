from typing import Optional, Dict

from langdetect import detect
from nltk import tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
import torch

# Mapping from ISO 639-1 codes to NLLB-200 language codes
LANG_CODE_MAP: Dict[str, str] = {
    "af": "afr_Latn",
    "sq": "sqi_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "hy": "hye_Armn",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bn": "ben_Beng",
    "bs": "bos_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "hr": "hrv_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "en": "eng_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "ka": "kat_Geor",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "gu": "guj_Gujr",
    "ht": "hat_Latn",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hu": "hun_Latn",
    "is": "isl_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "kn": "kan_Knda",
    "kk": "kaz_Cyrl",
    "ko": "kor_Hang",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "mk": "mkd_Cyrl",
    "ms": "zsm_Latn",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "no": "nob_Latn",
    "fa": "pes_Arab",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "pa": "pan_Guru",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sr": "srp_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "es": "spa_Latn",
    "sw": "swh_Latn",
    "sv": "swe_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Default chat-oriented model used for prompt-based translation
CHAT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

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


class UnlimitedTranslator:
    """Translate large texts using the NLLB neural model."""

    def __init__(self, text: str, src: Optional[str] = None, dest: str = "en") -> None:
        self.text = text
        self.src = self._resolve_lang(src or detect(text))
        self.dest = self._resolve_lang(dest)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.src_lang = self.src
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.translated_text = self._translate_text()

    def _resolve_lang(self, lang: str) -> str:
        lang = lang.lower()
        return LANG_CODE_MAP.get(lang, lang)

    def _get_lang_id(self, lang: str) -> int:
        """Return the token ID used for the given language code."""
        if hasattr(self.tokenizer, "lang_code_to_id"):
            return self.tokenizer.lang_code_to_id[lang]
        return self.tokenizer.convert_tokens_to_ids(lang)

    def _translate_sentence(self, sentence: str) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            bos_token_id = self._get_lang_id(self.dest)
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=bos_token_id,
                max_length=1024,
            )
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _translate_text(self) -> str:
        sentences = tokenize.sent_tokenize(self.text)
        outputs = [self._translate_sentence(sent) for sent in sentences]
        return " ".join(outputs)


class PromptTranslator:
    """Translate text using a chat-oriented model and a prompt."""

    def __init__(
        self,
        text: str,
        dest: str = "en",
        model_name: str = CHAT_MODEL,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        self.text = text
        self.dest = dest
        self.prompt_template = prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        self.translated_text = self._translate_text()

    def _translate_text(self) -> str:
        prompt = self.prompt_template.format(lng=self.dest, text=self.text)
        output = self.generator(prompt, max_new_tokens=512, do_sample=False)[0][
            "generated_text"
        ]
        return output[len(prompt) :].strip()
