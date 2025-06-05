from typing import Optional, Dict

from langdetect import detect
from nltk import tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Mapping from ISO 639-1 codes to NLLB-200 language codes
LANG_CODE_MAP: Dict[str, str] = {
    'af': 'afr_Latn', 'sq': 'sqi_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab',
    'hy': 'hye_Armn', 'az': 'azj_Latn', 'be': 'bel_Cyrl', 'bn': 'ben_Beng',
    'bs': 'bos_Latn', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'zh': 'zho_Hans',
    'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant', 'hr': 'hrv_Latn', 'cs': 'ces_Latn',
    'da': 'dan_Latn', 'nl': 'nld_Latn', 'en': 'eng_Latn', 'et': 'est_Latn',
    'fi': 'fin_Latn', 'fr': 'fra_Latn', 'ka': 'kat_Geor', 'de': 'deu_Latn',
    'el': 'ell_Grek', 'gu': 'guj_Gujr', 'ht': 'hat_Latn', 'ha': 'hau_Latn',
    'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hu': 'hun_Latn', 'is': 'isl_Latn',
    'id': 'ind_Latn', 'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'kn': 'kan_Knda',
    'kk': 'kaz_Cyrl', 'ko': 'kor_Hang', 'lv': 'lvs_Latn', 'lt': 'lit_Latn',
    'mk': 'mkd_Cyrl', 'ms': 'zsm_Latn', 'mr': 'mar_Deva', 'ne': 'npi_Deva',
    'no': 'nob_Latn', 'fa': 'pes_Arab', 'pl': 'pol_Latn', 'pt': 'por_Latn',
    'pa': 'pan_Guru', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'sr': 'srp_Cyrl',
    'sk': 'slk_Latn', 'sl': 'slv_Latn', 'es': 'spa_Latn', 'sw': 'swh_Latn',
    'sv': 'swe_Latn', 'ta': 'tam_Taml', 'te': 'tel_Telu', 'th': 'tha_Thai',
    'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 'vi': 'vie_Latn'
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"

class UnlimitedTranslator:
    """Translate large texts using the NLLB neural model."""

    def __init__(self, text: str, src: Optional[str] = None, dest: str = "en") -> None:
        self.text = text
        self.src = self._resolve_lang(src or detect(text))
        self.dest = self._resolve_lang(dest)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.translated_text = self._translate_text()

    def _resolve_lang(self, lang: str) -> str:
        lang = lang.lower()
        return LANG_CODE_MAP.get(lang, lang)

    def _translate_sentence(self, sentence: str) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.dest],
                src_lang=self.src,
                max_length=1024,
            )
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _translate_text(self) -> str:
        sentences = tokenize.sent_tokenize(self.text)
        outputs = [self._translate_sentence(sent) for sent in sentences]
        return " ".join(outputs)
