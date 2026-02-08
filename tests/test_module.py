import unittest

import module


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):
        return "".join(chr(token_id) for token_id in token_ids)


class ModuleTests(unittest.TestCase):
    def test_generate_prompt_includes_language_and_text(self):
        text = "Hello 🌍"
        prompt = module.generate_prompt(text, "fr")

        self.assertIn("Translate the text into fr", prompt)
        self.assertIn(text, prompt)
        self.assertTrue(prompt.endswith(text))

    def test_generate_prompt_includes_source_language_when_provided(self):
        text = "Bonjour"
        prompt = module.generate_prompt(text, "en", src="fr")

        self.assertIn("Translate the text from fr into en", prompt)
        self.assertTrue(prompt.endswith(text))

    def test_split_text_empty_string(self):
        tokenizer = DummyTokenizer()
        chunks = module.split_text_by_tokens("", tokenizer, max_tokens=10)
        self.assertEqual(chunks, [""])


class UppercaseTranslator(module.PromptTranslator):
    def __init__(self, text: str, max_chunk_tokens: int) -> None:
        self.text = text
        self.dest = "en"
        self.tokenizer = DummyTokenizer()
        self._context_window = 256
        self._prompt_overhead_tokens = 10
        self._max_chunk_tokens = max_chunk_tokens

    def _translate_chunk(self, chunk: str) -> str:
        return chunk.upper()


class LongTextTranslationTests(unittest.TestCase):
    def test_long_text_translation_example(self):
        long_text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        ) * 8

        translator = UppercaseTranslator(text=long_text, max_chunk_tokens=40)
        translated = translator._translate_text()

        self.assertEqual(translated, long_text.upper())


class EchoTranslator(module.PromptTranslator):
    def __init__(self, text: str, max_chunk_tokens: int) -> None:
        self.text = text
        self.dest = "en"
        self.tokenizer = DummyTokenizer()
        self._context_window = 128
        self._prompt_overhead_tokens = 10
        self._max_chunk_tokens = max_chunk_tokens

    def _translate_chunk(self, chunk: str) -> str:
        return chunk


class SymbolTranslationTests(unittest.TestCase):
    def test_translate_text_handles_symbols_and_length(self):
        text = ("Hi 😀 Привет こんにちは\n" * 40) + "Symbols: <> & © ✓ — end."
        translator = EchoTranslator(text=text, max_chunk_tokens=9)

        translated = translator._translate_text()

        self.assertEqual(translated, text)


if __name__ == "__main__":
    unittest.main()
