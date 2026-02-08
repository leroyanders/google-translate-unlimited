import unittest

import module


class DummyTokenizer:
    def __init__(self, model_max_length: int = 64) -> None:
        self.model_max_length = model_max_length

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):
        return "".join(chr(token_id) for token_id in token_ids)


class ChunkingTests(unittest.TestCase):
    def test_split_text_preserves_content(self):
        tokenizer = DummyTokenizer()
        text = "Hi 👋 Привет こんにちは\nSymbols: <> & © ✓ — end."

        chunks = module.split_text_by_tokens(text, tokenizer, max_tokens=10)

        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(tokenizer.encode(chunk)) <= 10 for chunk in chunks))

    def test_split_text_handles_long_unbroken_segments(self):
        tokenizer = DummyTokenizer()
        text = "A" * 105

        chunks = module.split_text_by_tokens(text, tokenizer, max_tokens=25)

        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 25 for chunk in chunks))


class DummyTranslator(module.PromptTranslator):
    def __init__(self, text: str, max_chunk_tokens: int) -> None:
        self.text = text
        self.dest = "en"
        self.tokenizer = DummyTokenizer()
        self._context_window = 128
        self._prompt_overhead_tokens = 10
        self._max_chunk_tokens = max_chunk_tokens

    def _translate_chunk(self, chunk: str) -> str:
        return f"<{chunk}>"


class TranslatorChunkingTests(unittest.TestCase):
    def test_translate_text_uses_chunking(self):
        text = "Hello world " * 20
        translator = DummyTranslator(text=text, max_chunk_tokens=10)

        result = translator._translate_text()

        self.assertTrue(result.startswith("<"))
        self.assertTrue(result.endswith(">"))
        self.assertIn("Hello", result)


if __name__ == "__main__":
    unittest.main()
