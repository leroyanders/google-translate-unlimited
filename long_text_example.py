import argparse

from module import PromptTranslator


DEFAULT_LONG_TEXT = (
    "This is a realistic long-form text example meant to exercise chunked "
    "translation. It mixes sentences of different lengths, includes numbers "
    "like 2025 and 17.5, and uses punctuation to resemble everyday writing. "
    "The goal is to verify that the module can translate long input without "
    "truncating the content.\n\n"
    "Second paragraph: A project update. We finished the data migration on "
    "Friday and verified the records against the audit log. Next week we will "
    "roll out the UI changes, then coordinate a soft launch with customer "
    "support. If anything fails, the rollback plan is documented and tested.\n\n"
    "Final paragraph: A short request. Please review the draft, check the tone "
    "for clarity, and confirm the timeline. If you see any ambiguous terms, "
    "flag them so we can fix the wording before publishing."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a long text example")
    parser.add_argument("--dest", default="en", help="Destination language code")
    parser.add_argument("--text", default=None, help="Override the default long text")
    args = parser.parse_args()

    text = args.text if args.text is not None else DEFAULT_LONG_TEXT
    translator = PromptTranslator(text=text, dest=args.dest)
    print(translator.translated_text)


if __name__ == "__main__":
    main()
