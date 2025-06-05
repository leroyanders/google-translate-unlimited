import argparse
import module

parser = argparse.ArgumentParser(description="Translate text using NLLB or a prompt-based model")
parser.add_argument("text", help="Text to translate")
parser.add_argument("--src", dest="src", default=None, help="Source language code")
parser.add_argument("--dest", dest="dest", default="en", help="Destination language code")
parser.add_argument(
    "--engine",
    choices=["nllb", "prompt"],
    default="nllb",
    help="Translation engine to use",
)
args = parser.parse_args()

if args.engine == "prompt":
    translator = module.PromptTranslator(text=args.text, dest=args.dest)
else:
    translator = module.UnlimitedTranslator(text=args.text, src=args.src, dest=args.dest)

print(translator.translated_text)
