import argparse
from module import PromptTranslator

parser = argparse.ArgumentParser(description="Translate text using a GPT model")
parser.add_argument("text", help="Text to translate")
parser.add_argument("--dest", dest="dest", default="en", help="Destination language code")
args = parser.parse_args()

translator = PromptTranslator(text=args.text, dest=args.dest)
print(translator.translated_text)
