import argparse
from module import PromptTranslator

parser = argparse.ArgumentParser(description="Translate text using a GPT model")
parser.add_argument("--dest", dest="dest", default="en", help="Destination language code")
args = parser.parse_args()

text = """new post with links and more of diferent staff. Here it go some youtube video https://www.youtube.com/watch?v=08qJJ4RBZho   . Ok next test is google link https://www.google.com/ and with out http www.google.com and youtube same www.youtube.com/watch?v=08qJJ4RBZho and with out www youtube.com/watch?v=08qJJ4RBZho and some text to give back. Some more links https://www.pexels.com/photo/ and pexels.com/photo . Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."""

translator = PromptTranslator(text=text, dest=args.dest)
print(translator.translated_text)
