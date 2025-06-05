import module

# Translate a simple sentence from English to French
translator = module.UnlimitedTranslator(text="Hello, world!", src="en", dest="fr")

print(translator.translated_text)
