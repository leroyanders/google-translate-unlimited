import module

# Translate a simple sentence from English to French
translator = module.UnlimitedTranslator(text="Hello, world!", src="en", dest="fr")

print(translator.translated_text)

# Translate using a chat-oriented model with a detailed prompt
chat_translator = module.PromptTranslator(text="Hello, world!", dest="fr")
print(chat_translator.translated_text)
