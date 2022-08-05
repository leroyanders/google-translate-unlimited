from googletrans import Translator
from nltk import tokenize
from deep_translator import GoogleTranslator

class UnlimitedTranslator:
    def __init__(self, text_, src=None, dest='en'):
      
        # Tokenize source and assign translator
        self.translated_text = ''
        self.token = tokenize.sent_tokenize(text_)
        self.translator = GoogleTranslator(source=src, target=dest)
        
        # Limits and offsets
        list_of_lines = []
        max_length = 5000
        
        # If length more than limit(15000), trim in parts and push them to array
        if len(text_) > max_length:
            # split the text into lines
            while len(text_) > max_length:
                line_length = text_[:max_length].rfind('.')                
                list_of_lines.append(text_[:line_length])
                text_ = text_[line_length + 1:]
                
            # Will translate all parts of text
            for line in list_of_lines:
                if src is None:
                    self.translated_text += self.translator.translate(line)
                else:
                    self.translated_text += self.translator.translate(line)
        else:
            if src is None:
                self.translated_text += self.translator.translate(text_)
            else:
                self.translated_text += self.translator.translate(text_)
