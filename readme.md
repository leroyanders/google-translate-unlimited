# About
<p>This python script allows you translate any text in any language directly by Google Translate, without using paid API services. <br>

+ <b>No limits.</b>
+ <b>Fully free.</b>
+ <b>Supports all 7,151 languages.</b>

</p>

# Install dependencies
```$
$ pip3 install -r requirements.txt
```

# Using
```python
from .module import Translator

tr = Translator('Hello, World!', src='en', dest='fr')
translated_text = tr.translated_text

print(f'Result: {translated_text}');

```
The result will be:
``` Result: Bonjour le monde! ```
