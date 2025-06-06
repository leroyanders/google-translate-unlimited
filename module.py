from typing import Optional, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



# Default chat-oriented model used for prompt-based translation. The
# lightweight `openai-community/openai-gpt` model is used by default
# to avoid the heavy NLLB dependency when translating via prompts.
CHAT_MODEL = "openai-community/openai-gpt"

# Prompt template for the chat model
PROMPT_TEMPLATE = (
    "Translate the following text into {lng}."
    "\n{text}\n"
)



class PromptTranslator:
    """Translate text using a chat-oriented model and a prompt."""

    def __init__(
        self,
        text: str,
        dest: str = "en",
        model_name: str = CHAT_MODEL,
        prompt_template: str = PROMPT_TEMPLATE,
        device: Optional[str] = None,
    ) -> None:
        self.text = text
        self.dest = dest
        self.prompt_template = prompt_template
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.translated_text = self._translate_text()

    def _translate_text(self) -> str:
        prompt = self.prompt_template.format(lng=self.dest, text=self.text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.n_positions,
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated_ids[0][inputs["input_ids"].size(1) :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
