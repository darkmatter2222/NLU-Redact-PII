import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", max_new_tokens=1000, do_sample=True, top_p=0.9, temperature=0.7):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def extract_valid_sentence(matches, data):
        """
        Iterates over all JSON code blocks (skipping the first one if there are multiple) and
        returns the first candidate whose "sentence" value can be parsed.
        
        Additionally, if a candidate contains all provided variable values exactly, return it immediately.
        Otherwise, return the last valid candidate found.
        """
        valid_candidate = None
        candidates = matches[1:] if len(matches) > 1 else matches
        for block in candidates:
            try:
                output_json = json.loads(block.strip())
                if "sentence" in output_json:
                    candidate = output_json["sentence"]
                    valid_candidate = candidate  # update fallback candidate
                    if all(value in candidate for value in data.values()):
                        return candidate
            except json.JSONDecodeError:
                continue
        return valid_candidate

    def generate_sentence(self, data, custom_prompt):
        """
        Uses the provided custom prompt to generate a sentence.
        The prompt must instruct the model to output exactly one markdown code block tagged as `json`
        containing valid JSON with a single key "sentence".
        """
        prompt = custom_prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        # Extract all markdown code blocks tagged with json.
        matches = re.findall(r"```json\s*([\s\S]+?)\s*```", generated_text, re.DOTALL)
        sentence = self.extract_valid_sentence(matches, data)
        if not sentence:
            print("No valid JSON block with a 'sentence' key that contains all variable values was found.")
            sentence = ""
        return sentence, generated_text

def validate_sentence(sentence, data):
    """
    Validates that each provided variable value is present exactly (character-for-character) in the generated sentence.
    Returns a list of (key, value) tuples for which the value was not found.
    """
    missing = []
    for key, value in data.items():
        if value not in sentence:
            missing.append((key, value))
    return missing
