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
        Iterates over all JSON code blocks (skipping the first one if there are multiple), parses each block, 
        deduplicates candidate sentences, and collects those whose "sentence" value contains all provided 
        variable values exactly.
        
        This function is verbose: it logs its evaluation process, describes deduplication actions, 
        and explains how each decision is made.
        
        Returns one candidate selected at random among all valid candidates.
        If no valid candidate is found, returns None.
        """
        import json
        import random

        valid_candidates = []
        unique_candidates = set()  # To store unique candidate sentences
        # Skip the first block if there are multiple code blocks.
        candidates = matches[1:] if len(matches) > 1 else matches
        print(f"Total code blocks found: {len(matches)}. Evaluating {len(candidates)} candidate block(s) (skipping the first if multiple).")
        
        for idx, block in enumerate(candidates, start=1):
            print(f"\nEvaluating candidate block {idx}:")
            try:
                output_json = json.loads(block.strip())
                print(" - Successfully parsed JSON from block.")
            except json.JSONDecodeError as e:
                print(f" - Failed to parse JSON from block. Error: {e}")
                continue

            if "sentence" not in output_json:
                print(" - JSON does not contain the 'sentence' key. Skipping this block.")
                continue

            candidate = output_json["sentence"]
            if candidate in unique_candidates:
                print(f" - Duplicate candidate sentence found. Skipping duplicate: {candidate}")
                continue
            else:
                unique_candidates.add(candidate)
                print(f" - Unique candidate sentence added: {candidate}")
            
            # Check if all variable values are present in the candidate sentence.
            missing_values = [value for value in data.values() if value not in candidate]
            if missing_values:
                print(" - Candidate is missing the following variable value(s):", missing_values)
            else:
                print(" - Candidate contains all required variable values. Adding to valid candidates.")
                valid_candidates.append(candidate)

        if valid_candidates:
            selected_candidate = random.choice(valid_candidates)
            print(f"\nValid candidates found: {len(valid_candidates)}. Randomly selected candidate: {selected_candidate}")
            return selected_candidate
        else:
            print("\nNo valid candidate found that contains all variable values. Returning None.")
            return None




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
