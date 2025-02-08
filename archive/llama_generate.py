import os
import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, torch_dtype):
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("[INFO] Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically assigns model parts to available devices.
        torch_dtype=torch_dtype
    )
    print("[INFO] Model loaded successfully.")
    return model, tokenizer

def generate_sentences(model, tokenizer, device):
    """
    First LLM execution:
    Generate a JSON array of exactly 5 synthetic sentences.
    Each sentence should be natural and may include random data elements chosen from the following categories:
    "People Name", "Card Number", "Account Number", "Social Security Number", "Government ID Number",
    "Date of Birth", "Password", "Tax ID Number", "Phone Number", "Residential Address", "Email Address",
    "IP Number", "Passport", and "Driver License".
    
    Output exactly a JSON array of 5 strings inside a markdown code block with the language identifier "json" and no additional commentary.
    """
    prompt = (
        "Generate a JSON array of exactly 5 synthetic sentences. "
        "Each sentence should be natural and may include random data elements chosen from the following categories: "
        "\"People Name\", \"Card Number\", \"Account Number\", \"Social Security Number\", \"Government ID Number\", "
        "\"Date of Birth\", \"Password\", \"Tax ID Number\", \"Phone Number\", \"Residential Address\", \"Email Address\", "
        "\"IP Number\", \"Passport\", and \"Driver License\". "
        "Ensure that the output is exactly a JSON array of 5 strings, enclosed in a markdown code block delimited with triple backticks and the language identifier json. "
        "Do not include any extra commentary."
    )
    
    print("[INFO] Generating synthetic sentences...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[INFO] Sentence generation complete.")
    print("[DEBUG] Raw generated sentences output:")
    print(generated_text)
    return generated_text

def extract_json_array_from_markdown(generated_text):
    """
    Attempt to extract a JSON array from a markdown code block with language 'json'.
    If no markdown block is found, try to parse the entire generated_text.
    Returns the parsed JSON (a Python list) if successful; otherwise, returns None.
    """
    matches = re.findall(r"```json\s*([\s\S]*?)\s*```", generated_text)
    json_str = None
    if matches:
        json_str = matches[0].strip()
        print("[DEBUG] Extracted JSON from markdown code block:")
        print(json_str)
    else:
        json_str = generated_text.strip()
        print("[DEBUG] No markdown block found; using full output as JSON:")
        print(json_str)
    
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        else:
            print("[ERROR] Parsed data is not a list.")
            return None
    except Exception as e:
        print("[ERROR] Exception while parsing JSON:", e)
        return None

def extract_entities(model, tokenizer, device, sentences):
    """
    Second LLM execution:
    Given a JSON array of 5 synthetic sentences, extract the data elements into JSON format.
    For each sentence, create a JSON object with exactly two keys: "sentence" and "entities". 
    The "sentence" key should contain the original sentence.
    The "entities" key should be an array of objects; each object must have two keys: "text" and "category". 
    Only include an entity if it is explicitly mentioned in the sentence, and the extracted text must exactly match what appears in the sentence.
    
    Allowed categories are: "People Name", "Card Number", "Account Number", "Social Security Number", "Government ID Number",
    "Date of Birth", "Password", "Tax ID Number", "Phone Number", "Residential Address", "Email Address", 
    "IP Number", "Passport", and "Driver License".
    
    For example, if a sentence is: 
    "John Doe, born on 1988-02-17, updated his phone number (555) 123-4567."
    then the output object should be:
    {
      "sentence": "John Doe, born on 1988-02-17, updated his phone number (555) 123-4567.",
      "entities": [
         { "text": "John Doe", "category": "People Name" },
         { "text": "(555) 123-4567", "category": "Phone Number" }
      ]
    }
    
    Output exactly a JSON array of 5 such objects inside a markdown code block (with language identifier json) with no extra commentary.
    """
    # Convert the sentences list back into a JSON-formatted string.
    sentences_json = json.dumps(sentences, indent=2)
    prompt = (
        "Given the following JSON array of 5 synthetic sentences, extract the data elements into a JSON array of objects. "
        "For each sentence, produce a JSON object with exactly two keys: \"sentence\" and \"entities\". "
        "The value for \"sentence\" must be the original sentence. "
        "The value for \"entities\" must be an array of objects, where each object has exactly two keys: \"text\" and \"category\". "
        "Only include an entity if it is explicitly mentioned in the sentence, and ensure that the extracted entity text exactly matches what appears in the sentence. "
        "Allowed categories are: \"People Name\", \"Card Number\", \"Account Number\", \"Social Security Number\", \"Government ID Number\", "
        "\"Date of Birth\", \"Password\", \"Tax ID Number\", \"Phone Number\", \"Residential Address\", \"Email Address\", "
        "\"IP Number\", \"Passport\", and \"Driver License\". "
        "For example, if a sentence is: "
        "\"John Doe, born on 1988-02-17, updated his phone number (555) 123-4567.\", "
        "the output should include an object like: { \"sentence\": \"John Doe, born on 1988-02-17, updated his phone number (555) 123-4567.\", "
        "\"entities\": [ { \"text\": \"John Doe\", \"category\": \"People Name\" }, { \"text\": \"(555) 123-4567\", \"category\": \"Phone Number\" } ] }. "
        "Output exactly a JSON array of 5 objects inside a markdown code block delimited with triple backticks and the language identifier json, with no extra commentary.\n\n"
        "Here are the sentences:\n\n"
        "```json\n" + sentences_json + "\n```"
    )
    
    print("[INFO] Extracting entities from sentences...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[INFO] Entity extraction complete.")
    print("[DEBUG] Raw generated entity extraction output:")
    print(generated_text)
    return generated_text

def validate_and_format_samples(samples):
    """
    Validates and reformats each sample to ensure it matches the expected output format.
    Each sample should be a dict with two keys: "sentence" and "entities".
    Each element in the "entities" list should be a dict with keys "text" and "category".
    Returns a list of formatted samples.
    """
    formatted = []
    for sample in samples:
        if isinstance(sample, dict) and "sentence" in sample and "entities" in sample:
            entities = sample["entities"]
            valid_entities = []
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, dict) and "text" in ent and "category" in ent:
                        valid_entities.append({
                            "text": ent["text"],
                            "category": ent["category"]
                        })
            formatted.append({
                "sentence": sample["sentence"],
                "entities": valid_entities
            })
        else:
            print("[WARNING] Skipping sample due to invalid format:", sample)
    return formatted

def append_samples_to_file(new_samples, filename=r"O:\master_data_collection\redact\synthetic_data.json"):
    # Validate and reformat the samples.
    new_samples = validate_and_format_samples(new_samples)
    print("[DEBUG] Validated and formatted new samples:")
    print(json.dumps(new_samples, indent=2))
    
    # Load existing data if file exists.
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                print("[WARNING] Existing file data is not a list. Overwriting file.")
                existing_data = []
        except Exception as e:
            print("[WARNING] Failed to load existing file. Overwriting file. Error:", e)
            existing_data = []
    else:
        existing_data = []
    
    # Append the new samples.
    existing_data.extend(new_samples)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"[INFO] Appended {len(new_samples)} samples to {filename}. Total samples now: {len(existing_data)}")
    return len(new_samples), new_samples

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        print("[INFO] GPU detected. Running on CUDA with torch.float16.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("[WARNING] No GPU detected. Running on CPU. For better performance, install a CUDA-enabled PyTorch version.")
    
    model, tokenizer = load_model(model_name, torch_dtype)
    
    iteration = 1
    while True:
        print(f"\n[INFO] Starting iteration #{iteration}")
        
        # First LLM execution: Generate synthetic sentences in JSON format.
        generated_sentences_text = generate_sentences(model, tokenizer, device)
        sentences = extract_json_array_from_markdown(generated_sentences_text)
        if sentences is None:
            print("[ERROR] No sentences extracted; skipping iteration.")
            time.sleep(2)
            iteration += 1
            continue
        
        print("[OUTPUT] Generated Sentences (parsed):")
        print(json.dumps(sentences, indent=2))
        
        # Second LLM execution: Extract entities from the sentences.
        extracted_entities_text = extract_entities(model, tokenizer, device, sentences)
        new_samples = extract_json_array_from_markdown(extracted_entities_text)
        if new_samples is None:
            print("[ERROR] No valid JSON extracted for entities; skipping iteration.")
            time.sleep(2)
            iteration += 1
            continue
        
        # If new_samples is a string, try parsing it (though extract_json_array_from_markdown should return a list)
        if isinstance(new_samples, str):
            try:
                new_samples = json.loads(new_samples)
            except Exception as e:
                print("[ERROR] Failed to parse new_samples JSON:", e)
                time.sleep(2)
                iteration += 1
                continue
        
        if not isinstance(new_samples, list):
            print("[ERROR] Extracted new_samples is not a list; skipping iteration.")
            time.sleep(2)
            iteration += 1
            continue
        
        num_appended, _ = append_samples_to_file(new_samples)
        print("[OUTPUT] Final JSON samples (validated & formatted):")
        print(json.dumps(new_samples, indent=2))
        
        time.sleep(2)
        iteration += 1

if __name__ == "__main__":
    main()
