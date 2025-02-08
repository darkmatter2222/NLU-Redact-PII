import re
import json
import random
import string
import torch
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Data Generators (unchanged) ---
def generate_people_name():
    first_length = random.randint(3, 10)
    last_length = random.randint(3, 10)
    first_name = random.choice(string.ascii_uppercase) + ''.join(random.choices(string.ascii_lowercase, k=first_length - 1))
    last_name = random.choice(string.ascii_uppercase) + ''.join(random.choices(string.ascii_lowercase, k=last_length - 1))
    return f"{first_name} {last_name}"

def generate_card_number():
    return " ".join(''.join(random.choices(string.digits, k=4)) for _ in range(4))

def generate_account_number():
    return ''.join(random.choices(string.digits, k=10))

def generate_ssn():
    part1 = ''.join(random.choices(string.digits, k=3))
    part2 = ''.join(random.choices(string.digits, k=2))
    part3 = ''.join(random.choices(string.digits, k=4))
    return f"{part1}-{part2}-{part3}"

def generate_government_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    digits = ''.join(random.choices(string.digits, k=7))
    return letters + digits

def generate_dob():
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2005, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    dob = start_date + timedelta(days=random_days)
    return dob.strftime("%Y-%m-%d")

def generate_password():
    allowed_chars = string.ascii_letters + string.digits + "@#$%^&*"
    length = random.randint(8, 16)
    return ''.join(random.choices(allowed_chars, k=length))

def generate_tax_id():
    part1 = ''.join(random.choices(string.digits, k=2))
    part2 = ''.join(random.choices(string.digits, k=7))
    return f"{part1}-{part2}"

def generate_phone_number():
    area = ''.join(random.choices(string.digits, k=3))
    mid = ''.join(random.choices(string.digits, k=3))
    last = ''.join(random.choices(string.digits, k=4))
    return f"({area}) {mid}-{last}"

def generate_address():
    number = random.randint(100, 9999)
    street_names = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill"]
    street_types = ["St", "Ave", "Rd", "Blvd", "Ln", "Dr"]
    street = random.choice(street_names)
    street_type = random.choice(street_types)
    return f"{number} {street} {street_type}"

def generate_email_address():
    username_length = random.randint(5, 10)
    domain_length = random.randint(3, 8)
    username = ''.join(random.choices(string.ascii_lowercase, k=username_length))
    domain = ''.join(random.choices(string.ascii_lowercase, k=domain_length))
    return f"{username}@{domain}.com"

def generate_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def generate_passport():
    letter = random.choice(string.ascii_uppercase)
    digits = ''.join(random.choices(string.digits, k=8))
    return letter + digits

def generate_driver_license():
    letter_count = random.randint(1, 2)
    letters = ''.join(random.choices(string.ascii_uppercase, k=letter_count))
    digit_count = random.randint(6, 8)
    digits = ''.join(random.choices(string.digits, k=digit_count))
    return letters + digits

def add_noise(s, noise_level=0.15):
    noisy = s
    if random.random() < noise_level:
        pos = random.randint(0, len(noisy))
        noisy = noisy[:pos] + " " + noisy[pos:]
    if random.random() < noise_level:
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
        pos = random.randint(0, len(noisy))
        noisy = noisy[:pos] + random.choice(special_chars) + noisy[pos:]
    if random.random() < noise_level and " " in noisy:
        space_positions = [i for i, c in enumerate(noisy) if c == " "]
        pos = random.choice(space_positions)
        noisy = noisy[:pos] + noisy[pos+1:]
    return noisy

# Map each field to its generator function.
generators = {
    "People Name": generate_people_name,
    "Card Number": generate_card_number,
    "Account Number": generate_account_number,
    "Social Security Number": generate_ssn,
    "Government ID Number": generate_government_id,
    "Date of Birth": generate_dob,
    "Password": generate_password,
    "Tax ID Number": generate_tax_id,
    "Phone Number": generate_phone_number,
    "Residential Address": generate_address,
    "Email Address": generate_email_address,
    "IP Number": generate_ip,
    "Passport": generate_passport,
    "Driver License": generate_driver_license
}

# --- LLM Integration ---
def extract_valid_sentence(matches, data):
    """
    Iterates over all JSON code blocks (skipping the first one if there are multiple) and
    returns the first block whose "sentence" value can be parsed.
    
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
                # If candidate contains all variable values exactly, return immediately.
                if all(value in candidate for value in data.values()):
                    return candidate
        except json.JSONDecodeError:
            continue
    return valid_candidate

def call_llm_to_generate_sentence(data):
    """
    Calls the LLM (meta-llama/Llama-3.2-3B-Instruct) with a prompt that instructs it to generate a creative,
    coherent, and grammatically correct sentence that naturally incorporates the provided variable values exactly as given.
    
    The entire output must be exactly one markdown code block tagged as `json` (and nothing else) containing valid JSON
    with a single key "sentence" whose value is the generated sentence.
    
    IMPORTANT: You must use the provided variable values *exactly as given*, preserving all spaces and punctuation, even if it looks incorrect.
    """
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = (
        "You are an AI assistant. Generate a creative, coherent, and grammatically correct sentence that naturally "
        "incorporates the following variable values exactly as provided (case-sensitive). Do not modify, omit, or alter any value, "
        "and do not use placeholders. It is essential that you use the variable values in the exact format provided, including all spaces and punctuation, even if it appears incorrect.\n\n"
        "Output your answer as exactly one markdown code block tagged as `json` and nothing else. Inside that code block, "
        "output valid JSON representing an object with a single key \"sentence\" whose value is the generated sentence. "
        "Do not include any other keys or any text outside the markdown code block.\n\n"
        "The variable values are:\n"
    )
    for key, value in data.items():
        prompt += f"- {key}: {value}\n"
    prompt += "\nOutput exactly one markdown code block with the required JSON."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Use regex to extract all markdown code blocks tagged with json.
    matches = re.findall(r"```json\s*([\s\S]+?)\s*```", generated_text, re.DOTALL)
    sentence = extract_valid_sentence(matches, data)
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

# --- Main Flow ---
def main():
    # Randomly decide how many fields to generate (between 0 and 5).
    num_fields = random.randint(1, 5)
    selected_fields = random.sample(list(generators.keys()), num_fields)

    data = {}  # Holds field names and their (possibly noised) values.
    print("Generated Fields:")
    for field in selected_fields:
        raw_value = generators[field]()
        noisy_value = add_noise(raw_value)
        data[field] = noisy_value
        print(f"- {field}: {noisy_value}")

    if not data:
        print("\nNo variables generated. Skipping LLM sentence generation.")
        return

    sentence, raw_llm_output = call_llm_to_generate_sentence(data)
    print("\nLLM Generated Sentence (raw output):")
    print(raw_llm_output)
    print("\nParsed Sentence from JSON:")
    print(sentence)

    missing = validate_sentence(sentence, data)
    if missing:
        print("\nValidation FAILED. The following variables were not found exactly in the sentence:")
        for key, value in missing:
            print(f"- {key}: {value}")
    else:
        print("\nValidation SUCCESSFUL: all variable values are present in the sentence.")

if __name__ == '__main__':
    main()
