import random
# Import data generators and helper functions from your package.
from redact.data_generators import (
    generate_people_name,
    generate_card_number,
    generate_account_number,
    generate_ssn,
    generate_government_id,
    generate_dob,
    generate_password,
    generate_tax_id,
    generate_phone_number,
    generate_address,
    generate_email_address,
    generate_ip,
    generate_passport,
    generate_driver_license,
    add_noise
)
from redact.llama_integration import LlamaGenerator, validate_sentence
from redact.synthetic_data import SyntheticDataWriter

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

def build_custom_prompt(data):
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
    return prompt

def build_entry(sentence, data):
    """
    Constructs an entry with the generated sentence and a list of entity objects.
    Each entity object includes the text (value) and its category (the generator key).
    """
    entities = [{"text": value, "category": key} for key, value in data.items()]
    return {"sentence": sentence, "entities": entities}

def main():
    # Create an instance of the LlamaGenerator and SyntheticDataWriter.
    llama = LlamaGenerator()
    file_path = r"O:\master_data_collection\redact\synthetic_data.json"
    writer = SyntheticDataWriter(file_path)
    
    while True:
        try:
            # Randomly decide how many fields to generate (between 1 and 5).
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
                print("\nNo variables generated. Retrying...\n")
                continue
    
            custom_prompt = build_custom_prompt(data)
    
            sentence, raw_llm_output = llama.generate_sentence(data, custom_prompt=custom_prompt)
    
            print("\nLLM Generated Sentence (raw output):")
            print(raw_llm_output)
            print("\nParsed Sentence from JSON:")
            print(sentence)
    
            missing = validate_sentence(sentence, data)
            if missing:
                print("\nValidation FAILED. The following variables were not found exactly in the sentence:")
                for key, value in missing:
                    print(f"- {key}: {value}")
                print("Retrying...\n")
                continue  # try again
            else:
                print("\nValidation SUCCESSFUL: all variable values are present in the sentence.")
    
            # Build the entry and append it to the JSON file.
            entry = build_entry(sentence, data)
            writer.append_entry(entry)
            print("Entry appended to synthetic_data.json.\n")
            
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...\n")
        
        # Optionally, wait a moment before the next iteration.
        # For example: time.sleep(1)

if __name__ == '__main__':
    main()
