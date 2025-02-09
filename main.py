import random, time
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
        """"You are an AI assistant. Your task is to generate a creative, coherent, and grammatically correct sentence that naturally incorporates the following variable values exactly as provided (case-sensitive). You must include each variable value exactly as provided, including all spaces, punctuation, and special characters. Do not modify, omit, or alter any value.

IMPORTANT:
1. You are permitted to internally generate and consider several variants. However, your final answer must be exactly one markdown code block tagged with `json` and nothing else.
2. Inside that markdown code block, output valid JSON representing an object with a single key "sentence". The value associated with the "sentence" key must be the generated sentence.
3. Do not include any other keys or any additional text outside the code block.
4. Before submitting, ensure that every variable value below appears exactly as provided in your sentence. Do not use placeholders or additional keys.
"""
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
    
    # Scoreboard to keep track of actions and performance.
    score = {
        "iterations": 0,
        "fields_generated": 0,
        "validation_success": 0,
        "validation_failure": 0,
        "entries_appended": 0,
        "total_sentence_gen_time": 0.0,  # Total time spent generating sentences (in seconds).
        "sentence_generation_count": 0,  # Number of sentence generation calls.
    }
    
    start_time = time.time()  # Record the start time for rate calculations.
    
    while True:
        score["iterations"] += 1
        try:
            # Randomly decide how many fields to generate (between 1 and 5).
            num_fields = random.randint(1, 5)
            selected_fields = random.sample(list(generators.keys()), num_fields)
    
            data = {}  # Holds field names and their (possibly noised) values.
            print("\nGenerated Fields:")
            for field in selected_fields:
                raw_value = generators[field]()
                noisy_value = add_noise(raw_value)
                data[field] = noisy_value
                print(f"- {field}: {noisy_value}")
            
            # Update score for fields generated in this iteration.
            score["fields_generated"] += len(data)
            
            # Calculate elapsed time in minutes.
            elapsed_minutes = (time.time() - start_time) / 60.0
            # Calculate entries appended per minute (avoid division by zero).
            entries_per_minute = score["entries_appended"] / elapsed_minutes if elapsed_minutes > 0 else score["entries_appended"]
            
            # Calculate validation rates.
            total_validations = score["validation_success"] + score["validation_failure"]
            if total_validations > 0:
                success_rate = (score["validation_success"] / total_validations) * 100
                failure_rate = (score["validation_failure"] / total_validations) * 100
            else:
                success_rate = failure_rate = 0.0
            
            # Calculate average sentence generation time in milliseconds.
            if score["sentence_generation_count"] > 0:
                avg_sentence_gen_time_ms = (score["total_sentence_gen_time"] / score["sentence_generation_count"]) * 1000
            else:
                avg_sentence_gen_time_ms = 0.0

            # Pretty-print the scoreboard table just before building the custom prompt.
            try:
                from tabulate import tabulate
                table_data = [
                    ["Iteration", score["iterations"]],
                    ["Fields this Iteration", len(data)],
                    ["Total Fields Generated", score["fields_generated"]],
                    ["Validation Successes", score["validation_success"]],
                    ["Validation Failures", score["validation_failure"]],
                    ["Entries Appended", score["entries_appended"]],
                    ["Entries per Minute", f"{entries_per_minute:.2f}"],
                    ["Validation Success Rate", f"{success_rate:.2f}%"],
                    ["Validation Failure Rate", f"{failure_rate:.2f}%"],
                    ["Avg Sentence Gen Time (ms)", f"{avg_sentence_gen_time_ms:.2f}"]
                ]
                print("\nScoreboard:")
                print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))
            except ImportError:
                # Fallback if tabulate isn't installed.
                print("\nScoreboard:")
                print(f"Iteration: {score['iterations']}")
                print(f"Fields this Iteration: {len(data)}")
                print(f"Total Fields Generated: {score['fields_generated']}")
                print(f"Validation Successes: {score['validation_success']}")
                print(f"Validation Failures: {score['validation_failure']}")
                print(f"Entries Appended: {score['entries_appended']}")
                print(f"Entries per Minute: {entries_per_minute:.2f}")
                print(f"Validation Success Rate: {success_rate:.2f}%")
                print(f"Validation Failure Rate: {failure_rate:.2f}%")
                print(f"Avg Sentence Gen Time (ms): {avg_sentence_gen_time_ms:.2f}")
    
            # Build custom prompt.
            custom_prompt = build_custom_prompt(data)
    
            # Time the sentence generation.
            sentence_gen_start = time.time()
            sentence, raw_llm_output = llama.generate_sentence(data, custom_prompt=custom_prompt)
            sentence_gen_end = time.time()
    
            # Update the sentence generation time KPI.
            gen_duration = sentence_gen_end - sentence_gen_start
            score["total_sentence_gen_time"] += gen_duration
            score["sentence_generation_count"] += 1
    
            missing = validate_sentence(sentence, data)
            if missing:
                print("\nValidation FAILED. The following variables were not found exactly in the sentence:")
                for key, value in missing:
                    print(f"- {key}: {value}")
                print("Retrying...\n")
                score["validation_failure"] += 1
                continue  # Try again.
            else:
                print("\nValidation SUCCESSFUL: all variable values are present in the sentence.")
                score["validation_success"] += 1
    
            # Build the entry and append it to the JSON file.
            entry = build_entry(sentence, data)
            writer.append_entry(entry)
            print("Entry appended to synthetic_data.json.\n")
            score["entries_appended"] += 1
            
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...\n")
        
        # Optionally, wait a moment before the next iteration.
        # For example: time.sleep(1)


if __name__ == '__main__':
    main()
