import random
import string
import json
from datetime import datetime, timedelta

def generate_people_name():
    """Generates a fake people name: Capitalized first and last name."""
    first_length = random.randint(3, 10)
    last_length = random.randint(3, 10)
    first_name = random.choice(string.ascii_uppercase) + ''.join(random.choices(string.ascii_lowercase, k=first_length - 1))
    last_name = random.choice(string.ascii_uppercase) + ''.join(random.choices(string.ascii_lowercase, k=last_length - 1))
    return f"{first_name} {last_name}"

def generate_card_number():
    """Generates a fake card number in the format: '#### #### #### ####'."""
    return " ".join(''.join(random.choices(string.digits, k=4)) for _ in range(4))

def generate_account_number():
    """Generates a fake account number of 10 digits."""
    return ''.join(random.choices(string.digits, k=10))

def generate_ssn():
    """Generates a fake Social Security Number in the format: '###-##-####'."""
    part1 = ''.join(random.choices(string.digits, k=3))
    part2 = ''.join(random.choices(string.digits, k=2))
    part3 = ''.join(random.choices(string.digits, k=4))
    return f"{part1}-{part2}-{part3}"

def generate_government_id():
    """Generates a fake Government ID Number: two uppercase letters followed by 7 digits."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    digits = ''.join(random.choices(string.digits, k=7))
    return letters + digits

def generate_dob():
    """Generates a random Date of Birth between January 1, 1950 and December 31, 2005."""
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2005, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    dob = start_date + timedelta(days=random_days)
    return dob.strftime("%Y-%m-%d")

def generate_password():
    """Generates a fake password of 8 to 16 characters using letters, digits, and special characters."""
    allowed_chars = string.ascii_letters + string.digits + "@#$%^&*"
    length = random.randint(8, 16)
    return ''.join(random.choices(allowed_chars, k=length))

def generate_tax_id():
    """Generates a fake Tax ID Number in the format: '##-#######'."""
    part1 = ''.join(random.choices(string.digits, k=2))
    part2 = ''.join(random.choices(string.digits, k=7))
    return f"{part1}-{part2}"

def generate_phone_number():
    """Generates a fake Phone Number in the format: '(###) ###-####'."""
    area = ''.join(random.choices(string.digits, k=3))
    mid = ''.join(random.choices(string.digits, k=3))
    last = ''.join(random.choices(string.digits, k=4))
    return f"({area}) {mid}-{last}"

def generate_address():
    """Generates a fake Residential Address."""
    number = random.randint(100, 9999)
    street_names = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill"]
    street_types = ["St", "Ave", "Rd", "Blvd", "Ln", "Dr"]
    street = random.choice(street_names)
    street_type = random.choice(street_types)
    return f"{number} {street} {street_type}"

def generate_email_address():
    """Generates a fake Email Address in the format: 'xxxxx@xxx.com'."""
    username_length = random.randint(5, 10)
    domain_length = random.randint(3, 8)
    username = ''.join(random.choices(string.ascii_lowercase, k=username_length))
    domain = ''.join(random.choices(string.ascii_lowercase, k=domain_length))
    return f"{username}@{domain}.com"

def generate_ip():
    """Generates a random IPv4 address."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def generate_passport():
    """Generates a fake Passport number: one uppercase letter followed by 8 digits."""
    letter = random.choice(string.ascii_uppercase)
    digits = ''.join(random.choices(string.digits, k=8))
    return letter + digits

def generate_driver_license():
    """Generates a fake Driver License: 1-2 uppercase letters followed by 6-8 digits."""
    letter_count = random.randint(1, 2)
    letters = ''.join(random.choices(string.ascii_uppercase, k=letter_count))
    digit_count = random.randint(6, 8)
    digits = ''.join(random.choices(string.digits, k=digit_count))
    return letters + digits

def add_noise(s, noise_level=0.15):
    """
    Applies a small amount of "noise" to a string.
    
    At most one extra space, one random special character insertion,
    or removal of one space is applied with low probability.
    
    :param s: The original string.
    :param noise_level: The probability for each noise action.
    :return: A modified string with minor noise.
    """
    noisy = s

    # With low probability, insert an extra space at a random position.
    if random.random() < noise_level:
        pos = random.randint(0, len(noisy))
        noisy = noisy[:pos] + " " + noisy[pos:]

    # With low probability, insert a random special character.
    if random.random() < noise_level:
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
        pos = random.randint(0, len(noisy))
        noisy = noisy[:pos] + random.choice(special_chars) + noisy[pos:]

    # With low probability, remove a space (if any exist).
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

def main():
    executions = []
    # Generate 5 executions.
    for _ in range(5):
        execution = {}
        # Randomly decide how many fields to generate (between 0 and 5)
        num_fields = random.randint(0, 5)
        selected_fields = random.sample(list(generators.keys()), num_fields)
        
        for field in selected_fields:
            raw_value = generators[field]()
            # Apply a mild amount of noise to the generated data.
            noisy_value = add_noise(raw_value)
            execution[field] = noisy_value

        executions.append(execution)
    
    # Print the JSON list of executions.
    print(json.dumps(executions, indent=2))

if __name__ == '__main__':
    main()
