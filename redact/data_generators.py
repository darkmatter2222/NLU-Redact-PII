import random
import string
from datetime import datetime, timedelta

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
