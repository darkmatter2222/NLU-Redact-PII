import random
import string
from datetime import datetime, timedelta
from faker import Faker

try:
    from faker import Faker
    faker_available = True
except ImportError:
    faker_available = False


def generate_people_name(use_real_name=False):
    """
    Generate a person's name using one of two methods.
    
    Depending on the input and randomness, the function may return:
      - Only a first name.
      - Only a last name.
      - A first and last name.
      - A first name with a middle initial and a last name.
    
    If `use_real_name` is True and the Faker library is available, the name parts are
    generated using realistic names. Otherwise, each name part is randomly constructed
    from letters with a randomly chosen style (all upper-case, all lower-case, or a "proper"
    style where the first letter is randomly capitalized).
    
    Args:
        use_real_name (bool): If True (and Faker is available), use Faker to generate realistic names.
    
    Returns:
        str: The generated name.
    """
    # Define the possible name formats.
    # We assign weights so that a full name is more common than a single name.
    name_formats = ["first", "last", "first_last", "first_middle_last"]
    chosen_format = random.choices(name_formats, weights=[1, 1, 2, 1])[0]
    
    # If using Faker and it's available, generate realistic name parts.
    if use_real_name and faker_available:
        fake = Faker()
        if chosen_format == "first":
            return fake.first_name()
        elif chosen_format == "last":
            return fake.last_name()
        elif chosen_format == "first_last":
            return f"{fake.first_name()} {fake.last_name()}"
        elif chosen_format == "first_middle_last":
            # For a middle initial, we simply choose a random uppercase letter.
            middle_initial = random.choice(string.ascii_uppercase)
            return f"{fake.first_name()} {middle_initial}. {fake.last_name()}"
    
    # Otherwise, generate a random name using random letters.
    # Generate random first and last name components.
    first_length = random.randint(3, 10)
    last_length = random.randint(3, 10)
    first_name = ''.join(random.choices(string.ascii_lowercase, k=first_length))
    last_name = ''.join(random.choices(string.ascii_lowercase, k=last_length))
    
    # Randomly choose a style for letter casing.
    # 'upper': all letters become uppercase.
    # 'lower': letters remain lowercase.
    # 'proper': only the first letter is randomly chosen to be uppercase or lowercase.
    style = random.choice(['upper', 'lower', 'proper'])
    
    def apply_style(word):
        if style == 'upper':
            return word.upper()
        elif style == 'lower':
            return word.lower()
        elif style == 'proper':
            return (word[0].upper() if random.choice([True, False]) else word[0].lower()) + word[1:]
    
    # Apply the chosen style to the generated first and last names.
    first_name = apply_style(first_name)
    last_name = apply_style(last_name)
    
    # Decide which format to return.
    if chosen_format == "first":
        return first_name
    elif chosen_format == "last":
        return last_name
    elif chosen_format == "first_last":
        return f"{first_name} {last_name}"
    elif chosen_format == "first_middle_last":
        # Generate a middle initial (a single random letter) and apply the same style.
        middle_initial = random.choice(string.ascii_lowercase)
        if style == 'upper':
            middle_initial = middle_initial.upper()
        elif style == 'lower':
            middle_initial = middle_initial.lower()
        elif style == 'proper':
            middle_initial = middle_initial.upper() if random.choice([True, False]) else middle_initial.lower()
        return f"{first_name} {middle_initial}. {last_name}"


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
