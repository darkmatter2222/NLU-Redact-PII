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

    Regardless of whether Faker is used or not, the name parts are styled
    using a randomly chosen letter style (all upper-case, all lower-case, or “proper”)
    and—in the case of a middle initial—the letter’s case is randomized and it may
    or may not be followed by a period.

    Args:
        use_real_name (bool): If True (and Faker is available), use Faker to generate realistic names.

    Returns:
        str: The generated name.
    """
    # Define possible name formats and choose one.
    name_formats = ["first", "last", "first_last", "first_middle_last"]
    chosen_format = random.choices(name_formats, weights=[1, 1, 2, 1])[0]

    # Choose a random style to apply to the name parts.
    style = random.choice(['upper', 'lower', 'proper'])

    def apply_style(word):
        """Apply the chosen style to a string."""
        if style == 'upper':
            return word.upper()
        elif style == 'lower':
            return word.lower()
        elif style == 'proper':
            # For "proper", randomly decide if the first letter is uppercase or lowercase.
            return (word[0].upper() if random.choice([True, False]) else word[0].lower()) + word[1:]
    
    # Initialize name parts.
    first_name = ""
    last_name = ""
    middle_initial = None

    if use_real_name and faker_available:
        fake = Faker()
        if chosen_format in ["first", "first_last", "first_middle_last"]:
            first_name = fake.first_name()
        if chosen_format in ["last", "first_last", "first_middle_last"]:
            last_name = fake.last_name()
        if chosen_format == "first_middle_last":
            # Initially choose a letter from ascii_uppercase.
            middle_initial = random.choice(string.ascii_uppercase)
    else:
        if chosen_format in ["first", "first_last", "first_middle_last"]:
            first_length = random.randint(3, 10)
            first_name = ''.join(random.choices(string.ascii_lowercase, k=first_length))
        if chosen_format in ["last", "first_last", "first_middle_last"]:
            last_length = random.randint(3, 10)
            last_name = ''.join(random.choices(string.ascii_lowercase, k=last_length))
        if chosen_format == "first_middle_last":
            # Initially choose a letter from ascii_lowercase.
            middle_initial = random.choice(string.ascii_lowercase)

    # Apply the chosen style to the first and last names (if they exist).
    if first_name:
        first_name = apply_style(first_name)
    if last_name:
        last_name = apply_style(last_name)

    # If we need a middle initial, apply the style and randomly decide whether to include a dot.
    if middle_initial is not None:
        if style == 'upper':
            middle_initial = middle_initial.upper()
        elif style == 'lower':
            middle_initial = middle_initial.lower()
        elif style == 'proper':
            middle_initial = middle_initial.upper() if random.choice([True, False]) else middle_initial.lower()
        # Randomly decide whether to add a period.
        dot = random.choice(['.', ''])
        middle_initial = f"{middle_initial}{dot}"

    # Build and return the final name based on the chosen format.
    if chosen_format == "first":
        return first_name
    elif chosen_format == "last":
        return last_name
    elif chosen_format == "first_last":
        return f"{first_name} {last_name}"
    elif chosen_format == "first_middle_last":
        return f"{first_name} {middle_initial} {last_name}"


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
