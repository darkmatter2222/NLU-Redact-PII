import numpy as np  # random is broken from the granite model
random = np.random.default_rng()
import string
from datetime import datetime, timedelta
from faker import Faker

try:
    from faker import Faker
    faker_available = True
except ImportError:
    faker_available = False

def np_choice(seq):
    """
    Wrapper for random.choice that ensures if the input is a string,
    it is first converted to a list of characters.
    """
    if isinstance(seq, str):
        seq = list(seq)
    return random.choice(seq)

def np_choices(population, k, weights=None):
    """
    Mimics random.choices using NumPy's default_rng.
    
    Parameters:
        population (list or str): A list or string of items to choose from.
        k (int): Number of items to choose (with replacement).
        weights (list or None): A list of relative weights for each item.
    
    Returns:
        list: A list of k chosen items.
    """
    rng = np.random.default_rng()
    
    # If population is a string, convert it to a list of characters.
    if isinstance(population, str):
        population = list(population)
    
    if weights is not None:
        # Ensure weights is a numpy array of floats and normalize them.
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
    
    # Use replace=True for sampling with replacement (mimicking random.choices).
    return list(rng.choice(population, size=k, replace=True, p=weights))

def np_sample(population, k):
    """
    Mimics random.sample using NumPy's random generator.
    
    Parameters:
        population (sequence): The collection to sample from.
        k (int): The number of unique items to select.
        
    Returns:
        list: A list of k randomly selected items (without replacement).
    """
    rng = np.random.default_rng()
    # Ensure that the population is a list. If population is a string, convert it to a list of characters.
    if isinstance(population, str):
        population = list(population)
    else:
        population = list(population)
    return list(rng.choice(population, size=k, replace=False))

def randomize_case(word):
    """
    Given a string, randomly transform its case by choosing one of the following modes:
      - Fully random per character (each character is randomly upper or lower)
      - Entire string in uppercase
      - Entire string in lowercase
    """
    mode = np_choice(['random', 'upper', 'lower'])
    
    if mode == 'upper':
        return word.upper()
    elif mode == 'lower':
        return word.lower()
    else:
        # Mode 'random': Each character is randomly converted to upper or lower.
        return ''.join(np_choice([char.lower(), char.upper()]) for char in word)

def convert_alphanumerics(s):
    """
    Takes in an alphanumeric string `s`. With a 50% chance, returns `s` unchanged;
    otherwise, converts each digit to its English word equivalent with randomly varying
    letter case, and randomly adds spaces before and/or after the conversion.
    
    For example:
      "abc123" -> "abc OnE TwoThReE" (actual output may vary)
      "hello9world" -> "helloNiNeworld" (again, letter cases and spacing vary randomly)
    """
    # 50/50 decision: 50% chance to modify, 50% chance to return as-is.
    if np_choice([True, False]):
        # Mapping from digit characters to their corresponding word (no extra spaces).
        digit_mapping = {
            '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three',
            '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven',
            '8': 'Eight', '9': 'Nine'
        }
        result = []
        for char in s:
            if char.isdigit():
                # Convert the digit to its word with randomly varying case.
                converted = randomize_case(digit_mapping[char])
                # Randomly add a space before the conversion.
                if np_choice([True, False]):
                    converted = " " + converted
                # Randomly add a space after the conversion.
                if np_choice([True, False]):
                    converted = converted + " "
                result.append(converted)
            else:
                result.append(char)
        return ''.join(result)
    else:
        # Return the original string unchanged.
        return s

def generate_people_name():
    """
    Generate a person's name using one of two methods.

    Depending on randomness, the function may return:
      - Only a first name.
      - Only a last name.
      - A first and last name.
      - A first name with a middle initial and a last name.

    The method (realistic or random letters) is chosen at random (50/50 chance),
    provided the Faker library is available. Regardless of which method is used,
    the name parts are styled using a randomly chosen letter style (all upper-case,
    all lower-case, or “proper”) and—in the case of a middle initial—the letter’s
    case is randomized and it may or may not be followed by a period.

    Returns:
        str: The generated name.
    """
    # Define possible name formats and choose one.
    name_formats = ["first", "last", "first_last", "first_middle_last"]
    chosen_format = np_choices(name_formats, k=1, weights=[1, 1, 2, 1])[0]

    # Choose a random style to apply to the name parts.
    style = np_choice(['upper', 'lower', 'proper'])

    def apply_style(word):
        """Apply the chosen style to a string."""
        if style == 'upper':
            return word.upper()
        elif style == 'lower':
            return word.lower()
        elif style == 'proper':
            # For "proper", randomly decide if the first letter is uppercase or lowercase.
            return (word[0].upper() if np_choice([True, False]) else word[0].lower()) + word[1:]

    # Decide randomly whether to use a realistic name (if Faker is available) or a random one.
    use_real_name = faker_available and np_choice([True, False])

    # Initialize name parts.
    first_name = ""
    last_name = ""
    middle_initial = None

    if use_real_name:
        fake = Faker()
        if chosen_format in ["first", "first_last", "first_middle_last"]:
            first_name = fake.first_name()
        if chosen_format in ["last", "first_last", "first_middle_last"]:
            last_name = fake.last_name()
        if chosen_format == "first_middle_last":
            # Choose a middle initial from uppercase letters initially.
            middle_initial = np_choice(string.ascii_uppercase)
    else:
        if chosen_format in ["first", "first_last", "first_middle_last"]:
            first_length = int(random.integers(3, 10))
            first_name = ''.join(np_choices(string.ascii_lowercase, k=first_length))
        if chosen_format in ["last", "first_last", "first_middle_last"]:
            last_length = int(random.integers(3, 10))
            last_name = ''.join(np_choices(string.ascii_lowercase, k=last_length))
        if chosen_format == "first_middle_last":
            # Choose a middle initial from lowercase letters initially.
            middle_initial = np_choice(string.ascii_lowercase)

    # Apply the chosen style to the first and last names (if they exist).
    if first_name:
        first_name = apply_style(first_name)
    if last_name:
        last_name = apply_style(last_name)

    # If we need a middle initial, adjust its case according to the chosen style
    # and randomly decide whether to include a period.
    if middle_initial is not None:
        if style == 'upper':
            middle_initial = middle_initial.upper()
        elif style == 'lower':
            middle_initial = middle_initial.lower()
        elif style == 'proper':
            middle_initial = middle_initial.upper() if np_choice([True, False]) else middle_initial.lower()
        dot = np_choice(['.', ''])
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
    return convert_alphanumerics(" ".join(''.join(np_choices(string.digits, k=4)) for _ in range(4)))

def generate_account_number():
    return convert_alphanumerics(''.join(np_choices(string.digits, k=10)))

def generate_ssn():
    part1 = ''.join(np_choices(string.digits, k=3))
    part2 = ''.join(np_choices(string.digits, k=2))
    part3 = ''.join(np_choices(string.digits, k=4))
    return convert_alphanumerics(f"{part1}-{part2}-{part3}")

def generate_government_id():
    letters = ''.join(np_choices(string.ascii_uppercase, k=2))
    digits = ''.join(np_choices(string.digits, k=7))
    return convert_alphanumerics(letters + digits)

def generate_dob():
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2005, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = int(random.integers(0, delta_days))
    dob = start_date + timedelta(days=random_days)
    return convert_alphanumerics(dob.strftime("%Y-%m-%d"))

def generate_password():
    allowed_chars = string.ascii_letters + string.digits + "@#$%^&*"
    length = int(random.integers(8, 16))
    return convert_alphanumerics(''.join(np_choices(allowed_chars, k=length)))

def generate_tax_id():
    part1 = ''.join(np_choices(string.digits, k=2))
    part2 = ''.join(np_choices(string.digits, k=7))
    return convert_alphanumerics(f"{part1}-{part2}")

def generate_phone_number():
    area = ''.join(np_choices(string.digits, k=3))
    mid = ''.join(np_choices(string.digits, k=3))
    last = ''.join(np_choices(string.digits, k=4))
    return convert_alphanumerics(f"({area}) {mid}-{last}")

def generate_address():
    number = int(random.integers(100, 9999))
    street_names = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill"]
    street_types = ["St", "Ave", "Rd", "Blvd", "Ln", "Dr"]
    street = np_choice(street_names)
    street_type = np_choice(street_types)
    return convert_alphanumerics(f"{number} {street} {street_type}")

def generate_email_address():
    username_length = int(random.integers(5, 10))
    domain_length = int(random.integers(3, 8))
    username = ''.join(np_choices(string.ascii_lowercase, k=username_length))
    domain = ''.join(np_choices(string.ascii_lowercase, k=domain_length))
    return convert_alphanumerics(f"{username}@{domain}.com")

def generate_ip():
    # Ensure each call to random.integers is cast to int.
    return convert_alphanumerics(".".join(str(int(random.integers(0, 255))) for _ in range(4)))

def generate_passport():
    letter = np_choice(string.ascii_uppercase)
    digits = ''.join(np_choices(string.digits, k=8))
    return convert_alphanumerics(letter + digits)

def generate_driver_license():
    letter_count = int(random.integers(1, 2))
    letters = ''.join(np_choices(string.ascii_uppercase, k=letter_count))
    digit_count = int(random.integers(6, 8))
    digits = ''.join(np_choices(string.digits, k=digit_count))
    return convert_alphanumerics(letters + digits)

def add_noise(s, noise_level=0.15):
    noisy = s
    if random.random() < noise_level:
        pos = int(random.integers(0, len(noisy)))
        noisy = noisy[:pos] + " " + noisy[pos:]
    if random.random() < noise_level:
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
        pos = int(random.integers(0, len(noisy)))
        noisy = noisy[:pos] + np_choice(special_chars) + noisy[pos:]
    if random.random() < noise_level and " " in noisy:
        space_positions = [i for i, c in enumerate(noisy) if c == " "]
        pos = np_choice(space_positions)
        noisy = noisy[:pos] + noisy[pos+1:]
    return noisy
