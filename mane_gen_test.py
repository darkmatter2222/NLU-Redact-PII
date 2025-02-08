import random
import string

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

# Example usage:
if __name__ == '__main__':
    # Generate a name using the custom random generator.
    print("Randomly generated name:", generate_people_name(use_real_name=False))
    
    # Generate a realistic name using Faker (if available).
    if faker_available:
        print("Faker-generated name:", generate_people_name(use_real_name=True))
    else:
        print("Faker is not installed; skipping realistic name generation.")
