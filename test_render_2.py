from PIL import Image, ImageDraw, ImageFont
import textwrap

def get_highlighted_text_image_wrapped(text, entities):
    """
    Creates and returns a PIL Image that renders the full text using text wrapping.
    For each wrapped line, if an entity phrase appears (as a substring), that substring is
    highlighted with a pastel-colored rounded rectangle and its category label is drawn below.
    
    Parameters:
      text: The full text to render.
      entities: A list of tuples (entity_text, category). (Entity phrases may contain spaces.)
    """
    # Define image dimensions and wrapping parameters
    img_width = 1000
    wrap_width = 80  # maximum number of characters per line
    wrapped_lines = textwrap.wrap(text, width=wrap_width)
    line_height = 30
    img_height = 40 + line_height * len(wrapped_lines)
    
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define pastel colors for categories.
    colors = {
        "People Name": "#FFB6C1",   # Light Pink
        "Card Number": "#FFD700",   # Gold
        "Account Number": "#FFA07A",   # Light Salmon
        "Social Security Number": "#FA8072",   # Salmon
        "Government ID Number": "#FF8C00",   # Dark Orange
        "Date of Birth": "#98FB98",   # Pale Green
        "Password": "#8A2BE2",   # Blue Violet
        "Tax ID Number": "#DC143C",   # Crimson
        "Phone Number": "#32CD32",   # Lime Green
        "Residential Address": "#4682B4",   # Steel Blue
        "Email Address": "#87CEEB",   # Sky Blue
        "IP Number": "#20B2AA",   # Light Sea Green
        "Passport": "#A020F0",   # Purple
        "Driver License": "#D2691E",   # Chocolate,
        "Organization": "#C0C0C0"  # Silver
    }
    
    # We assume entities is a list of tuples (entity_text, category)
    y = 20
    for line in wrapped_lines:
        x = 20
        # Draw the line normally first.
        draw.text((x, y), line, fill="black", font=font)
        # For each entity, check if it appears in this line.
        for entity_text, category in entities:
            index = line.find(entity_text)
            if index != -1:
                # Compute x coordinate of the entity substring within the line.
                prefix = line[:index]
                prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                x_start = x + (prefix_bbox[2] - prefix_bbox[0])
                # Compute the size of the entity substring.
                entity_bbox = draw.textbbox((0, 0), entity_text, font=font)
                entity_width = entity_bbox[2] - entity_bbox[0]
                entity_height = entity_bbox[3] - entity_bbox[1]
                # Draw a rounded rectangle behind the entity.
                padding = 2
                rect = [x_start - padding, y - padding, x_start + entity_width + padding, y + entity_height + padding]
                draw.rounded_rectangle(rect, fill=colors.get(category, "#FFB6C1"), radius=5)
                # Redraw the entity text on top.
                draw.text((x_start, y), entity_text, fill="black", font=font)
                # Draw the category label below the entity in a smaller gray font.
                draw.text((x_start, y + entity_height + 2), category, fill="gray", font=font_small)
        y += line_height
    return img


get_highlighted_text_image_wrapped("yqxhedqfe@snuezoco. com was thrilled to receive her new credit card, which featured her Card Number, 1102 19069681 0973, and a unique Account Number, 9497059768, on the back. Her Social Security Number, 47;6-49-7457, was also printed for security purposes. The email address associated with the card, yqxhedqfe@snuezoco. com, allowed her to access her account online and monitor her transactions.", [('yqxhedqfe@snuezoco. com', 'Email Address')]).show()