from PIL import Image, ImageDraw, ImageFont

def draw_highlighted_text(text, entities, output_path="output.jpeg"):
    # Define image size
    img_width = 1000
    img_height = 300
    
    # Create blank white image
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    
    # Load a better font for UI design
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define professional pastel highlight colors for categories
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
        "Driver License": "#D2691E"   # Chocolate
    }
    
    # Split text into words
    words = text.split()
    x, y = 20, 50  # Starting position
    line_height = 40  # Space between lines
    
    for word in words:
        category = next((cat for w, cat in entities if w == word), None)
        
        # Get word size using textbbox
        word_bbox = draw.textbbox((0, 0), word, font=font)
        word_width, word_height = word_bbox[2] - word_bbox[0], word_bbox[3] - word_bbox[1]
        
        if category in colors:
            rect_x1, rect_y1 = x - 5, y - 5
            rect_x2, rect_y2 = x + word_width + 5, y + word_height + 5
            draw.rounded_rectangle(
                [(rect_x1, rect_y1), (rect_x2, rect_y2)], 
                fill=colors[category],
                radius=8
            )
            draw.text((x, y), word, fill="black", font=font)
            
            # Add category below in a lighter font
            cat_width, cat_height = draw.textbbox((0, 0), category, font=font_small)[2:]
            draw.text((x, y + word_height + 5), category, fill="gray", font=font_small)
        else:
            draw.text((x, y), word, fill="black", font=font)
        
        x += word_width + 20
        
        # Wrap text properly to next line
        if x > img_width - 100:
            x = 20
            y += line_height + 15  # Increase spacing
    
    img.save(output_path)
    img.show()

# Example input text and entities
text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
entities = [("Sebastian", "People Name"), ("Thrun", "People Name"), ("Google", "Organization"), ("2007", "Date of Birth")]

draw_highlighted_text(text, entities)
