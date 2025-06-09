#!/usr/bin/env python3
"""
Script to create a simple favicon for the HETROFL system
"""

try:
    from PIL import Image, ImageDraw
    import os

    # Create directory if it doesn't exist
    os.makedirs('hetrofl_system/gui/static', exist_ok=True)

    # Create a 32x32 image with a black background
    img = Image.new('RGB', (32, 32), color=(30, 30, 70))
    
    # Get a drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw an 'H' letter
    draw.rectangle([(5, 5), (10, 27)], fill=(230, 230, 255))
    draw.rectangle([(22, 5), (27, 27)], fill=(230, 230, 255))
    draw.rectangle([(10, 13), (22, 18)], fill=(230, 230, 255))
    
    # Save as ICO
    img.save('hetrofl_system/gui/static/favicon.ico')
    
    print("Favicon created successfully at hetrofl_system/gui/static/favicon.ico")

except ImportError:
    print("PIL (Pillow) library is required. Please install it with:")
    print("pip install pillow")
    exit(1) 