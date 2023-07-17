from PIL import Image
import cairosvg
import io
import os

def crop_image(svg_string, output_file):
    # Convert SVG to PNG with higher DPI
    png_bytes = cairosvg.svg2png(bytestring=svg_string, dpi=300)

    # Load PNG into PIL Image object
    image = Image.open(io.BytesIO(png_bytes))

    # Convert the image to RGBA if it isn't already
    image = image.convert("RGBA")

    # Calculate the bounding box of non-transparent pixels
    datas = image.getdata()

    non_transparent_pixels = [
        (i % image.size[0], i // image.size[0])  # Corrected to (x,y)
        for i, item in enumerate(datas)
        if item[3] > 0  # alpha value is greater than 0
    ]

    if not non_transparent_pixels:
        print(f"{output_file}: No non-transparent pixels found.")
        return

    min_x = min(x for x, y in non_transparent_pixels)
    min_y = min(y for x, y in non_transparent_pixels)
    max_x = max(x for x, y in non_transparent_pixels)
    max_y = max(y for x, y in non_transparent_pixels)

    print(f"{output_file}: Bounding box is ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Crop image
    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    # Save the cropped image
    cropped_image.save(output_file)

# Load all SVG files in the current directory
svg_files = [f for f in os.listdir('.') if f.endswith('.svg')]

for i, svg_file in enumerate(svg_files, start=1):
    with open(svg_file, 'r') as file:
        svg_string = file.read()
    crop_image(svg_string, f"output_{i}.png")
