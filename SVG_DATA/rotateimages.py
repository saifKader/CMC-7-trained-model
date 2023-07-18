import os
import math
from PIL import Image
from PIL import ImageEnhance


def rotate_and_sharpen_image(input_path, output_path, angle):
    image = Image.open(input_path)
    rotated_image = image.rotate(angle, Image.BICUBIC, expand=True)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(rotated_image)
    sharpened_image = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2.0

    sharpened_image.save(output_path)

# Path to the directory with the images
classes_path = "/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/train"

# List of angles to rotate the image
angles = list(range(0, 361, 15))

for class_folder in os.listdir(classes_path):
    class_path = os.path.join(classes_path, class_folder)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                input_path = os.path.join(class_path, filename)
                for angle in angles:
                    output_filename = f'{filename.rsplit(".", 1)[0]}_{angle}.{filename.rsplit(".", 1)[1]}'
                    output_path = os.path.join(class_path, output_filename)
                    rotate_and_sharpen_image(input_path, output_path, angle)
