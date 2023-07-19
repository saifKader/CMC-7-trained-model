import os
import random
from PIL import Image, ImageEnhance, ImageOps
from skimage.filters import threshold_otsu
import numpy as np


def glitch(image, glitch_factor):
    np_img = np.array(image.convert("L"))  # Convert image to grayscale

    # Binarize the image using Otsu's thresholding
    threshold = threshold_otsu(np_img)
    np_img = np_img > threshold

    # Define the maximum shift in pixels for each direction
    max_x = int(glitch_factor * image.width)
    max_y = int(glitch_factor * image.height)

    if max_x <= 0 or max_y <= 0:
        return image

    # Create random shifts for each pixel
    x_shifts = np.random.randint(-max_x, max_x, size=np_img.shape[:2])
    y_shifts = np.random.randint(-max_y, max_y, size=np_img.shape[:2])

    # Create a new array for the glitched image
    glitched_img = np.zeros_like(np_img)

    # Apply the pixel shifts to the new image
    for x in range(np_img.shape[1]):
        for y in range(np_img.shape[0]):
            shift_x = x_shifts[y, x]
            shift_y = y_shifts[y, x]

            new_x = (x + shift_x) % np_img.shape[1]
            new_y = (y + shift_y) % np_img.shape[0]

            glitched_img[new_y, new_x] = np_img[y, x]

    # Convert boolean array to int array, and then to an image
    glitched_img = Image.fromarray(glitched_img.astype(np.uint8) * 255)

    return glitched_img


def rotate_and_sharpen_image(image, angle):
    rotated_image = image.rotate(angle, Image.BICUBIC, expand=True)
    enhancer = ImageEnhance.Sharpness(rotated_image)
    sharpened_image = enhancer.enhance(2.0)
    return sharpened_image


def resize_image(image, image_size):
    aspect = image.size[0] / image.size[1]

    if aspect > 1:
        new_width = image_size[0]
        new_height = int(image_size[0] / aspect)
    else:
        new_height = image_size[1]
        new_width = int(image_size[1] * aspect)

    img = image.resize((new_width, new_height))
    new_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    new_img.paste(img, ((image_size[0] - new_width) // 2, (image_size[1] - new_height) // 2))

    enhancer = ImageEnhance.Sharpness(new_img)
    new_img = enhancer.enhance(2.0)
    return new_img


train_path = "/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/SVG_DATA/photos"
image_size = (64, 64)
angles = list(range(0, 361, 15))
glitch_levels = [i * 0.01 for i in range(4)]  # 0 to 0.03 in steps of 0.01

if os.path.isdir(train_path):
    for img_name in os.listdir(train_path):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(train_path, img_name)
            print(f'Processing image: {img_path}')
            original_img = Image.open(img_path)

            if original_img.mode != 'RGBA':
                original_img = original_img.convert('RGBA')

            for i, angle in enumerate(angles):
                for j, glitch_factor in enumerate(glitch_levels):
                    img = original_img.copy()
                    img = rotate_and_sharpen_image(img, angle)
                    img = resize_image(img, image_size)
                    img = glitch(img, glitch_factor)

                    base, ext = os.path.splitext(img_path)
                    new_img_path = f"{base}_aug_angle{angle}_glitch{j}{ext}"
                    print(f'Saving augmented image: {new_img_path}')
                    img.save(new_img_path)

print("All images augmented.")
