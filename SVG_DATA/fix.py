import os
from PIL import Image, ImageEnhance

# Path to the directory with the images
train_path = "/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/SVG_DATA/photos"

# New size of images
image_size = (64, 64)

# Traverse through each directory
for class_folder in os.listdir(train_path):
    class_path = os.path.join(train_path, class_folder)

    # Check if path is a directory
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):

            # Check if file is an image (jpg or png)
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(class_path, img_name)

                # Load the image
                img = Image.open(img_path)

                # If image is not in RGBA mode, convert it
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # Calculate the aspect ratio
                aspect = img.size[0] / img.size[1]

                # Resize while keeping the aspect ratio
                if aspect > 1:  # Width is greater than height
                    new_width = image_size[0]
                    new_height = int(image_size[0] / aspect)
                else:  # Height is greater than width
                    new_height = image_size[1]
                    new_width = int(image_size[1] * aspect)

                img = img.resize((new_width, new_height))

                # Create a new transparent image
                new_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
                new_img.paste(img, ((image_size[0] - new_width) // 2, (image_size[1] - new_height) // 2))

                # Increase sharpness
                enhancer = ImageEnhance.Sharpness(new_img)
                new_img = enhancer.enhance(2.0)

                # Save resized and enhanced image (overwrite original image)
                new_img.save(img_path)

print("All images resized and enhanced.")
