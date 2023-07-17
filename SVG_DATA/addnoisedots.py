import cv2
import os
import numpy as np


# Define the function to add random black dots to an image
def add_black_dots(image, num_dots=100):
    out = np.copy(image)

    # Generate random positions for the dots
    xs = np.random.randint(0, image.shape[1], size=num_dots)
    ys = np.random.randint(0, image.shape[0], size=num_dots)

    # Add the black dots to the image
    out[ys, xs, :] = 0

    return out


# Define the path to your classes
classes_path = "/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/train"  # replace with the correct path

# Iterate over each class folder
for class_folder in os.listdir(classes_path):
    class_path = os.path.join(classes_path, class_folder)

    # Make sure we only deal with folders
    if os.path.isdir(class_path):

        # Iterate over each image in the class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Make sure we only deal with files
            if os.path.isfile(image_path):
                # Read image using OpenCV
                img = cv2.imread(image_path)

                # Add black dots
                img_noisy_dots = add_black_dots(img, num_dots=100)

                # Save the noisy image with a new name
                base_filename, file_extension = os.path.splitext(image_name)
                new_filename_dots = f"{base_filename}_dots{file_extension}"
                new_image_path_dots = os.path.join(class_path, new_filename_dots)
                cv2.imwrite(new_image_path_dots, img_noisy_dots)

print("Black dots added to all images.")
