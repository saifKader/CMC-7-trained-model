import cv2
import os
import numpy as np


# Define the function to add random dots (salt noise) to an image
def add_salt_noise(image, amount=0.004):
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 255
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

                # Add salt noise
                img_noisy_salt = add_salt_noise(img)

                # Save the noisy image with a new name
                base_filename, file_extension = os.path.splitext(image_name)
                new_filename_salt = f"{base_filename}_salt{file_extension}"
                new_image_path_salt = os.path.join(class_path, new_filename_salt)
                cv2.imwrite(new_image_path_salt, img_noisy_salt)

print("Salt noise added to all images.")
