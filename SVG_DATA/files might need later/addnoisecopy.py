import cv2
import os
import numpy as np


# Define the function to add noise to an image
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


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

                # Apply noise
                img_noisy = noisy('s&p', img)  # change 's&p' to the desired noise type

                # Create a new filename for the noisy image
                base_filename, file_extension = os.path.splitext(image_name)
                new_filename = f"{base_filename}_noisy{file_extension}"
                new_image_path = os.path.join(class_path, new_filename)

                # Save the noisy image
                cv2.imwrite(new_image_path, img_noisy)

print("Noise added to all images.")
