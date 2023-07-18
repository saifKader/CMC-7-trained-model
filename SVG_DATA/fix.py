import tensorflow as tf
from PIL import Image, ImageOps
import os
import numpy as np


def load_and_process_image(image_path, target_size):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path)

    # Make the image square by padding the smaller dimension
    width, height = image.size
    if width != height:
        max_dim = max(width, height)
        new_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        upper_left = ((max_dim - width) // 2, (max_dim - height) // 2)
        new_image.paste(image, upper_left)
        image = new_image

    # Now resize the image to the target size
    image = image.resize(target_size)

    # Convert the PIL Image to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Invert the colors if the image is grayscale (this is optional and may depend on your specific images)
    if image.shape[2] == 1:
        image = 1 - image

    return image



def load_images_from_folder(folder_path, image_size=(28, 28)):
    data = []
    labels = []
    class_folders = os.listdir(folder_path)
    for class_folder in class_folders:
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                image = load_and_process_image(image_path, image_size)

                # Normalize the image
                image = image / 255.0

                data.append(image)
                labels.append(class_folder)
    return np.array(data), np.array(labels)


data, labels = load_images_from_folder("/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/train")
