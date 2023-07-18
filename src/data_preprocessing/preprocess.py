import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_data(directory):
    data = []
    labels = []

    # Load images and labels
    for class_folder in os.listdir(directory):
        class_folder_path = os.path.join(directory, class_folder)

        if os.path.isdir(class_folder_path):
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)

                # Load the image and convert to array
                image = load_img(image_path, color_mode='grayscale')
                image = img_to_array(image)

                # Normalize the image
                image = image / 255.0

                # Add the image and label to their respective lists
                data.append(image)
                labels.append(class_folder)

    # Convert data and labels to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # Encode the labels
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(integer_encoded)

    return data, labels
