import os
import numpy as np
from keras.src.utils import load_img, img_to_array, to_categorical
from sklearn.preprocessing import LabelEncoder



def preprocess_data(directory):
    data = []
    labels = []

    # Load images and labels
    for class_folder in os.listdir(directory):
        class_folder_path = os.path.join(directory, class_folder)

        if os.path.isdir(class_folder_path):
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)

                # Ensure the file is an image
                if not image_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    # Load the image and convert to array
                    print(f"Loading image at path: {image_path}")
                    image = load_img(image_path, color_mode='grayscale', target_size=(64,64))
                    image = img_to_array(image)
                    image = image.reshape((64, 64, 1))

                    # Normalize the image
                    image = image / 255.0

                    # Add the image and label to their respective lists
                    data.append(image)
                    labels.append(class_folder)
                except Exception as e:
                    print(f"Error loading image: {image_path}. Error: {str(e)}")

    # Convert data and labels to numpy arrays
    data = np.array(data, dtype="float32").reshape((-1, 64, 64, 1))
    labels = np.array(labels)

    # Encode the labels
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(integer_encoded)

    return data, labels
