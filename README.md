# CMC-7 Check Character Recognition System

## Table of Contents

- [Overview](#overview)
- [Built With](#built-with)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Run Project](#run-project)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Data Augmentation](#data-augmentation)


## Overview

The CMC-7 Check Character Recognition System is a deep learning project aimed at recognizing CMC-7 characters on checks. The project combines data preprocessing, model training, and optical character recognition (OCR) techniques to accurately identify CMC-7 characters from check images.

## Built With

This project utilizes the following frameworks and libraries:

- [![TensorFlow][TensorFlow]][TensorFlow-url] - Deep learning framework used for creating and training the neural network model.
- [![OpenCV][OpenCV]][OpenCV-url] - Computer vision library for image preprocessing and manipulation.
- [![PIL][PIL]][PIL-url] - Python Imaging Library for handling image operations.
- [![Augmentor][Augmentor]][Augmentor-url] - Augmentation library for generating diverse training data.
- [![Keras][Keras]][Keras-url] - High-level neural networks API running on top of TensorFlow.
- [![Python][Python]][Python-url] - Programming language used for the project.

[TensorFlow]: https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
[OpenCV]: https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[PIL]: https://img.shields.io/badge/PIL-F9DC3E?style=flat-square&logo=python&logoColor=white
[PIL-url]: https://python-pillow.org/
[Augmentor]: https://img.shields.io/badge/Augmentor-02A8AC?style=flat-square&logo=python&logoColor=white
[Augmentor-url]: https://augmentor.readthedocs.io/
[Keras]: https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
[Python]: https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white
[Python-url]: https://www.python.org/

## Project Structure

The project is organized into several directories:

- `data`: Contains training, validation, and test data in specific class folders.
- `models`: Contains the trained CNN model and its architecture definition.
- `src`: Contains modules for data preprocessing, model training.
- `main.py`: The entry point for preprocessing, training, and saving the model.
- `testModel.py`: Used to test the trained model's recognition performance.
- `data_augmentor.py`: A script for augmenting custom bank-specific CMC-7 characters to enhance model recognition.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/seifKader/CMC-7-trained-model.git
   cd CMC-7-trained-model
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run Project

To run the project and test the trained model's recognition on your own images, follow these steps:

1. **Prepare Your Data:** Before running the project, ensure that you have your training, validation, and test datasets ready as described in the `data` directory. Alternatively, you can use the existing trained model if available.

2. **Open `testModel.py`:** Navigate to the project directory and open the `testModel.py` script in your preferred text editor.

3. **Specify Your Test Image:** In the `testModel.py` script, find the following line:

   ```python
   image = cv2.imread("testImage.png")
   ```

   Replace `"testImage.png"` with the path to your own test image.

4. **Run the Script:** Open your terminal or command prompt and execute the following command to run the script:

   ```bash
   python testModel.py
   ```

5. **View Results:** The script will process the image and display the predicted class for each recognized character in the console output.

**Note:** The `testModel.py` script processes the provided image and prints the predicted classes for the recognized characters. If you want to further enhance the visual feedback, consider modifying the script to draw bounding boxes around the recognized characters.



## Data Preprocessing

The `preprocess.py` module under `src/data_preprocessing` prepares the data by loading images, converting them to grayscale, normalizing, and creating labels. This processed data is then ready for training.

## Model Training

The `train.py` module under `src/model_training` defines a Convolutional Neural Network (CNN) architecture and trains it on the preprocessed data. The trained model is saved for future use.


## Data Augmentation

Enhance model recognition for custom bank-specific CMC-7 characters by following these steps:

1. **Prepare Cropped Images:** Gather cropped images of bank-specific CMC-7 characters. Each character should be in its own image file.

2. **Specify The Character Image:** In the `data_augmentor.py` script, locate the following line:
   
   ```python
   p = Augmentor.Pipeline("path to your image")
   ```

   Replace `"path to your image"` with the path to the image of the character you want to augment.

3. **Run the `data_augmentor.py` Script:** Execute the following command in your terminal or command prompt:

   ```bash
   python data_augmentor.py
   ```

**Note:** Feel free to explore and customize the project to meet your specific needs and further enhance character recognition accuracy.
