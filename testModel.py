import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("models/cnn_model/model.tf")

# Load the image
image = cv2.imread("/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/SVG_DATA/Images_to_test/3.png")
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate characters
res,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

# Invert the image
thresh = cv2.bitwise_not(thresh)

# Dilate the image
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh, kernel, iterations=5)

# Find contours
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Filter contours based on size and sort
coord = [cv2.boundingRect(contour) for contour in contours if 40 < cv2.boundingRect(contour)[2] < 300 and 40 < cv2.boundingRect(contour)[3] < 300]
coord.sort(key=lambda tup: tup[0])  # sort by x coordinate

# Predict each character
for count, cor in enumerate(coord):
    x, y, w, h = cor
    char_image = gray[y:y+h, x:x+w]

    # Preprocess the character image and use the model to predict
    char_image = cv2.resize(char_image, (64, 64))
    char_image = np.array(char_image) / 255.0  # normalize
    char_image = char_image.reshape(1, 64, 64, 1)

    # Predict the character
    probs = model.predict(char_image)
    prediction = np.argmax(probs)
    print(f"The predicted class for character {count} is: {prediction}")
