import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

class_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd'
}

# Load the trained model
model = load_model("models/cnn_model/model.tf")

# Load the image
image = cv2.imread("testImage.png")
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate characters
res, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Invert the image
thresh = cv2.bitwise_not(thresh)

# Calculate contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('hi')
if contours:
    print('hi2')
    # Adjust kernel size based on average width and height of contours
    avg_width = sum([cv2.boundingRect(contour)[2] for contour in contours]) / len(contours)
    avg_height = sum([cv2.boundingRect(contour)[3] for contour in contours]) / len(contours)

    # play around with these value to get a better contouring
    if avg_width > 50 or avg_height > 50:
        print('5,5')
        kernel_size = (5, 5)
    elif avg_width > 40 or avg_height > 40:
        print('4,4')
        kernel_size = (4, 4)
    elif avg_width > 25 or avg_height > 25:
        print('2,2')
        kernel_size = (2, 2)
    elif avg_width > 15 or avg_height > 15:
        print('2,5')
        kernel_size = (2, 5)
    else:
        print('3,3')
        kernel_size = (3, 3)

    # Dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dilated = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours again
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on size and sort
    coord = [cv2.boundingRect(contour) for contour in contours if
             40 < cv2.boundingRect(contour)[2] < 300 and 40 < cv2.boundingRect(contour)[3] < 300]
    coord.sort(key=lambda tup: tup[0])  # sort by x coordinate

    # Predict each character
    for count, cor in enumerate(coord):
        x, y, w, h = cor
        char_image = gray[y:y + h, x:x + w]

        # Preprocess the character image and use the model to predict
        char_image = cv2.resize(char_image, (64, 64))
        char_image = np.array(char_image) / 255.0  # normalize
        char_image = char_image.reshape(1, 64, 64, 1)
        contour_img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imwrite("contours.png", contour_img)

        # Predict the character
        probs = model.predict(char_image, verbose=0)
        prediction = np.argmax(probs)
        predicted_char = class_map.get(prediction, '')

        print(f"The predicted class for character {count} is: {predicted_char}")
else:
    print("No contours detected.")
