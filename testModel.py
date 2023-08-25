import cv2
from keras.models import load_model
import numpy as np

class_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd'
}


def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def main():
    # Load the trained model
    model = load_model("models/cnn_model/model.tf")

    # Load the image
    image = cv2.imread("testImage.png")

    # Check if the image is blurry
    if is_blurry(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

    # Denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate characters
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image
    thresh = cv2.bitwise_not(thresh)

    # Dilate the image to help with contour detection
    kernel_size = (4, 4)  # Default kernel size
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        avg_width = sum([cv2.boundingRect(contour)[2] for contour in contours]) / len(contours)
        avg_height = sum([cv2.boundingRect(contour)[3] for contour in contours]) / len(contours)
        # play around with these values in order to get a better and more accurate contouring
        if avg_width > 50 or avg_height > 50:
            print(5, 2)
            kernel_size = (5, 2)
        elif avg_width > 40 or avg_height > 40:
            print(4, 3)
            kernel_size = (4, 3)
        elif avg_width > 25 or avg_height > 25:
            print(3, 2)
            kernel_size = (3, 2)
        elif avg_width > 15 or avg_height > 15:
            print(3, 4)
            kernel_size = (3, 4)
        else:
            kernel_size = (4, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours in the image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter and sort contours based on size
    MIN_CONTOUR_WIDTH = 15
    MIN_CONTOUR_HEIGHT = 30
    coord = [cv2.boundingRect(contour) for contour in contours if
             MIN_CONTOUR_WIDTH < cv2.boundingRect(contour)[2] < 300 and MIN_CONTOUR_HEIGHT < cv2.boundingRect(contour)[
                 3] < 300]
    coord.sort(key=lambda tup: tup[0])  # sort by x coordinate

    bank_account_number = ''

    # Predict each character from the contours
    for _, cor in enumerate(coord):
        x, y, w, h = cor
        char_image = gray[y:y + h, x:x + w]

        # Preprocess the character image for prediction
        char_image = cv2.resize(char_image, (64, 64))
        char_image = (char_image / 255.0).reshape(1, 64, 64, 1)
        contour_img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imwrite("contours.png", contour_img)

        # Predict the character using the trained model
        probs = model.predict(char_image, verbose=0)
        prediction = np.argmax(probs)
        predicted_char = class_map.get(prediction, '')

        bank_account_number += str(predicted_char)

    print(bank_account_number)


if __name__ == "__main__":
    main()
