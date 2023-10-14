import os
import uuid
import cv2
import numpy as np
from keras.models import load_model

from ROI_cmc7 import cmc7_crop

# your cheque image
img_path = "test.jpg"

class_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd'
}

output_folder = "testimages"
os.makedirs(output_folder, exist_ok=True)
blur = False


def unique_img_path():
    # Specify the folder path where images will be saved
    image_folder = '/Users/abdelkaderseifeddine/Documents/GitHub/NSFSHEILD-Backend/app/images'

    # Create the folder if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)

    # Generate a unique filename using UUID
    filename = f"crop-cmc7-{uuid.uuid4()}.png"
    return os.path.join(image_folder, filename)


def remove_small_noise(image, min_contour_area=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the regions to keep
    mask = np.zeros_like(gray)

    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            # Draw the contour on the mask to keep this region
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Bitwise-AND the mask with the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def is_blurry(image, threshold=100):
    global blur
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Determine if the image is blurry based on the threshold
    if laplacian_var < threshold:
        blur = True
        return True
    else:
        blur = False
        return False


def resize_to_hd(image):
    # Define the target resolution
    target_width = 1920
    target_height = 1080

    # Get the original image dimensions
    original_width, original_height = image.shape[1], image.shape[0]

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining the aspect ratio
    if aspect_ratio >= (target_width / target_height):
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))
    # Check if the image is blurry
    if is_blurry(resized_image):
        print('image is blur')
        # Define the sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Apply the sharpening kernel
        sharpened = cv2.filter2D(resized_image, -1, kernel)

        # Adjust brightness and contrast
        alpha = 1.5  # Contrast control (1.0 means no change)
        beta = 25  # Brightness control (0 means no change)
        adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

        # Apply gamma correction
        gamma = 1.8
        gamma_corrected = np.power(adjusted / 255.0, gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        return gamma_corrected
    else:
        return resized_image


def main(img_path):
    # Load the trained model
    model = load_model("models/cnn_model/model.tf")
    # Load the image
    # image = cv2.imread(img_path)
    # image = pytesseract.image_to_string(image)
    # print(image)
    image = cmc7_crop(img_path)
    image = resize_to_hd(image)
    cv2.imwrite(os.path.join(output_folder, "resized.png"), image)
    # Check if the image is blurry
    """
    if is_blurry(image):
        # Enhance image quality and sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
    else:
        print('Image is not blurry.')
    """
    # Denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate characters
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image
    thresh = cv2.bitwise_not(thresh)

    # Save thresholded image for debugging
    # cv2.imwrite("thresholded.png", thresh)

    # Dilate the image to help with contour detection
    kernel_size = (4, 4)  # Default kernel size
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Calculate the average width and height of detected contours
        avg_width = sum([cv2.boundingRect(contour)[2] for contour in contours]) / len(contours)
        avg_height = sum([cv2.boundingRect(contour)[3] for contour in contours]) / len(contours)

        # Define a scaling factor to adjust the kernel size based on contour sizes
        scaling_factor = 0.1  # You can adjust this factor as needed

        # Calculate the dynamic kernel size based on the average contour size and scaling factor
        if not blur:
            kernel_width = max(int(avg_width * scaling_factor), 3)
            kernel_height = max(int(avg_height * scaling_factor), 1)
        else:
            kernel_width = max(int(avg_width * scaling_factor), 3)
            kernel_height = max(int(avg_height * scaling_factor), 4)
        # Create the structuring element (kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))

        # Apply dilation with dynamic kernel and iterations
        dilated = cv2.dilate(thresh, kernel, iterations=3)
    else:
        print('no contour found')
        # Handle the case when no contours are detected
        dilated = thresh  # You may want to adjust this behavior

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
        cv2.imwrite(os.path.join(output_folder, "contours.png"), contour_img)
        # Predict the character using the trained model
        probs = model.predict(char_image, verbose=0)  # remove verbose to see the process
        prediction = np.argmax(probs)
        predicted_char = class_map.get(prediction, '')

        bank_account_number += str(predicted_char)

    print(bank_account_number)  # Debug print to check the output
    return bank_account_number  # Return the extracted bank account number


if __name__ == "__main__":
    main(img_path)
