# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
import os


def cmc7_crop(img_path):
    output_folder = "testimages"
    os.makedirs(output_folder, exist_ok=True)

    # load the input image, convert it to grayscale, and grab its
    # dimensions
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    # smooth the image using a 3x3 Gaussian blur and then apply a
    # blackhat morpholigical operator to find dark regions on a light
    # background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    cv2.imshow("Blackhat", blackhat)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Rect Close", thresh)
    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MICR, then perform a
    # series of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    cv2.imshow("Square Close", thresh)

    # Save images to the "testimages" folder
    cv2.imwrite(os.path.join(output_folder, "blackhat.png"), blackhat)
    cv2.imwrite(os.path.join(output_folder, "gradient.png"), grad)
    cv2.imwrite(os.path.join(output_folder, "square_close.png"), thresh)

    # find contours in the thresholded image and sort them from bottom
    # to top (since the MICR will always be at the bottom of the passport)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="bottom-to-top")[0]
    # initialize the bounding box associated with the MICR
    micrBox = None

    for c in cnts:
        # compute the bounding box of the contour and then derive the
        # how much of the image the bounding box occupies in terms of
        # both width and height
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
        print('width:', percentWidth)
        print('height:', percentHeight)
        # if the bounding box occupies > 80% width and > 4% height of the
        # image, then assume we have found the MICR
        if percentWidth > 0.56 and percentHeight > 0.04:
            micrBox = (x, y, w, h)
            break

    # if the MICR was not found, exit the script
    if micrBox is None:
        print("[INFO] MICR could not be found")
        sys.exit(0)
    # pad the bounding box since we applied erosions and now need to
    # re-grow it
    (x, y, w, h) = micrBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    # extract the padded MICR from the image
    micr = image[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(output_folder, "cmc7.png"), micr)
    return micr
