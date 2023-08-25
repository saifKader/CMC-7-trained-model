import Augmentor
import cv2
import random
import Augmentor.Operations as Operations
from PIL import Image
import numpy as np


class GaussianBlur(Operations.Operation):
    def __init__(self, probability, max_sigma):
        Operations.Operation.__init__(self, probability)
        self.max_sigma = max_sigma

    def perform_operation(self, images):
        for i, image in enumerate(images):
            # Convert the PIL image to a numpy array
            image = np.array(image)

            # Perform the Gaussian blur operation
            sigma = random.uniform(0, self.max_sigma)
            image = cv2.GaussianBlur(image, (5, 5), sigma)

            # Convert the numpy array back to a PIL image
            image = Image.fromarray(image)

            images[i] = image

        return images


# Define your pipeline
p = Augmentor.Pipeline("path to your images")

# Small rotations are a common realistic scenario
p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)

# Slight skewing can account for imperfect alignment
p.skew(probability=0.3, magnitude=0.2)

# Changes in brightness can simulate different lighting conditions
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)

# Changes in contrast can simulate different scanning/photographing conditions
p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)

# Gaussian distortion can simulate slight rotations and noise
p.gaussian_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=2, corner='bell', method='in')

# This can simulate small distortions in the image that can occur during scanning or photo capture.
p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)

# Resize all images
p.resize(probability=1, width=64, height=64)

# blur images
gaussian_blur = GaussianBlur(probability=0.3, max_sigma=100.0)
p.add_operation(gaussian_blur)
# Execute the pipeline
p.sample(1000)  # Generate 1000 augmented images
