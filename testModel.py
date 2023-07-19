from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("models/cnn_model/model.tf")

# Open the image file
img = Image.open('/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/SVG_DATA/Images to test/77.png').convert('L')  # add convert('L') to make sure it's grayscale

# Preprocess the image (resize, normalize, etc.)
img = img.resize((64, 64))  # resize to 64x64
img = np.array(img) / 255.0  # normalize

# Print the shape of img to debug
print(img.shape)

img = img.reshape(1, 64, 64, 1)  # reshape to have batch size 1

# Use the model to predict the class
probs = model.predict(img)
prediction = np.argmax(probs)

print(f"The predicted class is: {prediction}")
