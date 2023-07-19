import Augmentor

# Define your pipeline
p = Augmentor.Pipeline("/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/SVG_DATA/photos")

# Add operations to the pipeline
# Random rotation between -25 and +25 degrees
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)

# Randomly flip some of the images left/right
p.flip_left_right(probability=0.5)

# Randomly flip some of the images top/bottom
p.flip_top_bottom(probability=0.5)

# Randomly crop images
p.crop_random(probability=0.5, percentage_area=0.8)

# Randomly zoom into images
p.zoom_random(probability=0.5, percentage_area=0.8)

# Randomly distort the images
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

# Randomly skew the images
p.skew(probability=0.5, magnitude=0.6)

# Change the image contrast, saturation, and brightness
p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)
p.random_color(probability=0.5, min_factor=0.8, max_factor=1.2)
p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)

# Convert the images to grayscale
p.black_and_white(probability=0.2)

# Resize all images
p.resize(probability=1, width=64, height=64)

# Execute the pipeline
p.sample(1000)  # Generate 1000 augmented images
