#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import preprocessing
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import Augmentor
from scipy import ndimage

# Specify the root directory of your dataset
root_directory = "D:\GP\emotion detection datasets\KDEF_and_AKDEF\KDEF_and_AKDEF\KDEF"


# Create a directory to save preprocessed images
output_directory = "D:\GP\emotion detection datasets\preprocessed"
os.makedirs(output_directory, exist_ok=True)
# Load the face detection cascade classifier (adjust the path accordingly)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to perform face cropping
def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = image[y:y + h, x:x + w]
        return face_roi
    else:
        # If no face is detected, return the original image
        return image

# Function to remove background using chroma keying (replace green with transparency)
def remove_background(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the image to get only green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create a mask with transparency
    mask_with_alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)

    # Replace green with transparency in the original image
    img_with_transparency = cv2.addWeighted(img, 1.0, mask_with_alpha, 0.5, 0)

    return img_with_transparency


# Iterate through each folder in the root directory
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Create an Augmentor pipeline for each folder
        pipeline = Augmentor.Pipeline(folder_path, output_directory=os.path.join(output_directory, folder_name))
        # Resize images to 224x224 and convert to grayscale
        pipeline.resize(probability=1.0, width=224, height=224)
        pipeline.greyscale(probability=1.0)
        # Define augmentation operations for each pipeline
        # Customize these operations based on your requirements
        pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        pipeline.zoom_random(probability=0.5, percentage_area=0.8)
        pipeline.flip_left_right(probability=0.5)
        pipeline.flip_top_bottom(probability=0.5)

        # Set the number of augmented images you want to generate for each folder
        num_augmented_images = 10

        # Execute the augmentation process for each pipeline
        pipeline.sample(num_augmented_images)

        # Apply face cropping and background removal to the augmented images
        for augmented_image_path in os.listdir(os.path.join(output_directory, folder_name, "output")):
            img_path = os.path.join(output_directory, folder_name, "output", augmented_image_path)
            img = cv2.imread(img_path)

            # Crop face
            cropped_face = crop_face(img)

            # Remove background
            img_with_transparency = remove_background(cropped_face)

            # Save the image with removed background
            cv2.imwrite(img_path, img_with_transparency)