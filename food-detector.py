import tensorflow_hub as hub

model = hub.KerasLayer('/home/pi/.cache/kagglehub/models/google/aiy/tensorFlow1/vision-classifier-food-v1/1')

import numpy as np
import pandas as pd
import cv2
from skimage import io
from skimage.io import imread

image_jpg = input("Enter the path to the jpg file: ")
labelmap = "aiy_food_V1_labelmap.csv"
input_shape = (224, 224)
confidence_threshold = 0.5
classes = list(pd.read_csv(labelmap)["name"])

# Load image from file path
image = np.asarray(io.imread(image_jpg))
# Resize the image
image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
# Scale values to [0,1]
image = image / image.max()
# Model expects an input of (?,224,224,3)
images = np.expand_dims(image, 0)

# Use model to predict food
output = model(images)
# Predicted probabilities for each class
probabilities = output.numpy()[0]
# Predicted class index
predicted_index = np.argmax(probabilities)
# Predicted confidence level
confidence_level = probabilities[predicted_index]

# Print prediction
if confidence_level >= confidence_threshold:
    print("Prediction: ", classes[predicted_index])
    print("Confidence Level: ", confidence_level)
else:
    print("Not food")
    print("Confidence Level: ", confidence_level)