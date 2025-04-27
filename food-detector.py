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

try:
	# Load image from file path
	image = np.asarray(imread(image_jpg), dtype="float")
	# Resize the image
	image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
	# Scale values to [0,1]
	image = image / image.max()
	# Model expects an input of (?,224,224,3)
	images = np.expand_dims(image,0)

	# Use model to predict food
	output = model(images)
	# Find the index
	predicted_index = output.numpy().argmax()
	# Read labelmap
	classes = list(pd.read_csv(labelmap)["name"])
	# Print prediction
	print("Prediction : ", classes[predicted_index])

except FileNotFoundError:
	print(f"Error: File not found at path: {image_jpg}")
except Exception as e:
	print(f"Error: Unexpected: {e}")
