import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import cv2  # type: ignore
from skimage import io  # type: ignore
from skimage.io import imread  # type: ignore

# Load the TensorFlow Lite model
model_path = "/home/pi/.cache/kagglehub/models/google/aiy/tfLite/vision-classifier-food-v1/1/1.tflite"  # Update with the correct path to your .tflite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
labelmap = "aiy_food_V1_labelmap.csv"
classes = list(pd.read_csv(labelmap)["name"])

# Parameters
input_shape = (192, 192)
confidence_threshold = 0.4

# Load image from file path
image_jpg = input("Enter the path to the jpg file: ")
image = np.asarray(io.imread(image_jpg))

# Resize the image
image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)

# Scale values to [0,1]
image = image / image.max()

# Model expects an input of (1, 224, 224, 3)
images = np.expand_dims(image.astype(np.uint8), axis=0)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], images)

# Run inference
interpreter.invoke()

# Get the output tensor
probabilities = interpreter.get_tensor(output_details[0]['index'])[0]

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
