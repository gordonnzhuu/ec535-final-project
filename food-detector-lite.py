import numpy as np
import pandas as pd
import cv2
from ai_edge_litert.interpreter import Interpreter

# Initialize the TFLite interpreter
interp = Interpreter(model_path='/home/pi/.cache/kagglehub/models/google/aiy/tfLite/vision-classifier-food-v1/1/1.tflite')
interp.allocate_tensors()
labelmap = "aiy_food_V1_labelmap.csv"

# Load class labels
classes = list(pd.read_csv(labelmap)["name"])

# Get input and output details
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
h, w = inp_det['shape'][1], inp_det['shape'][2]

# Get image path from user
image_path = input("Enter path: ")

# Load and preprocess the image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot open image at {image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (w, h))
data = np.expand_dims(img_resized.astype(inp_det['dtype']), axis=0)

# Run inference
interp.set_tensor(inp_det['index'], data)
interp.invoke()

# Get raw scores
scores = interp.get_tensor(out_det['index'])[0]

# Dequantize if output is uint8
if out_det['dtype'] == np.uint8:
    scale, zero_point = out_det['quantization']
    scores = scale * (scores.astype(np.float32) - zero_point)

# Get prediction
predicted_index = np.argmax(scores)
confidence = scores[predicted_index]

threshold = 0.5

if confidence >= threshold:
    print(f"Predicted: {classes[predicted_index]}")
    print(f"Confidence: {confidence:.10f}")
else:
    print("Predicted: Not food")
    print(f"Confidence: {1-confidence:.10f}")
