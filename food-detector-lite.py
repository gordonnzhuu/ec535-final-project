import numpy as np
import pandas as pd
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# Initialize
interp = Interpreter(model_path='/home/pi/.cache/kagglehub/models/google/aiy/tfLite/vision-classifier-food-v1/1/1.tflite')
interp.allocate_tensors()
labelmap = "aiy_food_V1_labelmap.csv"

# Read aiy_food_V1_labelmap
classes = list(pd.read_csv(labelmap)["name"])

# Prepare input
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
h, w = inp_det['shape'][1], inp_det['shape'][2]

image_path = input("Enter path: ")

img = Image.open(image_path).convert('RGB').resize((w, h))
data = np.expand_dims(np.array(img, dtype=inp_det['dtype']), 0)

# Inference
interp.set_tensor(inp_det['index'], data)
interp.invoke()

scores = interp.get_tensor(out_det['index'])[0]
predicted_index = np.argmax(scores)

# Output
print("Predicted: ", classes[predicted_index])
