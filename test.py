#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import cv2
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite import Interpreter as tflite

def load_labels(csv_path):
    """Load label names from a CSV with a 'name' column."""
    return list(pd.read_csv(csv_path)["name"])

def preprocess_image(image_path, target_size):
    """Read an image, convert to RGB, resize and normalize to [0,1], add batch dim."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Couldn’t load image at {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_CUBIC)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_food_model.py <path_to_test_image>")
        sys.exit(1)
    test_image = sys.argv[1]

    # —— CONFIGURATION —— 
    MODEL_PATH    = "/home/pi/aiy_food_v1.tflite"
    LABEL_CSV     = "/home/pi/aiy_food_V1_labelmap.csv"
    INPUT_SHAPE   = (224, 224)    # model expects 224×224 RGB
    TOP_K         = 1             # how many top predictions to show
    # ——————————

    # Load labels
    labels = load_labels(LABEL_CSV)

    # Set up interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    input_data = preprocess_image(test_image, INPUT_SHAPE)

    # Feed the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape [num_classes]

    # Decode top-k
    top_k_idxs = np.argsort(output_data)[-TOP_K:][::-1]
    for idx in top_k_idxs:
        print(f"{labels[idx]}: {output_data[idx]:.4f}")

if __name__ == "__main__":
    main()
