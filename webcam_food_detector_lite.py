import numpy as np
import pandas as pd
import cv2
from ai_edge_litert.interpreter import Interpreter

# Load model and labels
interp = Interpreter(model_path='/home/pi/.cache/kagglehub/models/google/aiy/tfLite/vision-classifier-food-v1/1/1.tflite')
interp.allocate_tensors()
labelmap = "aiy_food_V1_labelmap.csv"
classes = list(pd.read_csv(labelmap)["name"])

# Get input and output details
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
h, w = inp_det['shape'][1], inp_det['shape'][2]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w, h))
    data = np.expand_dims(img_resized.astype(inp_det['dtype']), axis=0)

    # Run inference
    interp.set_tensor(inp_det['index'], data)
    interp.invoke()
    scores = interp.get_tensor(out_det['index'])[0]

    # Dequantize if needed
    if out_det['dtype'] == np.uint8:
        scale, zero_point = out_det['quantization']
        scores = scale * (scores.astype(np.float32) - zero_point)

    # Get prediction
    predicted_index = np.argmax(scores)
    confidence = scores[predicted_index]
    threshold = 0.5

    if confidence >= threshold:
        label = f"{classes[predicted_index]} ({confidence:.2f})"
    else:
        label = f"Not food ({1 - confidence:.2f})"

    # Display label on frame
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if confidence >= threshold else (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Food Classifier", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
