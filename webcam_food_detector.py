import tensorflow_hub as hub # type: ignore
import numpy as np
import pandas as pd
import cv2 # type: ignore

# Load the TensorFlow model
model = hub.KerasLayer('/Users/gordo/.cache/kagglehub/models/google/aiy/tensorFlow1/vision-classifier-food-v1/1')

# Load label map
labelmap = "aiy_food_V1_labelmap.csv"
classes = list(pd.read_csv(labelmap)["name"])

# Parameters
input_shape = (224, 224)
confidence_threshold = 0.5

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
    # Scale values to [0,1]
    normalized_frame = resized_frame / 255.0
    # Model expects an input of (?,224,224,3)
    images = np.expand_dims(normalized_frame, 0)

    # Use model to predict food
    output = model(images)
    probabilities = output.numpy()[0]
    predicted_index = np.argmax(probabilities)
    confidence_level = probabilities[predicted_index]

    # Display prediction
    if confidence_level >= confidence_threshold:
        prediction_text = f"Prediction: {classes[predicted_index]} ({confidence_level:.2f})"
    else:
        prediction_text = f"Not food ({confidence_level:.2f})"

    # Show the frame with prediction
    cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Food Detector", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()