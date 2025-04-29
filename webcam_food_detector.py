import tensorflow_hub as hub  # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import cv2  # type: ignore

# Load the TensorFlow Hub layer
model = hub.KerasLayer(
    '/Users/gordo/.cache/kagglehub/models/google/aiy/tensorFlow1/vision-classifier-food-v1/1',
    trainable=False
)

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
    exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert BGR (OpenCV default) to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    resized = cv2.resize(rgb, dsize=input_shape, interpolation=cv2.INTER_CUBIC)

    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension: (1, 224, 224, 3)
    batch = np.expand_dims(normalized, 0)

    # Run the model
    output = model(batch)               # Returns a Tensor
    probabilities = output.numpy()[0]   # Convert to NumPy, remove batch dim

    # Find best prediction
    pred_idx = np.argmax(probabilities)
    confidence = probabilities[pred_idx]

    # Prepare display text
    if confidence >= confidence_threshold:
        text = f"{classes[pred_idx]} ({confidence:.2f})"
    else:
        text = f"Not food ({confidence:.2f})"

    # Overlay text on the original BGR frame
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0), 2, cv2.LINE_AA
    )

    # Show result
    cv2.imshow("Food Detector", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
