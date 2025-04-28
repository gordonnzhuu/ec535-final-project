#!/usr/bin/env python3
"""
food_nutrition_detector.py

Capture frames from your webcam, classify the main food item using MobileNetV2,
and fetch nutrition facts from the Nutritionix API. Overlays the results on the video feed.

Requirements:
  • Python 3.7–3.10
  • pip install tensorflow==2.11.0 opencv-python requests
  • A webcam (index 0)
  • Nutritionix APP_ID and APP_KEY

Usage:
  1. Replace APP_ID and APP_KEY with your actual Nutritionix credentials.
  2. python food_nutrition_detector.py
  3. Press 'q' to quit.
"""

import cv2
import numpy as np
import tensorflow as tf
import requests

# ------------------------------
# CONFIGURATION
# ------------------------------

# Replace these with your own Nutritionix credentials:
APP_ID  = 'fdee5b24'
APP_KEY = '72a92983f8200a50bf667a30653c0e24'

API_URL = 'https://trackapi.nutritionix.com/v2/natural/nutrients'

# ------------------------------
# MODEL LOADING
# ------------------------------

print("Loading MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
preprocess     = tf.keras.applications.mobilenet_v2.preprocess_input
decode_preds   = tf.keras.applications.mobilenet_v2.decode_predictions

# ------------------------------
# WEBCAM SETUP
# ------------------------------

print("Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

# ------------------------------
# MAIN LOOP
# ------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize & preprocess for MobileNetV2
    img = cv2.resize(frame, (224, 224))
    x   = img.astype(np.float32)
    x   = preprocess(x)
    x   = np.expand_dims(x, axis=0)

    # Run inference
    preds = model.predict(x)
    # decode_predictions returns list of lists of tuples: (class_id, class_name, score)
    food_label = decode_preds(preds, top=1)[0][0][1]  # e.g. 'banana'

    # Query Nutritionix API
    payload = {'query': f'1 serving {food_label}'}
    headers = {
        'x-app-id':  APP_ID,
        'x-app-key': APP_KEY,
        'Content-Type': 'application/json'
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        data = resp.json()
        if resp.status_code == 200 and 'foods' in data:
            f = data['foods'][0]
            calories = f.get('nf_calories',     'N/A')
            protein  = f.get('nf_protein',      'N/A')
            fat      = f.get('nf_total_fat',    'N/A')
            carbs    = f.get('nf_total_carbohydrate', 'N/A')
            nutrition_text = (
                f"{calories:.0f} kcal | "
                f"P:{protein:.1f}g F:{fat:.1f}g C:{carbs:.1f}g"
            )
        else:
            nutrition_text = "Nutrition data unavailable"
    except Exception as e:
        nutrition_text = f"API error: {e}"

    # Overlay text on the frame
    cv2.putText(frame, f"Food: {food_label}",    (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, nutrition_text,            (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Food Nutrition Detector", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
