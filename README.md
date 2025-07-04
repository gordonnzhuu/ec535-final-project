# ec535-final-project

### Project: Real-time Food Classifier 

### Goal: Used a Raspberry Pi 5 and webcam to detect and classify foods using a lightweight neural network Tensorflow Lite

### Hardware: Raspberry Pi 5, Logitech C270 webcam

### Software: Raspberry Pi OS, Tensorflow Lite, OpenCV, Python

### Model Architecture: MobileNet V1.0
### Model: Tensorflow Lite

# how to use:

1. create a virtual environment
```
python3 -m venv venv 
```
2. enter virtual environment 
```
source venv/bin/activate
```
3. download dependencies
```
pip install numpy pandas tflite-runtime opencv-python ai_edge_litert kagglehub
```
4. run webcam_food_detector_lite.py
```
python3 webcam_food_detector_lite.py
```
