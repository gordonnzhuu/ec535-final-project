import cv2

# Try different indices (0, 1, 2, etc.) to find your external webcam
for i in range(0, 3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
    else:
        print(f"No camera found at index {i}")