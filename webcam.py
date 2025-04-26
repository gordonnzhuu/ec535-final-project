import cv2
from ultralytics import YOLO
import time

def main():
    # 1. Use the Nano model (best for RPi)
    model = YOLO('yolov8n.pt')  # yolov8n is the fastest
    
    # 2. Open webcam with optimized settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Lower resolution for better FPS (Pi 5 can handle 720p, but 480p is smoother)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 3. Warm up the model
    print("Warming up the model...")
    _ = model.predict(cv2.resize(cap.read()[1], (320, 320)), verbose=False)
    
    prev_time = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4. Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 5. Run inference with optimizations for Pi 5:
        results = model(
            frame,
            imgsz=320,  # Smaller = faster (but less accurate)
            half=False,  # FP16 not well-supported on Pi 5
            verbose=False,
            stream=False,
            conf=0.5,   # Filter out weak detections
            iou=0.45    # Non-Max Suppression threshold
        )
        
        # 6. Display results
        if results:
            annotated = results[0].plot()
            cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 on Raspberry Pi 5', annotated)
        
        # 7. Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()