import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()

ret, frame = cap.read()
if not ret:
	print("Falied to grab frame")
else:
	cv2.imwrite('frame.jpg', frame)
	print("Captured!")

cap.release()
