import cv2 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Error: Could not open webcam.")
	exit(1)

while True:
	ret, frame = cap.read()
	if not ret:
		print("Error: Failed to capture image.")
		break

	cv2.imshow("Webcam Test", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
