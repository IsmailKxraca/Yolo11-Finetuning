from ultralytics import YOLO
import cv2

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break

    results = model(frame, stream=True) 
    
    for result in results:
        # Visualisierung
        annotated_frame = result.plot()
    
    cv2.imshow('YOLO Webcam', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()