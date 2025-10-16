import cv2
from ultralytics import YOLO

import pathlib
root_dir = str(pathlib.Path(__file__).parent.parent)

model = YOLO(root_dir+"/runs/train/face_detection_run/weights/best.pt")

# Open laptop camera (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    # Loop through detections and draw boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])               # Confidence
            cls = int(box.cls[0])                   # Class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLOv11-F Live Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
