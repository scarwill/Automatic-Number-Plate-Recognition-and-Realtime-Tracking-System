import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use the appropriate YOLOv8 model (e.g., yolov8n.pt)

def detect_number_plate(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if confidence > 0.5 and class_id == 2:  # Assuming class_id 2 is for number plates
                return (x1, y1, x2, y2)
    return None

def recognize_characters(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    details = pytesseract.image_to_string(thresh_plate, config='--psm 8')
    return details.strip()

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detected_plate = detect_number_plate(frame)
        if detected_plate:
            x1, y1, x2, y2 = detected_plate

            # Ensure coordinates are within the image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x1 < x2 and y1 < y2:
                license_crop = frame[y1:y2, x1:x2]
                plate_text = recognize_characters(license_crop)
                
                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Number Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
