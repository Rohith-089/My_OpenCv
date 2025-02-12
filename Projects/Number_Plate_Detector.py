import cv2
import numpy as np
import os
import pytesseract
# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# Load number plate cascade
nPlateCascade = cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
# Tesseract OCR path (set once)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Output directory for scanned images
output_dir = "Resources/Scanned"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
minArea = 500
color = (255, 0, 0)
count = 1
def preprocess_image(img):
    """Preprocess image to enhance OCR accuracy"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary
def detect_text(image_path):
    """Detect text from the processed number plate image"""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return None
    img = cv2.imread(image_path)
    processed_img = preprocess_image(img)
    text = pytesseract.image_to_string(processed_img, config='--psm 6').strip()
    # Validate detected text
    if text and len(text) > 3:  # Avoid saving empty or incorrect results
        print(f"Detected Number Plate: {text}")
        with open("detected_text.txt", "a") as file:
            file.write(f"{text}\n")
        return text
    else:
        print("No valid text detected!")
        return None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    imgRoi = None
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            
            imgRoi = frame[y:y + h, x:x + w]
            img_path = os.path.join(output_dir, f"NoPlate_{count}.jpg")
            cv2.imwrite(img_path, imgRoi)
            print(f"Saved number plate image at: {img_path}")
            detected_text = detect_text(img_path)
            if detected_text:
                cv2.imshow("Processed Image", preprocess_image(imgRoi))
            count += 1  # Increment count only for valid detections
    cv2.imshow('Webcam Feed', frame)
    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()
