import cv2
import numpy as np
import pytesseract
import os
import time
# Load the image
frame = cv2.imread('Resources/NP.png')
# Load the number plate cascade
nPlateCascade = cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
minArea = 500
color = (255, 0, 120)
count = 0
# Convert to grayscale
imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
imgRoi = None
for (x, y, w, h) in numberPlates:
    area = w * h
    if area > minArea:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Number Plate", (x, y - 5), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        imgRoi = frame[y:y + h, x:x + w]
        cv2.imshow("ROI", imgRoi)
cv2.imshow("Web Cam", frame)
output_dir = "Resources/Scanned"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if imgRoi is not None:
    img_path = os.path.join(output_dir, f"NoPlate_{count}.jpg")
    cv2.imwrite(img_path, imgRoi)
    print(f"Saved number plate image at: {img_path}")
    count += 1
else:
    print("No number plate detected!")
cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
cv2.putText(frame, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
cv2.waitKey(1000)
cv2.destroyAllWindows()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(binary, -1, kernel)
    return sharpened
def detect_text(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return
    img = cv2.imread(image_path)
    processed_img = preprocess_image(img)
    text = pytesseract.image_to_string(processed_img, config='--psm 6').strip()
    if text:
        print("\nDetected Text:\n", text)
        text_file = os.path.join("detected_text.txt")
        with open(text_file, "a") as file:
            file.write(f"\n{text}")
        print(f"Detected text saved in: {text_file}")
    else:
        print("No text detected!")
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
image_path = "Resources/Scanned/NoPlate_0.jpg"
detect_text(image_path)