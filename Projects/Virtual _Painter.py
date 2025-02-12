import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3, 140)  # Set width
cap.set(4, 140)  # Set height
cap.set(10, 150)
myColors = [[0, 179, 71, 20, 255, 255],
            [30, 110, 131, 46, 255, 187],
            [50, 116, 0, 78, 255, 255]]
myColorValues = [[13, 145, 248],
                 [0, 255, 243],
                 [0, 255, 0]]
myPoints = []  # List to store drawn points

def findColor(img, myColors, myColorValues):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for i, color in enumerate(myColors):
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHsv, lower, upper)
        x, y = getCounters(mask)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
    return newPoints
def getCounters(img):
    counters, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in counters:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)
while True:
    ret, frame = cap.read()
    imgResult = frame.copy()
    if not ret:
        break
    newPoints = findColor(frame, myColors, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
        
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)
    cv2.imshow("Web cam", imgResult)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
        break
cap.release()
cv2.destroyAllWindows()
