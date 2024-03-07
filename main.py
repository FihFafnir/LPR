from util import *

plate_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    plates = plate_classifier.detectMultiScale(b_filter, 1.2, 4)
    
    for (x, y, w, h) in plates:
        read_plate(frame[y:y+h, x:x+w])
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Camera:", frame)

cap.release()
cv2.destroyAllWindows()
