import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
plate_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
cap = cv2.VideoCapture(0)


def is_plate_format(text):
    if len(text) != 7:
        return False

    return text[:3].isalpha() and text[3].isdigit() and text[-2:].isdigit()

def read_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    thresh = cv2.threshold(b_filter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)

   # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
   # invert = 255 - opening
    
    cv2.imshow("Plate", invert)
    text = pytesseract.image_to_string(invert, lang="por", config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()
    
    if is_plate_format(text):
        print("Plate:", text) 

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    plates = plate_classifier.detectMultiScale(b_filter, 1.2, 4)
    
    for (x, y, w, h) in plates:
        read_plate(frame[y:y+h, x:x+w])
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Camera:", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
