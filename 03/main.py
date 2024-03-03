import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
cap = cv2.VideoCapture(0)
plate_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

def show_img(img, name = "Imagem"):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 700, 300)
    cv2.imshow(name, img)
    close_img = cv2.waitKey(0) == ord("q") 
    while not close_img:
        close_img = cv2.waitKey(0) == ord("q") 
    cv2.destroyAllWindows()

def handle_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b_filter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(b_filter, 30, 200)
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5) 
    return (gray, b_filter, edged, thresholded)

def get_contours(img):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def get_roi(contours):
    roi = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, perimeter * 0.02, True)
        if len(approx) == 4:
            roi = approx
            break
    return np.array([roi], np.int32)

def crop_image(img, roi):
    points = roi.reshape(4, 2)
    x, y = np.split(points, [-1], axis=1)
    (x1, x2) = (np.min(x), np.max(x))
    (y1, y2) = (np.min(y), np.max(y))
    
    return img[y1:y2, x1:x2]

def read_plate(img):
    gray, b_filter, edged, thresholded = handle_img(img)
    
    #contours = get_contours(edged) 
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    #show_img(img)
    #show_img(gray)
    #show_img(b_filter)
    #
    #roi = get_roi(contours)
    #number_plate = crop_image(b_filter, roi)
    
    number_plate = b_filter
           
    text = pytesseract.image_to_string(number_plate, "por", config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(text) >= 7:
        print(text)
    #show_img(number_plate)


while True:
    ret, frame = cap.read()
    gray, b_filter, edged, thresholded = handle_img(frame)
    plates = plate_classifier.detectMultiScale(b_filter, 1.2, 4)
    
    for (x, y, w, h) in plates:
        read_plate(frame[y:y+h, x:x+w])
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Camera:", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
