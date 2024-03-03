import pytesseract
import cv2
import imutils
import easyocr
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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
    return (gray, b_filter, edged)

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

def read_plate(src_path):
    img = cv2.imread(src_path)
    gray, b_filter, edged = handle_img(img)
    
    contours = get_contours(edged) 
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    show_img(img)
    show_img(gray)
    show_img(b_filter)
    
    roi = get_roi(contours)
    number_plate = crop_image(b_filter, roi)

    text = pytesseract.image_to_string(number_plate, "por")
    print(text)
    show_img(number_plate)

read_plate("images/placa_veicular_1.jpg")