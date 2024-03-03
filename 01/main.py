from curses.ascii import isalpha, isdigit
import pytesseract
import cv2
import re
import imutils
import easyocr
import numpy as np
# import regex as re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

alpha2num = {
    "A": set(),
    "B": set("8"),
    "C": set("0"),
    "D": set("0"),
    "E": set(),
    "F": set(),
    "G": set("0"),
    "H": set(),
    "I": set("17"),
    "J": set(),
    "K": set(),
    "L": set(),
    "M": set(),
    "N": set(),
    "O": set("0"),
    "P": set(),
    "Q": set("0"),
    "R": set(),
    "S": set("5"),
    "T": set("1"),
    "U": set("0"),
    "V": set(),
    "W": set(),
    "X": set(),
    "Y": set(),
    "Z": set("7")
}


num2alpha = { "0": set(), "1": set(), "2": set(), "3": set(), "4": set(), "5": set(), "6": set(), "7": set(), "8": set(), "9": set() }
for k in alpha2num.keys():
    for v in alpha2num[k]:
        num2alpha[v].add(k)

PLATE_LENGTH = 7
# MERCOSUL_PATTERN_REGEX = r"[A-Z]{3}\d[A-Z]\d{2}"
# OLD_PATTERN_REGEX = r"[A-Z]{3}\d{4}"
# GENERIC_PATTERN_REGEX = r"[A-Z0-9]{7}"
# 0 = Digit; 1 = Letter
MERCOSUL_PATTERN = [1,1,1,0,1,0,0]
OLD_PATTERN = [1,1,1,0,0,0,0]

def is_patterned_plate(pattern: list, plate: str):
    if len(plate) != PLATE_LENGTH:
        return False
    
    for i in range(PLATE_LENGTH):
        if pattern[i]:
            if plate[i].isdigit():
                return False
        elif plate[i].isalpha():
            return False
    return True

def get_possibles_plates(pattern: list, text: str):
    possibles = ["".join(filter(str.isalnum, word)) for word in text.split(" ")]
    
    for i in range(PLATE_LENGTH):
        for p in possibles:
            if len(p) != PLATE_LENGTH:
                continue
            
            if pattern[i]:
                if p[i].isdigit():
                    for c in num2alpha[p[i]]:
                        possibles.append(p[:i] + c + p[i+1:])
            elif p[i].isalpha():
                for c in alpha2num[p[i]]:
                    possibles.append(p[:i] + c + p[i+1:])    
    
    return set(filter(lambda p: is_patterned_plate(pattern, p), possibles))
    

def show_img(img, name = "Imagem"):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 700, 300)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

img = cv2.imread("images/placa_veicular_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b_filter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(b_filter, 30, 200)
thresh = cv2.adaptiveThreshold(b_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
# _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

contours = get_contours(edged)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

roi = get_roi(contours)
number_plate = crop_image(img, roi)
number_plate_thresh = crop_image(thresh, roi)

text_thresh = pytesseract.image_to_string(number_plate_thresh, "por")
text = pytesseract.image_to_string(number_plate, "por")


print("Thresh:" + text_thresh)
print("Normal:" + text)
print(get_possibles_plates(MERCOSUL_PATTERN, text_thresh))
print(get_possibles_plates(MERCOSUL_PATTERN, text))
show_img(number_plate)
show_img(number_plate_thresh)