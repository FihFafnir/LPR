import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

PLATE_LENGTH = 7
NUM_2_ALPHA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B"}
ALPHA_2_NUM = {"A": "5", "B": "8", "G": "0", "I": "1", "O": "0", "S": "5", "Z": "2"}
LOWER_RANGE_COLOR = np.array([100, 127, 127])
UPPER_RANGE_COLOR = np.array([130, 255, 255])
# Mercosul Color in HSV: 115, 255, 153


def is_plate_format(text, is_mercosul=None):
    if len(text) != PLATE_LENGTH:
        return False

    return (
        text[:3].isalpha()
        and text[3].isdigit()
        and text[-2:].isdigit()
        and (
            True
            if is_mercosul is None
            else (text[4].isalpha() if is_mercosul else text[4].isdigit())
        )
    )


def is_mercosul_plate(plate_img):
    hsv_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, LOWER_RANGE_COLOR, UPPER_RANGE_COLOR)
    for row in mask:
        for item in row:
            if item != 0:
                return True
    return False


def get_possible_character(char, from_to):
    return from_to[char] if char in from_to else char


def get_possible_plates(text, is_mercosul=None):
    plate = plate_2 = ""

    if len(text) == PLATE_LENGTH:
        for char in text[:3]:
            plate += get_possible_character(char, NUM_2_ALPHA)
        for char in text[3:]:
            plate += get_possible_character(char, ALPHA_2_NUM)

        plate_2 += plate[:4]
        plate_2 += get_possible_character(plate[4], NUM_2_ALPHA)
        plate_2 += plate[5:]

    if is_mercosul is None:
        return {plate, plate_2}
    return {plate_2} if is_mercosul else {plate}


def read_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    thresh = cv2.threshold(b_filter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # invert = 255 - opening
    # mercosul = is_mercosul_plate(plate_img)
    mercosul = None

    cv2.imshow("Plate", invert)
    strings = pytesseract.image_to_string(
        invert,
        lang="por",
        config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ).split()
    strings.append("".join(strings))
    strings.extend(
        [plate for text in strings for plate in get_possible_plates(text, mercosul)]
    )
    plates = set(filter(lambda string: is_plate_format(string, mercosul), strings))

    if len(plates) > 0:
        print("Plate:", *plates)
