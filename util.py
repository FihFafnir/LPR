import cv2
import pytesseract
import numpy as np
import socket

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

NUM_2_ALPHA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B"}
ALPHA_2_NUM = {"A": "5", "B": "8", "G": "0", "I": "1", "O": "0", "S": "5", "Z": "2"}
LOWER_RANGE_COLOR = np.array([100, 127, 127])
UPPER_RANGE_COLOR = np.array([130, 255, 255])


PLATE_LENGTH = 7
OLD_PLATE_PATTERN = "AAA0000"
MERCOSUL_PLATE_PATTERN = "AAA0A00"

# Mercosul Color in HSV: 115, 255, 153


def is_old_format(text):
    if len(text) != PLATE_LENGTH:
        return False

    for i in range(PLATE_LENGTH):
        if text[i].isalpha() != OLD_PLATE_PATTERN[i].isalpha():
            return False

    return True


def is_mercosul_format(text):
    if len(text) != PLATE_LENGTH:
        return False

    for i in range(PLATE_LENGTH):
        if text[i].isalpha() != MERCOSUL_PLATE_PATTERN[i].isalpha():
            return False

    return True


def is_plate_format(text, is_mercosul=None):
    if len(text) != PLATE_LENGTH:
        return False

    if is_mercosul is None:
        return is_mercosul_format(text) | is_old_format(text)

    return is_mercosul_format(text) if is_mercosul else is_old_format(text)


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
    mercosul_pattern_plate = ""
    old_pattern_plate = ""

    if len(text) != PLATE_LENGTH:
        return set("")

    for i in range(PLATE_LENGTH):
        mercosul_pattern_plate += get_possible_character(
            text[i], NUM_2_ALPHA if MERCOSUL_PLATE_PATTERN[i].isalpha() else ALPHA_2_NUM
        )
        old_pattern_plate += get_possible_character(
            text[i], NUM_2_ALPHA if OLD_PLATE_PATTERN[i].isalpha() else ALPHA_2_NUM
        )

    if is_mercosul is None:
        return {mercosul_pattern_plate, old_pattern_plate}
    return {mercosul_pattern_plate} if is_mercosul else {old_pattern_plate}


def treat_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    thresh = cv2.threshold(b_filter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)

    return gray, blur, b_filter, thresh, invert


def read_plate(plate_img):
    treated_img = treat_image(plate_img)[4]
    mercosul = None
    # mercosul = is_mercosul_plate(plate_img)

    cv2.imshow("Plate", treated_img)

    strings = pytesseract.image_to_string(
        treated_img,
        lang="por",
        config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ).split()
    strings.append("".join(strings))
    strings.extend(
        [plate for text in strings for plate in get_possible_plates(text, mercosul)]
    )

    return set(filter(lambda string: is_plate_format(string, mercosul), strings))


def get_ipv4():
    try:
        host_name = socket.gethostname()
        ipv4_address = socket.gethostbyname(host_name)
        return ipv4_address
    except socket.error as e:
        print(f"Erro ao obter o endereço IPv4: {e}")
        return None
