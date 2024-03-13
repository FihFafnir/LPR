import cv2
import pytesseract
import numpy as np
import socket

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
        config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
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
