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

plate_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)


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


def resize_image(img):
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_width = 480
    new_height = int(new_width * aspect_ratio)

    return cv2.resize(img, (new_width, new_height))


def treat_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    b_filter = cv2.bilateralFilter(blur, 11, 17, 17)
    thresh = cv2.threshold(b_filter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    invert = 255 - closing

    return gray, blur, b_filter, thresh, invert


def crop_plate(plate_img):
    cnts = cv2.findContours(plate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and area > 1000 and area < 100000:
            # Get the left top corner, width and height of the contour
            # bounding rectangle:
            x, y, w, h = cv2.boundingRect(c)

            # Slice the area of interest from the original image:
            plate_img = plate_img[y : y + h, x : x + w]
            # return teste
            # cv2.imshow("teste", plate_img)

    return plate_img


import glob


def read_plate(plate_img):
    treated_img = treat_image(plate_img)[4]
    treated_img = crop_plate(treated_img)
    cv2.imshow("Plate", treated_img)
    mercosul = None
    # mercosul = is_mercosul_plate(plate_img)
    # cv2.imwrite(
    #     f"./treated_images/plate_{len(glob.glob('./treated_images/plate_*.jpg')) + 1}.jpg",
    #     plate_img,
    # )

    # cv2.imwrite(
    #     f"./treated_images/treated_plate_{len(glob.glob('./treated_images/treated_plate_*.jpg')) + 1}.jpg",
    #     treated_img,
    # )

    strings = pytesseract.image_to_string(
        treated_img,
        lang="eng",
        config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ).split()
    strings.append("".join(strings))
    strings.extend(
        [plate for text in strings for plate in get_possible_plates(text, mercosul)]
    )
    # print(strings)

    return set(filter(lambda string: is_plate_format(string, mercosul), strings))


placas_registradas = {
    "QSB2148": "Pessoa 1",
    "RLW8F80": "Pessoa 2",
    "QSB1175": "Pessoa 3",
    "QFZ4J04": "Pessoa 4",
    "QFS9889": "Pessoa 5",
    # ""
}


def similarity(plate_1, plate_2):
    size = min(len(plate_1), len(plate_2))
    count = 0
    for i in range(size):
        if plate_1[i] == plate_2[i]:
            count += 1

    return count / size


def max_similarity(plates, plate):
    similarity_plate = ""
    max_similarity = 0.0

    for p in plates:
        current = similarity(p, plate)
        if current > max_similarity:
            max_similarity = current
            similarity_plate = p

    return similarity_plate, max_similarity


def recognize_plate(img):
    img = resize_image(img)
    # cv2.imwrite(
    #     f"./treated_images/car_{len(glob.glob('./treated_images/car_*.jpg')) + 1}.jpg",
    #     img,
    # )
    b_filter = treat_image(img)[2]
    # cv2.imwrite(
    #     f"./treated_images/treated_car_{len(glob.glob('./treated_images/treated_car_*.jpg')) + 1}.jpg",
    #     b_filter,
    # )
    plates = plate_classifier.detectMultiScale(b_filter, 1.2, 4)

    for x, y, w, h in plates:
        readed = read_plate(img[y : y + h, x : x + w])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Camera:", img)
        if len(readed) > 0:
            # similarity_plate = max(
            #     map(lambda r: max_similarity(placas_registradas, r), readed),
            #     key=lambda s: s[1],
            # )

            img = cv2.putText(
                img,
                f"{"/".join(readed)}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
                False,
            )
            # img = cv2.putText(
            #     img,
            #     f"{similarity_plate[0]} ({round(similarity_plate[1] * 100, 2)}%) - {placas_registradas[similarity_plate[0]]}",
            #     (x, y - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            #     False,
            # )
            # cv2.imwrite(
            #     f"./treated_images/readed_plate_{len(glob.glob('./treated_images/readed_plate_*.jpg')) + 1}.jpg",
            #     img,
            # )

            # print(similarity_plate)

            print("Plate:", *readed)
            # return set(readed)

    cv2.imshow("Camera:", img)
    # return set()


def get_ipv4():
    try:
        host_name = socket.gethostname()
        ipv4_address = socket.gethostbyname(host_name)
        return ipv4_address
    except socket.error as e:
        print(f"Erro ao obter o endereço IPv4: {e}")
        return None
