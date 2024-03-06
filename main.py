import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
plate_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
cap = cv2.VideoCapture(0)

num2alpha = { "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B" }
alpha2num = dict()

for key, value in num2alpha.items():
    alpha2num[value] = key

def is_plate_format(text):
    if len(text) != 7:
        return False

    return text[:3].isalpha() and text[3].isdigit() and text[-2:].isdigit()

def get_possible_plate(text):
    plate = ""

    if is_plate_format(text):
        return text
    if len(text) == 7:      
        plate += num2alpha[text[0]] if (text[0].isdigit() and text[0] in num2alpha) else text[0]
        plate += num2alpha[text[1]] if (text[1].isdigit() and text[1] in num2alpha) else text[1]
        plate += num2alpha[text[2]] if (text[2].isdigit() and text[2] in num2alpha) else text[2]
         
        plate += alpha2num[text[3]] if (text[3].isalpha() and text[3] in alpha2num) else text[3]
        plate += text[4]
        plate += alpha2num[text[5]] if (text[5].isalpha() and text[5] in alpha2num) else text[5]
        plate += alpha2num[text[6]] if (text[6].isalpha() and text[6] in alpha2num) else text[6]  
    return plate 


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
    texts = pytesseract.image_to_string(invert, lang="por", config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ").split()
    texts += "".join(texts)
    texts += [get_possible_plate(text) for text in texts]
    plates = set(filter(is_plate_format, texts))

    if len(plates) > 0:
        print("Plate:", *plates) 

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
