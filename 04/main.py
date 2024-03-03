import cv2
import pytesseract

def read_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Plate", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    data = pytesseract.image_to_string(invert, lang="eng", config="--psm 6")
    print("Plate: ", data)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    cv2.putText(frame, "Sei la", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    contours, h = cv2.findContours(thresh, 1, 2)
    largest_rectangle = [0, 0]

    for contour in contours:
        length = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, length, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > largest_rectangle[0]:
                largest_rectangle = [area, contour, approx]

    x, y, w, h = cv2.boundingRect(largest_rectangle[1])

    image = frame[y:y+h, x:x+w]
    cv2.drawContours(frame, [largest_rectangle[1]], 0, (0, 255, 0), 8)
    cropped = frame[y:y+h, x:x+w]
    cv2.putText(frame, "Plate", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
    cv2.imshow("Camera", frame)
    cv2.drawContours(frame, [largest_rectangle[1]], 0, (0, 255, 0), 18)

    # Leitura da Placa Capturada
    read_plate(image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
