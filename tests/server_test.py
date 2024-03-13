import cv2
import socket
import numpy as np
import struct

from util import read_plate, treat_image, get_ipv4

SERVER_IP = get_ipv4()
SERVER_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

print(f"Servidor escutando em {SERVER_IP}:{SERVER_PORT}")

plate_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)

try:
    while True:
        packet, _ = sock.recvfrom(65535)

        frame_id = struct.unpack('I', packet[:4])[0]

        frame_data = packet[4:]

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        b_filter = treat_image(frame)[2]
        plates = plate_classifier.detectMultiScale(b_filter, 1.2, 4)

        for x, y, w, h in plates:
            readed = read_plate(frame[y: y + h, x: x + w])
            if len(readed) > 0:
                print("Plate:", *plates)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Camera:", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Encerrando o servidor.")

sock.close()
cv2.destroyAllWindows()
