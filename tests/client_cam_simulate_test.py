import cv2
import socket
import struct

# Configuração do endereço IP e porta do servidor
SERVER_IP = '@ip_servidor'
SERVER_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame.")
        break

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    header = struct.pack('I', frame_id)
    message = header + buffer.tobytes()

    try:
        sock.sendto(message, (SERVER_IP, SERVER_PORT))
    except OSError as e:
        print(f"Erro ao enviar frame: {e}")

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock.close()
