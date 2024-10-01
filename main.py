from util import *
import glob


cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord("q"):
    # Exemplo de uso com um URL RTSP
    # rtsp_url = 'rtsp://seu_usuario:senha@endereco_ip_da_camera:554/caminho_do_stream' or http:endereco_ip_da_camera:8080
    # streamer = VideoStreamer(rtsp_url)
    # streamer.start_capture()

    ret, frame = cap.read()
    recognize_plate(frame)

cap.release()
cv2.destroyAllWindows()

# filepaths = sorted(glob.glob("./images/*.jpg"))

# for filepath in filepaths:
#     img = cv2.imread(filepath)

#     recognize_plate(img)

#     while cv2.waitKey(1) != ord("q"):
#         pass
