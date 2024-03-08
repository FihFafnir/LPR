import cv2


class VideoStreamer:
    def __init__(self, video_stream_url):
        """Inicializa o streamer com o URL do vídeo."""
        self.video_stream_url = video_stream_url
        self.cap = None

    def start_capture(self):
        """Inicia a captura do vídeo do URL fornecido."""
        self.cap = cv2.VideoCapture(self.video_stream_url)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir o fluxo de vídeo.")
            return

        self.display_video()

    def display_video(self):
        """Exibe o vídeo em uma janela."""
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Video Stream', frame)
            else:
                print("Erro: Não foi possível ler o frame.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def release_resources(self):
        """Libera os recursos."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados e janela fechada.")
