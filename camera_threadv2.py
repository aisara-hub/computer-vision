# libraries
import cv2
from concurrent.futures import ThreadPoolExecutor
from facial_detectors import detect_mtcnn, detect_haar
import configs

# function to parallalise
def read(source):
    _, frame = source.read()
    return frame

def detect(frame, detector="haar"):
    if detector == "mtcnn": 
        frame = detect_mtcnn(frame)
    elif detector == "haar":
        frame = detect_haar(frame)
    return frame

class Task:
    def __init__(self, source, detect=True, recognise=True):
        print("initiating self: ",source)
        if configs.RUNNER == "taufiq":
            if source == 0:
                self.source = 'rtsp://admin:MJEVUD@192.168.0.8:554/H.264'
            if source == 1:
                self.source = 'rtsp://admin:EJVCDI@192.168.0.11:554/H.264'
            if source == 2:
                self.source = 2
            if source == 3:
                self.src = 3
        if configs.RUNNER == "daus":
            self.source = source
            pass
        # default source from webcam (0), set source as needed
        self.video = cv2.VideoCapture(self.source)
        self.detect = detect
        self.recognise = recognise
    
    def run(self):
        executor = ThreadPoolExecutor(max_workers=5)  
        while True:
            f_read = executor.submit(read, self.video)
            frame = f_read.result()
            if self.detect:
                f_detect = executor.submit(detect, frame)
                frame = f_detect.result()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
        