# libraries
import cv2
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from datetime import datetime

from facial_detectors import detect_face, recog_face, Recogniser
from checking import CountsPerSec, putIterationsPerSec
import configs

# setup reading frames for threading
class VideoGet:
    def __init__(self, src=0, detect_method="mtcnn"):
        self.stream = cv2.VideoCapture(src)
        self.grabbed = self.stream.read()[0]
        self.detect = detect_method
        if self.detect:
            self.frame = detect_face(self.stream.read()[1], detect_method)
        else:
            self.frame = self.stream.read()[1]
        self.stopped = False
    
    def start(self):    
        Thread(target=self.get, args=()).start()
        return self
    
    def get(self, detect_recog=True):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed = self.stream.read()[0]
                if self.detect:
                    self.frame = detect_face(self.stream.read()[1])
                else:
                    self.frame = self.stream.read()[1]   
    
    def stop(self):
        self.stopped = True

# compiled task
class Task:
    def __init__(self, source, detect_method='mtcnn'):
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
        # self.video = cv2.VideoCapture(self.source)
        self.video = VideoGet(self.source, detect_method).start()
        self.detect_method = detect_method
        self.global_timestamp = datetime.now().timestamp()
        self.cps = CountsPerSec().start()
    
    def stream(self):
        # while True:
        #     frame = self.video.frame
        #     # frame = stream_detect(frame)
        #     frame = putIterationsPerSec(frame, self.cps.countsPerSec())
        #     self.cps.increment()
        #     yield (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' +  cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
        
        while True:
            frame = self.video.frame
            frame = putIterationsPerSec(frame, self.cps.countsPerSec())
            self.cps.increment()
            # recogniser
            if (int(self.global_timestamp)-int(datetime.now().timestamp())) <0:
                faces = recog_face(frame, self.detect_method)
                self.global_timestamp += 1
                Recogniser(list_faces=faces)
            # return frame
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +  cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
    
    def recognise(self):
        while True:
            if (int(self.global_timestamp)-int(datetime.now().timestamp())) <0:
                frame = self.video.frame
                faces = recog_face(frame, self.detect_method)
                self.global_timestamp += 1
                Recogniser(list_faces=faces)