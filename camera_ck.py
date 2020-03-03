import cv2, time
from threading import Thread
from datetime import datetime

import configs
from facial_detectors import detect_face, recog_face, Recogniser

# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# multiple cameras - https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

class VideoCamera:
    '''
    Setup source for threading and streaming to flask
    '''
    def __init__(self, src=0, detect_method="mtcnn"):
        print("initiating self: ",src)
        if configs.RUNNER == "taufiq":
            if src == 0:
                self.src = 'rtsp://admin:MJEVUD@192.168.0.8:554/H.264'
            if src == 1:
                self.src = 'rtsp://admin:EJVCDI@192.168.0.11:554/H.264'
            if src == 2:
                self.src = 2
            if src == 3:
                self.src = 3
        if configs.RUNNER == "daus":
            if src == 0:
                self.src = 0
            if src == 1:
                self.src = "rtsp://192.168.0.5:8080/video/h264"
            if src == 2:
                self.src = 2
            if src == 3:
                self.src = 3
        # default source from webcam (0), set source as needed
        self.stream = cv2.VideoCapture(self.src)
        self.global_timestamp = datetime.now().timestamp()
        self.detect_method = detect_method
        # start thread to read frames from video stream
        _, self.frame = self.stream.read()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.stream.isOpened():
                _, self.frame = self.stream.read()
            time.sleep(.01)

    def __del__(self):
        self.stream.release()
    
    def get_frame(self):
        if self.detect_method:
            jpeg = detect_face(self.frame, self.detect_method)
            # perform recogniser and save every second
            if (int(self.global_timestamp)-int(datetime.now().timestamp())) <0:
                faces = recog_face(self.frame, self.detect_method)
                self.global_timestamp += 1
                Recogniser(list_faces=faces)
        else:
            jpeg = self.frame
        _, jpeg = cv2.imencode('.jpg', jpeg)
        return jpeg.tobytes()

    def recognise(self):
        if (int(self.global_timestamp)-int(datetime.now().timestamp())) <0:
            faces = recog_face(self.frame, self.detect_method)
            self.global_timestamp += 1
            Recogniser(list_faces=faces)
