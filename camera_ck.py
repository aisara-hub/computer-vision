import cv2
from threading import Thread
from mtcnn import MTCNN
import numpy as np
import configs
import tensorflow as tf

# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# multiple cameras - https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoCamera:
    '''
    Setup source for threading and streaming to flask
    '''
    def __init__(self, src=0):
        print("initiating self: ",src)
        if configs.RUNNER == "taufiq":
            if src == 0:
                self.src = 'rtsp://admin:MJEVUD@192.168.0.8:554/H.264'
            if src == 1:
                self.src = 1
            if src == 2:
                self.src = 2
            if src == 3:
                self.src = 3
            environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        if configs.RUNNER == "daus":
            pass
        # default source from webcam (0), set source as needed
        self.video = cv2.VideoCapture(src)
    
    def __del__(self):
        self.video.release()
    
    def mtcnn_faces(self, image):
        # using mtcnn to detect faces
        try:
            faces = detector.detect_faces(image)
            # for faces detected, draw a box around it
            for face in faces:
                # get coordinates
                x1, y1, w, h = face['box']
                # plot in frame
                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg
    
    def haar_faces(self, image):
        # using haar to detect faces
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            # for faces detected, draw a box around it
            for (x,y,w,h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg
    
    def read_frames(self):
        # separate frame reading for threading
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int)
        
    def get_frame(self):
        _, image = self.video.read()
        if configs.RUNNER == "taufiq":
            if not _:
                self.video = cv2.VideoCapture(self.src)
                _, image = self.video.read()
        
        # choose detector (haar / mtcnn)
        jpeg = self.haar_faces(image)

        return jpeg.tobytes()
