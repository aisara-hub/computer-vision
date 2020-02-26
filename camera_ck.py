import cv2, time
from threading import Thread
from mtcnn import MTCNN
import numpy as np
import configs
import tensorflow as tf
from PIL import Image
from facerecognition import FaceRecog
from datetime import datetime
from os import urandom, environ

# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# multiple cameras - https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = FaceRecog()

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
                self.src = 'rtsp://admin:EJVCDI@192.168.0.11:554/H.264'
            if src == 2:
                self.src = 2
            if src == 3:
                self.src = 3
        if configs.RUNNER == "daus":
            pass
        # default source from webcam (0), set source as needed
        self.video = cv2.VideoCapture(src)
        self.global_timestamp = datetime.now().timestamp()
        # start thread to read frames from video stream
        _, self.image = self.video.read()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.video.isOpened():
                (_, self.image) = self.video.read()
            time.sleep(.01)

    def __del__(self):
        self.video.release()
    
    def extract_faces(self, image, x, y, w, h):
        # extract face from image with given coordinates
        im = Image.fromarray(image[y:y + h, x:x + w])
        face_array = im.resize((160, 160))
        return face_array
    
    def mtcnn_faces(self, image):
        # using mtcnn to detect faces
        list_face = []
        try:
            faces = detector.detect_faces(image)
            # for faces detected, draw a box around it
            for face in faces:
                # get coordinates
                x, y, w, h = face['box']
                # save each face to list
                list_face.append(self.extract_faces(image, x, y, w, h))
                # plot in frame
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except Exception as e:
            print(e)
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg, list_face
    
    def haar_faces(self, image):
        # using haar to detect faces
        list_face = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            # for faces detected, draw a box around it
            for (x, y, w, h) in faces:
                # save each face to list
                list_face.append(self.extract_faces(image, x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print("Date Time: ",datetime.now().timestamp())
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg, list_face
    
    def get_frame(self):
        # # normal - without multithreading
        # _, image = self.video.read()
        # if configs.RUNNER == "taufiq":
        #     if not _:
        #         self.video = cv2.VideoCapture(self.src)
        #         _, image = self.video.read()
        # # choose detector (haar / mtcnn)
        # jpeg, faces = self.mtcnn_faces(image)
        # with multithreading 
        jpeg, faces = self.mtcnn_faces(self.image)
        # perform recogniser and save every second
        if (int(self.global_timestamp)-int(datetime.now().timestamp())) <0:
            self.global_timestamp += 1
            Recogniser(face=faces)
        return jpeg.tobytes()

class Recogniser():
    '''
    separate out facial recogniser 
    '''
    def __init__(self, face):
        self.face = face      
        self.recognize_this()
    
    def recognize_this(self):
        # define the recogniser function here
        recognizer.face_recognition(self.face, threshold=0.95)

