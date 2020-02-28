# libraries
import cv2
import time
from mtcnn import MTCNN
from threading import Thread
from PIL import Image

from facerecognition import FaceRecog

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = FaceRecog()


def detect_face(frame, detect_method="mtcnn"):
    try:
        # using mtcnn to detect faces
        if detect_method=="mtcnn":
            faces = detector.detect_faces(frame)
            # for faces detected, draw a box around it
            for face in faces:
                # get coordinates
                x, y, w, h = face['box']
                # plot in frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # using haar to detect faces
        if detect_method=="haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            # for faces detected, draw a box around it
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    except Exception as e:
        img = cv2.imread("static/background.png")   # reads an image in the BGR format
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return frame

def extract_faces(frame, x, y, w, h):
    # extract face from image with given coordinates
    im = Image.fromarray(frame[y:y + h, x:x + w])
    face_array = im.resize((160, 160))
    return face_array

def recog_face(frame, detect_method='mtcnn'):
    # using mtcnn to output list of faces
    list_faces = []
    if detect_method=='mtcnn':
        faces = detector.detect_faces(frame)
        for face in faces:
            # get coordinates
            x, y, w, h = face['box']
            list_faces.append(extract_faces(frame, x, y, w, h))
    elif detect_method=='haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2,5)
        for (x, y, w, h) in faces:
            list_faces.append(extract_faces(frame, x, y, w, h))
    return list_faces

class Recogniser:
    '''
    separate out facial recogniser 
    '''
    def __init__(self, faces):
        self.faces = faces
        self.thread = Thread(target=self.update_recogniser, args=())
        self.thread.daemon = True
        self.thread.start()
        # define the recognise

    def recognize_this(self):
        #recognizer function here
        recognizer.face_recognition_thread(self.faces)

    def update_recogniser(self):
        # Read the next frame from the stream in a different thread
        if self.faces is not None:
            self.recognize_this()
        time.sleep(.01)
