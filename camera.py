import cv2
from threading import Thread
from mtcnn import MTCNN
import numpy as np
# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# multiple cameras - https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

# face detector
detector = MTCNN()

class VideoCamera():
    '''
    Setup source for threading and streaming to flask
    '''
    def __init__(self, src=0):
        # default source from webcam (0), set source as needed
        self.video = cv2.VideoCapture(src)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, image = self.video.read()
        # detect faces
        print(type(image))
        try:
            faces = detector.detect_faces(image)
            # for faces detected, draw a box around it
            for face in faces:
                # get coordinates
                x1, y1, w, h = face['box']
                # plot in frame
                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
            #key = self.waitKey(20)
            #if key == 'p':
            #   image = not image
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
