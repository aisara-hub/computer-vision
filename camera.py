import cv2
from threading import Thread
from mtcnn import MTCNN

# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

# face detector
detector = MTCNN()

class VideoCamera:
    '''
    Setup source for threading and streaming to flask
    '''
    def __init__(self, src=0):
        # default source from webcam (0), set source as needed
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, image = self.video.read()
        # detect faces
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
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()