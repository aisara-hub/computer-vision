# libraries
import cv2
from mtcnn import MTCNN

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Detector:
    def __init__(self, image):
        self.image = image

    def extract_faces(self, x, y, w, h):
        # extract face from image with given coordinates
        im = Image.fromarray(self.image[y:y + h, x:x + w])
        face_array = im.resize((160, 160))
        return face_array
    
    def mtcnn_faces(self):
        # using mtcnn to detect faces
        # list_face = []
        try:
            faces = detector.detect_faces(self.image)
            # for faces detected, draw a box around it
            for face in faces:
                # get coordinates
                x, y, w, h = face['box']
                # # save each face to list
                # list_face.append(self.extract_faces(x, y, w, h))
                # plot in frame
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.image
    
    def haar_faces(self):
        # using haar to detect faces
        list_face = []
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            # for faces detected, draw a box around it
            for (x, y, w, h) in faces:
                # save each face to list
                list_face.append(self.extract_faces(x, y, w, h))
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except Exception as e:
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.image, list_face