import cv2
from threading import Thread
from mtcnn import MTCNN
import numpy as np
import configs
import tensorflow as tf
from os import urandom, environ

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# streaming with flask - https://stackoverflow.com/questions/49939859/flask-video-stream-using-opencv-images
# fix frame lag - https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# mtcnn detection - https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# multiple cameras - https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

# face detector

"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)"""
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class VideoCamera():
    '''
    Setup source for threading and streaming to flask
    '''
    def __init__(self, src=0):
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
    
    def get_frame(self):

        _, image = self.video.read()
        if not _:
            self.video = cv2.VideoCapture(self.src)
            _, image = self.video.read()

        # detect faces
        print(type(image))
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            #faces = detector.detect_faces(image)
            # for faces detected, draw a box around it
            count = 0
            for (x1,y1,w,h) in faces:
                print("found face")
                # get coordinates
                #x1, y1, w, h = face['box']
                # plot in frame
                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
                #if using haarcascade
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print("checking storage")
                configs.assure_path_exists(configs.STORAGE_PATH)
                count += 1
                # Save the captured image into the datasets folder
                print("saving to storage")
                #if using haarcascade
                #cv2.imwrite(configs.STORAGE_PATH+"/User_" + str(urandom(7).hex()) + '_' + str(count) + ".jpg", image[y:y+h,x:x+w])

                cv2.imwrite(configs.STORAGE_PATH + "/User_" + str(urandom(7).hex()) + '_' + str(count) + ".jpg",image[y1:y1 + h, x1:x1 + w])
            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
            #key = self.waitKey(20)
            #if key == 'p':
            #   image = not image
        except Exception as e:
            print("something happen:", e)
            img = cv2.imread("static/background.png")   # reads an image in the BGR format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
