import cv2, time
from threading import Thread
from datetime import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
import configs
from facial_detectors import detect_face, recog_face, recognition_thread

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
        try:
            #set initial empty array for previous frame in the first grabbed frame
            self.previous_frame = np.zeros(self.frame.shape, dtype=np.uint8)
        except Exception as e:
            self.previous_frame = self.frame
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.stream.isOpened():
                _, self.frame = self.stream.read()
                self.current_frame = self.frame
            time.sleep(.01)

    def __del__(self):
        self.stream.release()

    def compare_frame(self, previous_frame, current_frame):
        grayprevious = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        graycurrent = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        score, frame_diff = ssim(graycurrent, grayprevious, full=True)
        self.previous_frame = current_frame
        return score

    def get_frame(self):
        if self.detect_method:
            jpeg = detect_face(self.frame, self.detect_method)
            # perform recogniser and save every second
            # performa frame differencing from previous frame

            # make threading for faces recog
            Thread(target=self.recognition, args=(), daemon=True).start()
        else:
            jpeg = self.frame
        _, jpeg = cv2.imencode('.jpg', jpeg)
        return jpeg.tobytes()

    def recognition(self):
        if self.previous_frame is not None and self.current_frame is not None:
            score = self.compare_frame(self.previous_frame, self.current_frame)
            if score < 0.85:
                if (int(self.global_timestamp) - int(datetime.now().timestamp())) < 0:
                    faces = recog_face(self.frame, self.detect_method)
                    self.global_timestamp += 1
                    recognition_thread(faces)


