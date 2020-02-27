# libraries
import cv2
from threading import Thread
from datetime import datetime
import configs
from facial_detectors import Detector

# multithreading - https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        try:
            return self._num_occurrences / elapsed_time
        except Exception as e:
            return 0

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self, detect_recog=True):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
    
    def stop(self):
        self.stopped = True

class VideoProcess:
    """
    Process frame using dedicated thread
    """
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self.processed = frame
        self.faces = []

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
            self.processed = Detector(self.frame).mtcnn_faces()
            
        

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Show frame using dedicated thread
    """
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
    
    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        pass
        # while not self.stopped:
        #     cv2.imshow("Video", self.frame)
        #     if cv2.waitKey(1) == ord("q"):
        #         self.stopped = True

    def stop(self):
        self.stopped = True

class threadBoth:
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """
    def __init__(self,source=0):
        print("initiating self: ",source)
        if configs.RUNNER == "taufiq":
            if source == 0:
                self.source = 'rtsp://admin:MJEVUD@192.168.0.8:554/H.264'
            if source == 1:
                self.source = 'rtsp://admin:EJVCDI@192.168.0.11:554/H.264'
            if source == 2:
                self.source = 2
            if source == 3:
                self.source = 3
        if configs.RUNNER == "daus":
            pass
        self.video_getter = VideoGet(source).start()
        self.video_processor = VideoProcess(self.video_getter.frame).start()
        self.video_shower = VideoShow(self.video_processor.frame).start()
        self.cps = CountsPerSec().start()

    def return_frame(self):
        while True:
            if self.video_getter.stopped or self.video_shower.stopped:
                self.video_shower.stop()
                self.video_processor.stop()
                self.video_getter.stop()
                break
            
            # get the frame
            print('extracting frame')
            frame = self.video_getter.frame
            # edit the frame
            frame = putIterationsPerSec(frame, self.cps.countsPerSec())
            print('sendign to process')
            self.video_processor.frame = frame
            # push edited frame to show
            print('sending to show')
            self.video_shower.frame = self.video_processor.processed
            self.cps.increment()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +  cv2.imencode('.jpg', self.video_shower.frame)[1].tobytes() + b'\r\n\r\n')
    
