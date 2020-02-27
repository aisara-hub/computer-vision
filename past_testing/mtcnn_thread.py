# import the necessary packages
from threading import Thread
import cv2
from mtcnn import MTCNN

detector = MTCNN()

class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
        # start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

vid_cam = WebcamVideoStream().start()

# while looping video frame
while True:
    # capture frame-by-frame
    frame = vid_cam.read()
    # detect faces
    faces = detector.detect_faces(frame)
    # check faces are detected
    # if not faces:
    #     # loop faces if detected
    for face in faces:
        # get coordinates
        x1, y1, w, h = face['box']
        # plot in frame
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        # display video frame, with bounded rectangle on face
    cv2.imshow("TITLE", frame)
    # stop taking video
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Stop video
vid_cam.stop()

# Close all started windows
cv2.destroyAllWindows()
