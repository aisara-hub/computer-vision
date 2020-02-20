# %% libraries
import cv2 as cv
from mtcnn import MTCNN

# %% setups
# detector - default mtcnn
detector = MTCNN()

# start capturing video
vid_cam = cv.VideoCapture(0)

# while looping video frame
while True:
    # capture frame-by-frame
    _, frame = vid_cam.read()
    # detect faces
    faces = detector.detect_faces(frame)
    # check faces are detected
    # if not faces:
    #     # loop faces if detected
    for face in faces:
        # get coordinates
        x1, y1, w, h = face['box']
        # plot in frame
        cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        # display video frame, with bounded rectangle on face
    cvshow = cv.imshow("TITLE", frame)
    # stop taking video
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

# Stop video
vid_cam.release()

# Close all started windows
cv.destroyAllWindows()
