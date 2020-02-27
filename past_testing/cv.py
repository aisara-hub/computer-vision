# %% libraries
import cv2 as cv
import glob

# %% 
# preprocess the data and train
def detect_display(frame):
    # cv load images in BGR NOT RGB - convert to grayscale
    frame_gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equalise using histogram equalization
    frame_gray = cv.equalizeHist(frame_gray)

    cv.imshow("Computed Image", frame_gray)


# %%
test_images = [cv.imread(file) for file in glob.glob("test_images/*.jpeg")]


# %%
detect_display(test_images[0])


# %%
len(test_images)

# %%
