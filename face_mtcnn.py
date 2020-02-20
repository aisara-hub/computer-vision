# %% libaries
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

# %% Functions
# using https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

# default weight mtcnn
detector = MTCNN()

# detect faces
def detected_faces(image, result_list):
    # plot images
    plt.imshow(image)
    # get context for drawing boxes
    ax = plt.gca()
    # plotting each box
    for result in result_list:
        # get coordinates
        x, y, w, h = result['box']
        # create shape
        rect = Rectangle((x,y), w, h, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show plot
    plt.show()

# extract each identified face in each image
def draw_faces(image, result_list):
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, w, h = result_list[i]['box']
        x2, y2 = x1 + w, y1 + h
        # define subplot
        plt.subplot(1, len(result_list), i+1)
        plt.axis('off')
        # plot face
        plt.imshow(image[y1:y2, max(x1, 0):x2])
    # show plot
    plt.show()

def detect_draw(image):
    faces = detector.detect_faces(image)
    detect_faces(image, faces)
    draw_faces(image, faces)


# %% load images
for name in glob.glob('test_images/*'):
    image = plt.imread(name)
    detect_draw(image)

# # %% load & train images
# for name in glob.glob('test_images/*'):
#     image = plt.imread(name)
#     faces = detector.detect_faces(image)
#     save_name = name.replace('test_images\\', '')
#     detected_faces(image, faces)
#     plt.savefig('extracted_images/detect_{}'.format(save_name))
#     draw_faces(image, faces)
#     plt.savefig('extracted_images/draw_{}'.format(save_name))



# %%
