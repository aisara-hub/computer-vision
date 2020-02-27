# %% libraries
from skimage.feature import Cascade
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

# %% training faces detection
trained_file = data.lbp_frontal_face_cascade_filename()

detector = Cascade(trained_file)

def show_detected_face(result, detected, title="Face Image"):
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')

    for patch in detected: 
        img_desc.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False, color='r', linewidth=2)
        )
    
    plt.show()

# %% loading images
test_images = []
for name in glob.glob('test_images/*'):
    test_images.append(plt.imread(name))

# %% detected faces
face_detect = []
for i in test_images:
    detected = detector.detect_multi_scale(img = i,
                                        scale_factor=1.2,
                                        step_ratio=1,
                                        min_size=(10,10),
                                        max_size=(200,200))
    show_detected_face(i, detected)


# %%
