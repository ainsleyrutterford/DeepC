import models
import data
import numpy as np
import cv2 as cv
from ctypes import CDLL
import ctypes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from skimage import morphology
from tqdm import tqdm

ret = os.system("cc -fPIC -shared -std=c99 -o hausdorff.so hausdorff.c -lm")

if ret == 0:
    print("Successfully compiled C library.")
    C = CDLL(os.path.abspath("hausdorff.so"))
else:
    print("Couldn't compile C library. Exiting...")
    exit()

def get_accuracy(folder, tests, weights):
    image_gen = data.test_generator(f"{folder}/image", num_image=tests)
    label_gen = data.test_generator(f"{folder}/label", num_image=tests)
    model = models.compile("unet2D", "binary_crossentropy", weights)
    results = model.predict_generator(image_gen, tests, verbose=0)

    accuracies = np.zeros(tests)

    for i, item in enumerate(results):
        image = item[:, :, 0]
        image *= 255
        image = image.astype(np.uint8)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

        label = next(label_gen)[0, :, :, 0]
        label *= 255
        label = label.astype(np.uint8)
        _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)

        # Step 1: Create an empty skeleton
        size = np.size(image)
        skel = np.zeros(image.shape, np.uint8)

        image[image == 255] = 1

        skel = morphology.skeletonize(image)
        skel = skel.astype(int) * 255

        image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
        label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

        if (len(image_boundaries) == 0):
            image_boundaries = [0, 0]

        c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
        c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)

        C.hausdorff.restype = ctypes.c_double
        accuracies[i] = C.hausdorff(c_image_boundaries, len(image_boundaries),
                                    c_label_boundaries, len(label_boundaries))

        # if tests == 66:
        #     cv.imwrite(f"skel/{i}_skel.png", np.array(skel))
        #     print(i, accuracies[i])

    return np.mean(accuracies)

epochs = 20

accuracies = np.zeros(epochs)
val_accuracies = np.zeros(epochs)

for e in tqdm(range(epochs)):
    accuracies[e] = get_accuracy("data/train", 322, f"isambard-{e+1:02d}.hdf5")
    val_accuracies[e] = get_accuracy("data/test", 66, f"isambard-{e+1:02d}.hdf5")

print(list(accuracies))
print(list(val_accuracies))