import argparse

# All possible arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="unet2D", help="Architecture [unet2D | unet3D | segnet2D]")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to test")
parser.add_argument("--size", type=int, default=256, help="Size to reshape the images to when training")
parser.add_argument("--train_samples", type=int, default=322, help="Number of training samples")
parser.add_argument("--val_samples", type=int, default=10, help="Number of validation samples")
parser.add_argument("--verbose", action="store_true", help="Show TensorFlow startup messages and warnings")
parser.add_argument("--ablated", action="store_true", help="Use ablated architecture")
parser.add_argument("--dir", type=str, default="data", help="Data directory")
parser.add_argument("--weights", type=str, default="checkpoint", help="Weights files prefix")
args = parser.parse_args()

# If the --verbose argument is not supplied, suppress all of the TensorFlow startup messages.
if not args.verbose:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import models
import data
import numpy as np
import cv2 as cv
from ctypes import CDLL
import ctypes
from skimage import morphology
from tqdm import tqdm

# Compile the C library.
ret = os.system("cc -fPIC -shared -std=c99 -o accuracy.so accuracy.c -lm")

# If the compilation was not successful then exit.
if ret == 0:
    print("Successfully compiled C library.")
    C = CDLL(os.path.abspath("accuracy.so"))
else:
    print("Couldn't compile C library. Exiting...")
    exit()

def get_accuracy(folder, tests, weights):
    """
    Calculate the accuracy achieved.

    Args:
        folder: (str) the base directory of the samples.
        tests: (int) the number of tests to carry out.
        weights (str) the name of the file that the weights are
            saved in.
    Returns:
        accuracy: (int) the average accuracy across all of the samples
            inside the specified directory.
    """

    image_gen = data.test_generator(f"{folder}/image", num_image=tests)
    label_gen = data.test_generator(f"{folder}/label", num_image=tests)

    # Compile the model using a compile() method defined in models/__init__.py.
    model = models.compile(arch=args.model, pretrained_weights=weights, size=args.size, abl=args.ablated)

    results = model.predict_generator(image_gen, tests, verbose=0)

    # Create an empty accuracies array.
    accuracies = np.zeros(tests)

    # For each prediction...
    for i, item in enumerate(results):
        # Extract the 2D prediction, multiply it by 255 so that values are
        # now in the range of 0-255, and threshold using Otsu's method.
        image = item[:, :, 0]
        image *= 255
        image = image.astype(np.uint8)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

        # Extract the 2D label, multiply it by 255 so that values are
        # now in the range of 0-255, and threshold.
        label = next(label_gen)[0, :, :, 0]
        label *= 255
        label = label.astype(np.uint8)
        _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)

        # Turn all 255s into 1s for the skeletonization.
        image[image == 255] = 1

        # Skeletonize the thresholded prediction and turn it back into
        # a range of 0-255.
        skel = morphology.skeletonize(image)
        skel = skel.astype(int) * 255

        # Find the white pixels and save their positions as a flattened 1D
        # array to be used by the C library.
        image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
        label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

        # If a black image is produced then error would be inf. Place
        # a single white pixel to get a finite accuracy score.
        if (len(image_boundaries) == 0):
            image_boundaries = [0, 0]

        # Convert the boundary arrays to C int types.
        c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
        c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)

        # Make sure that the return type is double otherwise the results will be wrong.
        C.euclidean.restype = ctypes.c_double
        # Call the C euclidean() method.
        accuracies[i] = C.euclidean(c_image_boundaries, len(image_boundaries),
                                    c_label_boundaries, len(label_boundaries))

        # if tests == 66:
        #     cv.imwrite(f"skel/{i}_skel.png", np.array(skel))
        #     print(i, accuracies[i])

    return np.mean(accuracies)

accuracies = np.zeros(args.epochs)
val_accuracies = np.zeros(args.epochs)

# Work out the accuracy for all weight files saved at each epoch.
for e in tqdm(range(args.epochs)):
    accuracies[e] = get_accuracy(f"{args.dir}/train", args.train_samples, f"{args.weights}-{e+1:02d}.hdf5")
    val_accuracies[e] = get_accuracy(f"{args.dir}/test", args.val_samples, f"{args.weights}-{e+1:02d}.hdf5")

print(list(accuracies))
print(list(val_accuracies))