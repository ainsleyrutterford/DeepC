import argparse

# All possible arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="unet2D", help="Architecture [unet2D | unet3D | segnet2D]")
parser.add_argument("--image", type=str, default="test.png", help="Image to predict")
parser.add_argument("--size", type=int, default=256, help="Size to reshape the images to when training")
parser.add_argument("--weights", type=str, default="isambard-20.hdf5", help="Weights file to load")
parser.add_argument("--verbose", action="store_true", help="Show TensorFlow startup messages and warnings")
parser.add_argument("--ablated", action="store_true", help="Use ablated architecture")
parser.add_argument("--dir", type=str, default="data/", help="Data directory")
args = parser.parse_args()

# If the --verbose argument is not supplied, suppress all of the TensorFlow startup messages.
if not args.verbose:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import models
import math
import cv2 as cv
import numpy as np
import skimage.transform as trans
from skimage import img_as_ubyte, morphology
from tqdm import trange

def reshape(input):
    """
    Reshape a given 2D image and place in a (1, y, x, 1) tensor to be
    fed into the network.

    Args:
        input: (arr) a 2D array.
    Returns:
        output: (arr) the reshaped image in a (1, y, x, 1) tensor.
    """

    input = input / 255
    input = trans.resize(input, (args.size, args.size))
    input = np.reshape(input, input.shape + (1,))
    input = np.reshape(input, (1,) + input.shape)
    return input

# Compile the model using a compile() method defined in models/__init__.py.
model = models.compile(arch=args.model, weights=args.weights, size=args.size, abl=args.ablated)

stride = 56

# Read the main image whose boundaries are to be predicted.
image = cv.imread(args.image, -1)
image = img_as_ubyte(image)
# Pad the image so that all of the image can be processed in
# args.size x args.size windows.
image = cv.copyMakeBorder(image, 0, 200, 0, 200, cv.BORDER_CONSTANT)

# Work out the number of times the window will slide in the x and y axes.
ys = (image.shape[0] - args.size) / stride
xs = (image.shape[1] - args.size) / stride

# Create an empty output array.
output = np.zeros(image.shape)

# Slide the window over the entire image...
for x in trange(math.floor(xs)):
    for y in range(math.floor(ys)):
        x1 = x * stride
        y1 = y * stride

        # Cut a args.size * args.size image out.
        input = image[ y1:y1+args.size , x1:x1+args.size ]
        # Also rotate a copy of it.
        input_r = np.rot90(input, k=1)

        # Turn the two cropped images into tensors that can be fed to
        # the network.
        input = reshape(input)
        input_r = reshape(input_r)

        # Use the model to predict the boundaries present in the cropped images.
        prediction = model.predict(input)
        prediction_r = model.predict(input_r)

        # Extract the 2D images from the tensors returned by the network.
        prediction = trans.resize(prediction[0,:,:,0], (args.size, args.size))
        prediction_r = trans.resize(prediction_r[0,:,:,0], (args.size, args.size))
        prediction_r = np.rot90(prediction_r, k=3)

        # Construct the output image using only the inner areas of the predictions.
        for x2 in range(40, 216):
            for y2 in range(40, 216):
                output[y1+y2 , x1+x2] = (output[y1+y2 , x1+x2] + prediction[y2, x2]) / 2
                output[y1+y2 , x1+x2] = (output[y1+y2 , x1+x2] + prediction_r[y2, x2]) / 2

# Crop the padding introduced at the start.
output = output[:-200 , :-200]
output = img_as_ubyte(output)

# Threshold the image using Otsu's method.
_, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

# Replace all 255s with 1 in preparation for the skeletonization.
output[output == 255] = 1

# Skeletonize the thresholded predictions.
skel = morphology.skeletonize(output)
skel = skel.astype(int) * 255

# Output the skeletonized prediction.
cv.imwrite("out.png", np.array(skel))