import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet2D', help='Architecture [unet2D | unet3D | segnet2D]')
parser.add_argument('--image', type=str, default='test.png', help='Image to predict')
parser.add_argument('--size', type=int, default=256, help='Size to reshape the images to when training')
parser.add_argument('--weights', type=str, default='isambard-20.hdf5', help='Weights file to load')
parser.add_argument('--verbose', action='store_true', help='Show TensorFlow startup messages and warnings')
parser.add_argument('--ablated', action='store_true', help='Use ablated architecture')
parser.add_argument('--dir', type=str, default='data/', help='Data directory')
args = parser.parse_args()

if not args.verbose:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    input = input / 255
    input = trans.resize(input, (args.size, args.size))
    input = np.reshape(input, input.shape + (1,))
    input = np.reshape(input, (1,) + input.shape)
    return input

model = models.compile(arch=args.model, pretrained_weights=args.weights, size=args.size, abl=args.ablated)

stride = 56

image = cv.imread(args.image, -1)
image = img_as_ubyte(image)
image = cv.copyMakeBorder(image, 0, 200, 0, 200, cv.BORDER_CONSTANT)

ys = (image.shape[0] - args.size) / stride
xs = (image.shape[1] - args.size) / stride

output = np.zeros(image.shape)

for x in trange(math.floor(xs)):
    for y in range(math.floor(ys)):
        x1 = x * stride
        y1 = y * stride

        input = image[ y1:y1+args.size , x1:x1+args.size ]
        input_r = np.rot90(input, k=1)
        input = reshape(input)
        input_r = reshape(input_r)

        prediction = model.predict(input)
        prediction_r = model.predict(input_r)

        prediction = trans.resize(prediction[0,:,:,0], (args.size, args.size))
        prediction_r = trans.resize(prediction_r[0,:,:,0], (args.size, args.size))
        prediction_r = np.rot90(prediction_r, k=3)

        for x2 in range(40, 216):
            for y2 in range(40, 216):
                output[y1+y2 , x1+x2] = (output[y1+y2 , x1+x2] + prediction[y2, x2]) / 2
                output[y1+y2 , x1+x2] = (output[y1+y2 , x1+x2] + prediction_r[y2, x2]) / 2

output = output[:-200 , :-200]
output = img_as_ubyte(output)

_, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

output[output == 255] = 1

skel = morphology.skeletonize(output)
skel = skel.astype(int) * 255

cv.imwrite("out.png", np.array(skel))