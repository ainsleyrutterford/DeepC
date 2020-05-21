import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet2D', help='Architecture [unet2D | unet3D | segnet2D]')
parser.add_argument('--size', type=int, default=256, help='Size to reshape the images to when training')
parser.add_argument('--weights', type=str, default='isambard-20.hdf5', help='Weights file to load')
parser.add_argument('--loss', type=str, default='binary_cross', help='Loss function [binary_cross | focal | dice]')
parser.add_argument('--verbose', action='store_true', help='Show TensorFlow startup messages and warnings')
parser.add_argument('--dir', type=str, default='data/', help='Data directory')
args = parser.parse_args()

if not args.verbose:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import models
import data
from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

model = models.compile(args.model, args.loss, size=args.size, pretrained_weights=args.weights)

num_tests = 66
if args.model == 'unet3D':
    test_gen = data.test_generator_3D(args.dir + "/test/image", num_image=num_tests)
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result("test", results)
else:
    test_gen = data.test_generator(args.dir + "/test/image", num_image=num_tests, target_size=(args.size, args.size))
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result("test", results)