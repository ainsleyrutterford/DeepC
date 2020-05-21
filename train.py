import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet2D', help='Architecture [unet2D | unet3D | segnet2D]')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--steps', type=int, default=500, help='Number of batches seen per epoch')
parser.add_argument('--size', type=int, default=256, help='Size to reshape the images to when training')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--batch', type=int, default=2, help='Batch size')
parser.add_argument('--loss', type=str, default='binary_cross', help='Loss function [binary_cross | focal | dice]')
parser.add_argument('--verbose', action='store_true', help='Show TensorFlow startup messages and warnings')
parser.add_argument('--dir', type=str, default='data/', help='Data directory')
parser.add_argument('--name', type=str, default='unet', help='Name to be used for TensorBoard logs')
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

data_gen_args = dict(rotation_range=2,
                     width_shift_range=0.02,
                     height_shift_range=0.02,
                     shear_range=2,
                     zoom_range=0.02,
                     brightness_range=[0.9,1.1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

if args.model == 'unet3D':
    train_gen = data.train_generator_3D(args.batch, args.dir + '/train', 'image','label', data_gen_args, 9)
    val_gen = data.train_generator_3D(args.batch, args.dir + '/test', 'image','label', dict(), 9)
else:
    train_gen = data.train_generator(args.batch, args.dir + '/train', 'image','label', data_gen_args, target_size=(args.size, args.size))
    val_gen = data.train_generator(args.batch, args.dir + '/test', 'image','label', dict(), target_size=(args.size, args.size))

time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
tensorboard = TensorBoard(log_dir=f'logs/{args.name}_{time}')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model = models.compile(args.model, args.loss, size=args.size, lr=args.lr)

checkpoint_name = "isambard-{epoch:02d}.hdf5"
model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=False)
model.fit_generator(train_gen, 
                    steps_per_epoch=args.steps,
                    epochs=args.epochs,
                    validation_data=val_gen,
                    validation_steps=33,
                    callbacks=[tensorboard, model_checkpoint, es])

num_tests = 66
if args.model == 'unet3D':
    test_gen = data.test_generator_3D(args.dir + "/test/image", num_image=num_tests)
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result("test", results)
else:
    test_gen = data.test_generator(args.dir + "/test/image", num_image=num_tests, target_size=(args.size, args.size))
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result("test", results)
