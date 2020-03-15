import argparse
import models
from data import *
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet2D', help='Architecture to use [unet2D | unet3D | segnet2D]')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
parser.add_argument('--steps', type=int, default=2000, help='Number of steps per epoch')
parser.add_argument('--loss', type=str, default='binary_cross', 
                    help='Loss function to use [binary_cross | focal | dice]')
args = parser.parse_args()

data_gen_args = dict(rotation_range=2,
                     width_shift_range=0.02,
                     height_shift_range=0.02,
                     shear_range=2,
                     zoom_range=0.02,
                     brightness_range=[0.9,1.1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

train_gen = train_generator(2, 'data/train', 'image','label', data_gen_args, save_to_dir=None)

tensorboard = TensorBoard(log_dir=f'logs/{time()}')

model = models.compile(args.model, args.loss)
model_checkpoint = ModelCheckpoint('isambard.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train_gen, 
                    steps_per_epoch=args.steps, 
                    epochs=args.epochs, 
                    callbacks=[tensorboard, model_checkpoint])

num_tests = 75
test_gen = test_generator("data/test", num_image=num_tests)
results = model.predict_generator(test_gen, num_tests, verbose=1)
save_result("data/test", results)
