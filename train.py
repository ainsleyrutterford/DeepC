import argparse
import models
import data
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
                    #  brightness_range=[0.9,1.1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

if args.model == 'unet3D':
    train_gen = data.train_generator_3D(2, 'data/train', 'image','label', data_gen_args, 9)
else:
    train_gen = data.train_generator(2, 'data/train', 'image','label', data_gen_args)

tensorboard = TensorBoard(log_dir=f'logs/{time()}')

filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

model = models.compile(args.model, args.loss)
checkpoint_name = "isambard-{epoch:02d}.hdf5"
model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=False)
model.fit_generator(train_gen, 
                    steps_per_epoch=args.steps, 
                    epochs=args.epochs, 
                    callbacks=[tensorboard, model_checkpoint])

num_tests = 324
if args.model == 'unet3D':
    test_gen = data.test_generator_3D("data/test/image", num_image=num_tests)
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result_3D("test", results)
else:
    test_gen = data.test_generator("data/test/image", num_image=num_tests)
    results = model.predict_generator(test_gen, num_tests, verbose=1)
    data.save_result("test", results)
