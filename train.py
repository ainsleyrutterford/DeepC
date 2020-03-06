from data import *
from model import *
from time import time
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

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

model = unet()
model_checkpoint = ModelCheckpoint('isambard.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train_gen, steps_per_epoch=2000, epochs=5, callbacks=[tensorboard, model_checkpoint])

num_tests = 6
test_gen = test_generator("data/test", num_image=num_tests)
results = model.predict_generator(test_gen, num_tests, verbose=1)
saveResult("data/test", results)
