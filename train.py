from model import *
from data import *

data_gen_args = dict(rotation_range=2,
                     width_shift_range=0.02,
                     height_shift_range=0.02,
                     shear_range=2,
                     zoom_range=0.02,
                     brightness_range=[0.9,1.1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2, 'data/train', 'image','label', data_gen_args, save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('isambard.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

num_tests = 6
testGene = testGenerator("data/test", num_image=num_tests)
results = model.predict_generator(testGene, num_tests, verbose=1)
saveResult("data/test", results)
