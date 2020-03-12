import argparse
from models.models import *
from data import *

parser = argparse.ArgumentParser()

parser.add_argument('weights', type=str, help='hdf5 file containing saved weights')
parser.add_argument('num_tests', type=int, help='number of images to predict on')

args = parser.parse_args()

test_gen = test_generator("data/test", num_image=args.num_tests)
model = unet()
model.load_weights(args.weights)
results = model.predict_generator(test_gen, args.num_tests, verbose=1)
saveResult("data/test",results)