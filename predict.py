from model import *
from data import *

testGene = testGenerator("data/test")
model = unet()
model.load_weights("isambard.hdf5")
results = model.predict_generator(testGene, 6, verbose=1)
saveResult("data/test",results)