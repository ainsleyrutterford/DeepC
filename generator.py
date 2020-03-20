import cv2 as cv
import numpy as np
import os
from glob import glob

class ImageDataGenerator3D():

    def __init__(self, rotation_range=None, width_shift_range=None,
                 height_shift_range=None, shear_range=None, zoom_range=None,
                 brightness_range=None, horizontal_flip=None,
                 vertical_flip=None, fill_mode=None):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode

    def augment(self, batch):
        pass

    def flow_from_directory(self, path, image_folder, mask_folder,
                            target_size, batch_size, num_frames):
        # For i in batch_size, load num_frames images and pack
        # into a num_frames x target_size[0] x target_size[y] x 1
        # numpy array. Then augment using augment(). Then yield
        # the batch_size x num_frames x target_size[0] x
        # target_size[y] x 1 numpy array.

        final_path = os.path.join(path, image_folder, "*.png")
        file_names = np.array(sorted(glob(final_path)))
        samples = len(file_names) // num_frames
        file_names = file_names[ : samples * num_frames]
        file_names = file_names.reshape((samples, num_frames))
        
        batch = np.zeros((batch_size, num_frames, *target_size, 1))
        num_added = 0

        while True:
            for i in np.random.permutation(samples):
                for f in range(num_frames):
                    image = cv.imread(file_names[i, f], 0)
                    image = cv.resize(image, target_size)
                    image = np.expand_dims(image, axis=2)
                    batch[num_added, f] = image
                    num_added += 1
                    if num_added == 2:
                        num_added = 0
                        yield batch