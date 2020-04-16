import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy import ndimage

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
        self.seed = None

    def rotate(self, batch):
        np.random.seed(self.seed)
        r = np.random.randint(-self.rotation_range, self.rotation_range)
        np.random.seed(self.seed)
        ninety = np.random.choice([True, False])
        if ninety:
            r += 90
        batch = ndimage.rotate(batch, r, axes=(2, 3), reshape=False, mode='nearest')
        if self.seed != None:
            self.seed += 1
        return batch

    def flip(self, batch, axis):
        np.random.seed(self.seed)
        r = np.random.choice([True, False])
        if r:
            batch = np.flip(batch, axis=axis)
        if self.seed != None and axis != 1:
            self.seed += 1
        return batch

    def brightness(self, batch):
        np.random.seed(self.seed)
        r = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        batch *= r
        if self.seed != None:
            self.seed += 1
        return batch

    def augment(self, batch):
        if self.rotation_range != None:
            batch = self.rotate(batch)
        if self.brightness_range != None:
            batch = self.brightness(batch)
        if self.horizontal_flip != None:
            batch = self.flip(batch, axis=2)
            batch = self.flip(batch, axis=1)
        if self.vertical_flip != None:
            batch = self.flip(batch, axis=3)
        return batch

    def flow_from_directory(self, path, folder, target_size, batch_size, num_frames, seed):
        self.seed = seed
        final_path = os.path.join(path, folder, "*.png")
        file_names = np.array(sorted(glob(final_path)))
        samples = len(file_names) // num_frames
        file_names = file_names[ : samples * num_frames]
        file_names = file_names.reshape((samples, num_frames))
        print(f"Found {samples * num_frames} images ({samples} chunks of {num_frames} images).")
        
        batch = np.zeros((batch_size, num_frames, *target_size, 1))
        num_added = 0

        while True:
            if self.seed == None:
                permutation = range(samples)
            else:
                np.random.seed(self.seed)
                permutation = np.random.permutation(samples)
            for i in permutation:
                for f in range(num_frames):
                    image = cv.imread(file_names[i, f], 0)
                    image = cv.resize(image, target_size)
                    image = np.expand_dims(image, axis=2)
                    batch[num_added, f] = image
                num_added += 1
                if num_added == batch_size:
                    num_added = 0
                    yield self.augment(batch)
                    # yield batch

class LabelDataGenerator2D():

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
        self.seed = None

    def rotate(self, batch):
        np.random.seed(self.seed)
        r = np.random.randint(-self.rotation_range, self.rotation_range)
        np.random.seed(self.seed)
        ninety = np.random.choice([True, False])
        if ninety:
            r += 90
        batch = ndimage.rotate(batch, r, axes=(1, 2), reshape=False, mode='nearest')
        if self.seed != None:
            self.seed += 1
        return batch

    def flip(self, batch, axis):
        np.random.seed(self.seed)
        r = np.random.choice([True, False])
        if r:
            batch = np.flip(batch, axis=axis)
        if self.seed != None:
            self.seed += 1
        return batch

    def brightness(self, batch):
        np.random.seed(self.seed)
        r = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        batch *= r
        if self.seed != None:
            self.seed += 1
        return batch

    def augment(self, batch):
        if self.rotation_range != None:
            batch = self.rotate(batch)
        if self.brightness_range != None:
            batch = self.brightness(batch)
        if self.horizontal_flip != None:
            batch = self.flip(batch, axis=1)
        if self.vertical_flip != None:
            batch = self.flip(batch, axis=2)
        return batch

    def flow_from_directory(self, path, folder, target_size, batch_size, num_frames, seed):
        self.seed = seed
        final_path = os.path.join(path, folder, "*.png")
        file_names = np.array(sorted(glob(final_path)))
        samples = len(file_names)
        print(f"Found {samples} labels.")
        
        batch = np.zeros((batch_size, *target_size, 1))
        num_added = 0

        while True:
            if self.seed == None:
                permutation = range(samples)
            else:
                np.random.seed(self.seed)
                permutation = np.random.permutation(samples)
            for i in permutation:
                image = cv.imread(file_names[i], 0)
                image = cv.resize(image, target_size)
                image = np.expand_dims(image, axis=2)
                batch[num_added] = image
                num_added += 1
                if num_added == batch_size:
                    num_added = 0
                    yield self.augment(batch)
                    # yield batch