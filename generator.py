import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy import ndimage


class ImageDataGenerator3D():
    """
    A class designed to mimic the Keras ImageDataGenerator class that can be
    used to perform online augmentation of 3D data as opposed to 2D data.
    """

    def __init__(self, rotation_range=None, width_shift_range=None,
                 height_shift_range=None, shear_range=None, zoom_range=None,
                 brightness_range=None, horizontal_flip=None,
                 vertical_flip=None, fill_mode=None):
        """
        The ImageDataGenerator3D class constructor.

        Args:
            rotation_range: (int) the range of rotations to allow in degrees.
            width_shift_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            height_shift_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            shear_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            zoom_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            brightness_range: (int, int) the range of brightness shifts allowed.
                A value of 1 means no brightness shift, values below 1 mean
                the images are darkened, and values above 1 mean the images
                are brightened.
            horizontal_flip: (bool) whether or not to allow horizontal flips.
            vertical_flip: (bool) whether or not to allow vertical flips.
            fill_mode: (str) the fill mode to use ("nearest" | "mirror")
        """

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
        """
        Randomly rotate the images inside a batch tensor within some
        given range.

        Args:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images to be rotated.
        Returns:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the rotated images.
        """

        np.random.seed(self.seed)
        # Generate a number in degrees to rotate the images by.
        r = np.random.randint(-self.rotation_range, self.rotation_range)
        np.random.seed(self.seed)
        ninety = np.random.choice([True, False])
        # Randomly also rotate by 90 degrees.
        if ninety:
            r += 90
        # Rotate the images using the correct axes since the first axis
        # corresponds to the batch_size.
        batch = ndimage.rotate(batch, r, axes=(2, 3), reshape=False, mode=self.fill_mode)
        if self.seed != None:
            self.seed += 1
        return batch


    def flip(self, batch, axis):
        """
        Randomly flip the images in the specified axis.

        Args:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images to be flipped.
            axis: (int) the axis to flip the images in.
        Returns:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the flipped images.
        """

        np.random.seed(self.seed)
        r = np.random.choice([True, False])
        # Randomly flip the images.
        if r:
            batch = np.flip(batch, axis=axis)
        if self.seed != None and axis != 1:
            self.seed += 1
        return batch


    def brightness(self, batch):
        """
        Randomly shift the images brightness within some given range.

        Args:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images whose brightnesses will
                be shifted.
        Returns:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images whose brightnesses have
                been shifted.
        """

        np.random.seed(self.seed)
        # Generate a number to multiply the images' brightnesses by.
        r = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        # Randomly shift the images' brightnesses.
        batch *= r
        if self.seed != None:
            self.seed += 1
        return batch


    def augment(self, batch):
        """
        Apply the augmentation transformations to a given batch tensor.

        Args:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images to be transformed.
        Returns:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images that have been transformed.
        """

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
        """
        Create a generator that generates augmented batches containing 3D data that
        has been loaded from a specified path.

        Args:
            path: (str) the main data path.
            folder: (str) the subfolder of the main path which the images are
                stored in.
            target_size: (int, int) the size to reshape the images to during augmentation.
            batch_size: (int) the batch size.
            num_frames: (int) the number of 2D images that compose a 3D training sample.
            seed: (int) the seed used to specify the transformations that the
            generator will randomly apply.
        Yields:
            batch: (arr) the (batch_size, z, y, x, channels) tensor
                containing the images that have been transformed.
        """

        self.seed = seed

        # Find all of the .png images in the specified path and sort them into
        # a 2D array of shape (samples, num_frames).
        final_path = os.path.join(path, folder, "*.png")
        file_names = np.array(sorted(glob(final_path)))
        samples = len(file_names) // num_frames
        file_names = file_names[ : samples * num_frames]
        file_names = file_names.reshape((samples, num_frames))

        print(f"Found {samples * num_frames} images ({samples} chunks of {num_frames} images).")
        
        # Create an empty batch tensor.
        batch = np.zeros((batch_size, num_frames, *target_size, 1))
        num_added = 0

        # Yield randomly transformed 3D data indefinitely.
        while True:
            if self.seed == None:
                # If there is no random seed, do not shuffle the images.
                permutation = range(samples)
            else:
                # Otherwise, shuffle the images using the random seed specified.
                np.random.seed(self.seed)
                permutation = np.random.permutation(samples)
            # For each sample found in the directory...
            for i in permutation:
                # Load num_frames images and pack them into the next available
                # slot of the batch tensor.
                for f in range(num_frames):
                    image = cv.imread(file_names[i, f], 0)
                    image = cv.resize(image, target_size)
                    image = np.expand_dims(image, axis=2)
                    batch[num_added, f] = image
                num_added += 1
                # If the batch tensor is now full, augment the batch and yield it.
                if num_added == batch_size:
                    num_added = 0
                    yield self.augment(batch)


class LabelDataGenerator2D():
    """
    A class designed to mimic the Keras ImageDataGenerator class that can be
    used to perform online augmentation of the 2D labels corresponding to the
    3D samples loaded using the ImageDataGenerator3D class.

    This class is very similar to the ImageDataGenerator3D class and they
    should eventually be merged.
    """

    def __init__(self, rotation_range=None, width_shift_range=None,
                 height_shift_range=None, shear_range=None, zoom_range=None,
                 brightness_range=None, horizontal_flip=None,
                 vertical_flip=None, fill_mode=None):
        """
        The ImageDataGenerator3D class constructor.

        Args:
            rotation_range: (int) the range of rotations to allow in degrees.
            width_shift_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            height_shift_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            shear_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            zoom_range: (int) not currently used. Kept for compatibility
                with the Keras ImageDataGenerator class.
            brightness_range: (int, int) the range of brightness shifts allowed.
                A value of 1 means no brightness shift, values below 1 mean
                the images are darkened, and values above 1 mean the images
                are brightened.
            horizontal_flip: (bool) whether or not to allow horizontal flips.
            vertical_flip: (bool) whether or not to allow vertical flips.
            fill_mode: (str) the fill mode to use ("nearest" | "mirror")
        """

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
        """
        Randomly rotate the images labels a batch tensor within some
        given range.

        Args:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels to be rotated.
        Returns:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the rotated labels.
        """

        np.random.seed(self.seed)
        r = np.random.randint(-self.rotation_range, self.rotation_range)
        np.random.seed(self.seed)
        ninety = np.random.choice([True, False])
        if ninety:
            r += 90
        batch = ndimage.rotate(batch, r, axes=(1, 2), reshape=False, mode=self.fill_mode)
        if self.seed != None:
            self.seed += 1
        return batch


    def flip(self, batch, axis):
        """
        Randomly flip the labels in the specified axis.

        Args:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels to be flipped.
            axis: (int) the axis to flip the labels in.
        Returns:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the flipped labels.
        """

        np.random.seed(self.seed)
        r = np.random.choice([True, False])
        if r:
            batch = np.flip(batch, axis=axis)
        if self.seed != None:
            self.seed += 1
        return batch


    def brightness(self, batch):
        """
        Randomly shift the labels brightness within some given range.

        Args:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels whose brightnesses will
                be shifted.
        Returns:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels whose brightnesses have
                been shifted.
        """

        np.random.seed(self.seed)
        r = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        batch *= r
        if self.seed != None:
            self.seed += 1
        return batch


    def augment(self, batch):
        """
        Apply the augmentation transformations to a given batch tensor.

        Args:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels to be transformed.
        Returns:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels that have been transformed.
        """

        if self.rotation_range != None:
            batch = self.rotate(batch)
        if self.brightness_range != None:
            batch = self.brightness(batch)
        if self.horizontal_flip != None:
            batch = self.flip(batch, axis=1)
        if self.vertical_flip != None:
            batch = self.flip(batch, axis=2)
        return batch


    def flow_from_directory(self, path, folder, target_size, batch_size, seed):
        """
        Create a generator that generates augmented batches containing 2D labels that
        correspond to the 3D samples loaded. The 2D labels are loaded from a
        specified path.

        Args:
            path: (str) the main data path.
            folder: (str) the subfolder of the main path which the labels are
                stored in.
            target_size: (int, int) the size to reshape the images to during augmentation.
            batch_size: (int) the batch size.
            seed: (int) the seed used to specify the transformations that the
            generator will randomly apply.
        Yields:
            batch: (arr) the (batch_size, y, x, channels) tensor
                containing the labels that have been transformed.
        """

        self.seed = seed

        # Find all of the .png images in the specified path and sort them.
        final_path = os.path.join(path, folder, "*.png")
        file_names = np.array(sorted(glob(final_path)))
        samples = len(file_names)

        print(f"Found {samples} labels.")
        
        # Create an empty batch tensor.
        batch = np.zeros((batch_size, *target_size, 1))
        num_added = 0

        # Yield randomly transformed 2D labels indefinitely.
        while True:
            if self.seed == None:
                # If there is no random seed, do not shuffle the images.
                permutation = range(samples)
            else:
                # Otherwise, shuffle the images using the random seed specified.
                np.random.seed(self.seed)
                permutation = np.random.permutation(samples)
            # For each label found in the directory...
            for i in permutation:
                # Load the label into the next available slot of the batch tensor.
                image = cv.imread(file_names[i], 0)
                image = cv.resize(image, target_size)
                image = np.expand_dims(image, axis=2)
                batch[num_added] = image
                num_added += 1
                # If the batch tensor is now full, augment the batch and yield it.
                if num_added == batch_size:
                    num_added = 0
                    yield self.augment(batch)