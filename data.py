import glob
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from keras.preprocessing.image import ImageDataGenerator
from generator import ImageDataGenerator3D, LabelDataGenerator2D

def adjust_data(image, mask):
    if np.max(image) > 1:
        image = image / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return image, mask

def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                    image_color_mode="grayscale", mask_color_mode="grayscale",
                    image_save_prefix="image", mask_save_prefix="mask", num_class=2,
                    save_to_dir=None, target_size=(256, 256), seed=1):
    """
    can generate image and mask at the same time use the same seed for image_datagen
    and mask_datagen to ensure the transformation for image and mask is the same if
    you want to visualize the results of generator, set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(train_path,
                                                        classes=[image_folder],
                                                        class_mode=None,
                                                        color_mode=image_color_mode,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        save_to_dir=save_to_dir,
                                                        save_prefix=image_save_prefix,
                                                        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(train_path,
                                                      classes=[mask_folder],
                                                      class_mode=None,
                                                      color_mode=mask_color_mode,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      save_to_dir=save_to_dir,
                                                      save_prefix=mask_save_prefix,
                                                      seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for image, mask in train_generator:
        image, mask = adjust_data(image, mask)
        yield image, mask

def train_generator_3D(batch_size, path, image_folder, mask_folder, aug_dict,
                       num_frames, target_size=(256, 256), seed=1):

    image_datagen = ImageDataGenerator3D(**aug_dict)
    mask_datagen = LabelDataGenerator2D(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        path,
        image_folder,
        target_size,
        batch_size,
        num_frames,
        seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        path,
        mask_folder,
        target_size,
        batch_size,
        num_frames,
        seed
    )

    train_generator = zip(image_generator, mask_generator)
    for image, mask in train_generator:
        image, mask = adjust_data(image, mask)
        yield image, mask

def test_generator(test_path, num_image=30, target_size=(256, 256)):
    image_names = sorted(glob.glob(os.path.join(test_path, "*.png")))
    for i in range(num_image):
        image = io.imread(image_names[i], as_gray=True)
        image = image / 255
        image = trans.resize(image, target_size)
        image = np.reshape(image, (1,) + image.shape + (1,))
        yield image

def test_generator_3D(test_path, num_image=30, target_size=(256, 256), num_frames=9):
    image_datagen = ImageDataGenerator3D()
    image_generator = image_datagen.flow_from_directory(test_path, '', target_size, 1, num_frames, None)

    for i, image_stack in enumerate(image_generator):
        if i < num_image:
            image_stack = image_stack / 255
            yield image_stack
        else:
            return

def save_result(save_path, npyfile):
    for i, item in enumerate(npyfile):
        image = item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(image))

def save_result_3D(save_path, npyfile, num_frames=9):
    for i, item in enumerate(npyfile):
        for f in range(num_frames):
            image = item[f, :, :, 0]
            io.imsave(os.path.join(save_path, f"PREDICTION_{i}_{f}_predict.png"), img_as_ubyte(image))
