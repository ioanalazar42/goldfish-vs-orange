''' Functions to load and process training and test data '''

import numpy as np 
import os

from skimage import io, transform


def _resize_image(image, width, height):
    resized_image = transform.resize(image, [height, width, 3], anti_aliasing=True, mode='constant', preserve_range=True)

    return (resized_image.astype(np.uint8))


def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2: # if grayscale convert to RGB
        image = np.dstack([image, image, image])

    return _resize_image(image, 64, 64)


def _load_images(dir_path):
    files = os.listdir(dir_path)
    images = np.empty([len(files), 64, 64, 3], dtype=np.uint8) # 64x64 RGB pictures

    for i, file in enumerate(files):
        image_path = os.path.join(dir_path, file)
        images[i] = _load_image(image_path)

    return images


def load_training_data():
    goldfish_images = _load_images('data/training/goldfish')
    orange_images = _load_images('data/training/orange')

    training_images = np.concatenate((goldfish_images, orange_images), axis=0)

    # goldfish > class 0 ; orange > class 1
    training_labels = len(goldfish_images) * [0] + len(orange_images) * [1]

    return training_images, np.array(training_labels)


def load_test_data():
    goldfish_images = _load_images('data/test/goldfish')
    orange_images = _load_images('data/test/orange')

    test_images = np.concatenate((goldfish_images, orange_images), axis=0)

    # goldfish > class 0 ; orange > class 1
    test_labels = len(goldfish_images) * [0] + len(orange_images) * [1]

    return test_images, np.array(test_labels)
