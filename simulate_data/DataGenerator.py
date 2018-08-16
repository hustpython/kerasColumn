#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "DataGenerator"
__author__ = 'fangwudi'
__time__ = '17-12-22 19：30'

code is far away from bugs
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
    ┃ ┳┛  ┗┳ ┃
    ┃    ┻   ┃
    ┗━┓    ┏━┛
      ┃    ┗━━━━━┓
      ┃          ┣┓
      ┃          ┏┛
      ┗┓┓┏━━┳┓┏━━┛
       ┃┫┫  ┃┫┫
       ┗┻┛  ┗┻┛
with the god animal protecting

"""
import os
import threading
import random
import numpy as np
from PIL import Image as ImagePIL
from keras.preprocessing.image import img_to_array


def load_img(path, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.
    """
    fp = open(path, 'r')
    img = ImagePIL.open(path)
    fp.close()
    # simulate_data for process png
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


class DataGenerator(object):
    """Generates stitching image compesed of row * column objects
    each column is one type object, each row may or may not has this object.
    5*5 for example:
    generate image like,
    ---------
      b c d e
    a b c   e
    a b   d
      b c d e
    a b c
    ---------
    then, output y should be number of each type of object: (3, 5, 4, 3, 3)
    """

    def __init__(self, directory, batch_size, row=5, column=5, height=60,
                 width=60):
        # height and width is the size of sub image (object)
        # source_generator should have batch_size argument
        self.directory = directory
        self.batch_size = batch_size
        self.row = row
        self.column = column
        self.height = height
        self.width = width
        # next threading
        self.batch_index = 0
        self.lock = threading.Lock()
        # store sub image
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
        self.subimage_num, self.filenames = list_valid_filenames(directory,
                                                                 white_list_formats,
                                                                 False)
        print('Found %d sub data.' % self.subimage_num)
        subimages = []
        for fname in self.filenames:
            img = load_img(os.path.join(directory, fname),
                           target_size=(height, width))
            subimages.append(img)
        self.subimages = subimages
        # bg picture
        self.bg = ImagePIL.new('RGB', (column * width, row * height), (0, 0, 0))

    def produce(self):
        # define this batch's setting for object type and number
        img_group, numbers_group = [], []
        for i in range(self.batch_size):
            img, numbers = self.produce_one()
            img_group.append(img)
            numbers_group.append(numbers)
        # change numbers_group
        numbers_group = list(map(np.array, zip(*numbers_group)))
        return img_group, numbers_group

    def plan_one(self):
        # plan one image
        kinds, numbers, exists = [], [], []
        for _ in range(self.column):
            kinds.append(random.randint(0, self.subimage_num - 1))
            number = random.randint(0, self.row)
            number_cg = to_categorical(number, self.row + 1)
            numbers.append(number_cg)
            exist = [1] * number + [0] * (self.row - number)
            random.shuffle(exist)
            exists.append(exist)
        return kinds, numbers, exists

    def produce_one(self):
        kinds, numbers, exists = self.plan_one()
        img = self.bg.copy()
        for i in range(self.column):
            for j in range(self.row):
                if exists[i][j]:
                    # simulate_data, to generate more train image
                    img.paste(self.subimages[random.randint(0, self.subimage_num - 1)], (i * self.width, j * self.height))
                    # img.paste(self.subimages[kinds[i]], (i*self.width, j*self.height))
        img = img_to_array(img)
        return img, numbers

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # acquire/release the lock when updating self.value
        with self.lock:
            self.batch_index += 1
        img_group, numbers_group = self.produce()
        return np.array(img_group), numbers_group


def list_valid_filenames(directory, white_list_formats, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        samples: number of data
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda tpl: tpl[0])

    samples = 0
    filenames = []
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                filenames.append(fname)
                samples += 1
    return samples, filenames


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
