#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "my_image"
__author__ = 'fangwudi'
__email__ = 'fangwudi@foxmail.com'
__time__ = '17-10-24 上午10:30'
__abstract__ = 'my_image, '

       code is far away from bugs 
             ┏┓   ┏┓
            ┏┛┻━━━┛┻━┓
            ┃   ━    ┃
            ┃ ┳┛  ┗┳ ┃
            ┃    ┻   ┃
            ┗━┓    ┏━┛
              ┃    ┗━━━━━┓
              ┃ 神兽保佑  ┣┓
              ┃ 永无BUG!  ┏┛
              ┗┓┓┏━━┳┓┏━━┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
     with the god animal protecting
     
"""
from keras.preprocessing.image import *
import numpy as np
import json


class MyImageDataGenerator(ImageDataGenerator):
    def myflow_from_directory(self, directory,
                              target_size=(256, 256), color_mode='rgb',
                              batch_size=32, shuffle=True, seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='png',
                              follow_links=False,
                              interpolation='nearest'):
        return MyDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            interpolation=interpolation,
            # self define
            column_num=7)


class MyDirectoryIterator(Iterator):
    """Iterator capable of reading images and annotation json from a directory on disk.

    # Arguments
        directory: Path to the directory to read images and annotation json from.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest',
                 # self define
                 column_num=7):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        # self define
        self.column_num = column_num

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        self.samples, self.filenames = list_valid_filenames(directory,
                                                            white_list_formats=white_list_formats,
                                                            follow_links=follow_links)

        print('Found %d images.' % self.samples)
        super(MyDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_img = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_lc = np.zeros((len(index_array), 2 * self.column_num), dtype=K.floatx())
        # out is a batch of (column_num, 2) array (one is out_diff, the other is out_count)
        batch_y = np.zeros((len(index_array), 2 * self.column_num), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            # load img
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_img[i] = x
            # load annotation
            json_name = os.path.join(self.directory , fname.split(".")[0] + ".json")
            json_str = self.read_file(json_name)
            data = json.loads(json_str)
            # distribute annotation
            for z in range(self.column_num):
                # multi colomun attribute should be transformed to 2 attribute
                # 0 -> left 0, right 0
                # 1 -> left 1, right 0
                # 2 -> left 1, right 1
                # 3 -> left 0, right 1
                mc = data[1][z]
                if mc == 1 or mc == 2:
                    batch_lc[i][z * 2] = 1  # left, right, left ,right ... for column_num
                if mc == 2 or mc == 3:
                    batch_lc[i][z * 2 + 1] = 1
            
            batch_y[i, :self.column_num] = np.array(data[2])  # out_diff
            batch_y[i, self.column_num:] = np.array(data[3])  # out_count
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_img[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # bulid batch of input(include imge and link column)
        batch_x = [batch_img, batch_lc]
        return batch_x, batch_y

    @staticmethod
    def read_file(file_name):
        f = open(file_name)
        r = f.read()
        f.close()
        return r

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


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
