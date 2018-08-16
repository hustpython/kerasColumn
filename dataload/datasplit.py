#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'datasplit'
__author__ = 'fangwudi'
__time__ = '18-1-23 20:01'

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
from random import shuffle
import shutil


def main():
    source_dir = '/home/ubuntu/DataSpace/shushu'
    target_dir = '../data'
    # load file
    samples, filepaths = generate_valid_filepaths(source_dir)
    print('total samples: {}'.format(samples))
    # shuffle
    shuffle(filepaths)
    # split
    ratio = 0.9  # train ratio
    splitpoint = int(samples * ratio)
    train_filepaths = filepaths[:splitpoint]
    valid_filepaths = filepaths[splitpoint:]
    # do with train, valid or test
    dothings(source_dir, target_dir, train_filepaths, subpath='train')
    dothings(source_dir, target_dir, valid_filepaths, subpath='valid')


def dothings(source_dir, target_dir, filepaths, subpath='train'):
    sub_dir = os.path.join(target_dir, subpath)
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    os.mkdir(sub_dir)
    symlink_img_json(source_dir, filepaths, sub_dir)


def symlink_img_json(source_dir, filepaths, target_dir):
    """
    :param source_dir: source directory
    :param filepaths: source filepath
    :param target_dir: target directory
    :return:
    """
    for filepath in filepaths:
        relpath = os.path.relpath(filepath, start = source_dir).split('/')
        newfilename = '_'.join(relpath)
        newfilepath = os.path.join(target_dir, newfilename)
        os.symlink(filepath, newfilepath)
        # link json
        sourcejsonpath = os.path.splitext(filepath)[0] + '.json'
        newjsonpath = os.path.splitext(newfilepath)[0] + '.json'
        os.symlink(sourcejsonpath, newjsonpath)


def generate_valid_filepaths(directory, white_list_formats={'png', 'jpg', 'jpeg', 'bmp', 'ppm'}, follow_links=False):
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
    filepaths = []
    for dirpath, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                filepath = os.path.join(dirpath, fname)
                jsonpath = os.path.splitext(filepath)[0] + '.json'
                # check json exist
                if os.path.exists(jsonpath):
                    # add filepath
                    filepaths.append(filepath)
                    samples += 1
    return samples, filepaths


if __name__ == '__main__':
    main()
