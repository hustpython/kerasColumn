#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'coremodel'
__author__ = 'fangwudi'
__time__ = '18-1-9 16:11'

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
from keras.models import *
from keras.layers import *
from keras.applications import *


# model-s0
def coremodel(basemodel_name, model_image_size=(378, 504), column_num=7):
    """ use conv output 7 column
    :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
    :param model_image_size: canbe changed to 1280*960?
    :param column_num: column numbers
    :return: keras model
    """
    # img input
    input_img = Input((*model_image_size, 3))
    # input for linked_column, left=0/1, right=0/1 for columns
    input_lc = Input((2 * column_num,))
    # build basemodel
    if basemodel_name == 'resnet50':
        basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'xception':
        basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_v3':
        basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
    elif basemodel_name == 'inception_resnet_v2':
        basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        print('basemodel_name not defined！')
        raise NameError
    # begin
    x = GlobalAveragePooling2D()(basemodel.output)
    # output
    out_diff = Dense(column_num, activation='sigmoid', name='diff')(x)
    out_count = Dense(column_num, name='count')(x)
    out = concatenate([out_diff, out_count])
    return Model([input_img, input_lc], out)


# # model-s1
# def coremodel(basemodel_name, model_image_size=(378, 504), column_num=7):
#     """ use conv output 7 column
#     :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
#     :param model_image_size: canbe changed to 1280*960?
#     :param column_num: column numbers
#     :return: keras model
#     """
#     # img input
#     input_img = Input((*model_image_size, 3))
#     # input for linked_column, left=0/1, right=0/1 for columns
#     input_lc = Input((2 * column_num,))
#     # build basemodel
#     if basemodel_name == 'resnet50':
#         basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'xception':
#         basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_v3':
#         basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_resnet_v2':
#         basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
#     else:
#         print('basemodel_name not defined！')
#         raise NameError
#     # begin
#     x = GlobalAveragePooling2D()(basemodel.output)
#     # deal with input_lc
#     x = concatenate([x, input_lc], name='concat_lc')
#     # output
#     out_diff = Dense(column_num, activation='sigmoid', name='diff')(x)
#     out_count = Dense(column_num, name='count')(x)
#     out = concatenate([out_diff, out_count])
#     return Model([input_img, input_lc], out)

# model-c1
# def coremodel(basemodel_name, model_image_size=(378, 504), column_num=7):
#     """ use conv output 7 column
#     :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
#     :param model_image_size: canbe changed to 1280*960?
#     :param column_num: column numbers
#     :return: keras model
#     """
#     # img input
#     input_img = Input((*model_image_size, 3))
#     # input for linked_column, left=0/1, right=0/1 for columns
#     input_lc = Input((2 * column_num,))
#     # build basemodel
#     if basemodel_name == 'resnet50':
#         basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'xception':
#         basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_v3':
#         basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_resnet_v2':
#         basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
#     else:
#         print('basemodel_name not defined！')
#         raise NameError
#     # use conv to out put 7 column
#     x = AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(basemodel.output)
#     x = conv2d_bn(x, 256, 1, 1, padding='valid', strides=(1, 1))
#     x = conv2d_bn(x, 256, 10, 1, padding='valid', strides=(10, 1))
#     x = conv2d_bn(x, 256, 1, 1, padding='valid', strides=(1, 1))
#     x = conv2d_bn(x, 128, 1, 1, padding='valid', strides=(1, 1))
#     x = Flatten()(x)
#     # add dense layer 3 , not sure if useful
#     x = Dense(128, activation='relu')(x)
#     x = concatenate([x, input_lc], name='concat_lc')
#     # add dense layer 3 , not sure if useful
#     x = Dense(128, activation='relu')(x)
#     # output
#     out_diff = Dense(column_num, activation='sigmoid', name='diff')(x)
#     out_count = Dense(column_num, name='count')(x)
#     out = concatenate([out_diff, out_count])
#     return Model([input_img, input_lc], out)


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


# def coremodel(basemodel_name, model_image_size=(300, 300), column_num=7):
#     """
#     :param basemodel_name: string, one of 'resnet50', 'xception', 'inception_v3' or 'inception_resnet_v2'
#     :param model_image_size: canbe changed to 1280*960?
#     :param column_num: column numbers
#     :return: keras model
#     """
#     # img input
#     input_img = Input((*model_image_size, 3))
#     # input for linked_column, left=0/1, right=0/1 for columns
#     input_lc = Input((2 * column_num,))
#     # build basemodel
#     if basemodel_name == 'resnet50':
#         basemodel = ResNet50(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'xception':
#         basemodel = Xception(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_v3':
#         basemodel = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)
#     elif basemodel_name == 'inception_resnet_v2':
#         basemodel = InceptionResNetV2(input_tensor=input_img, weights='imagenet', include_top=False)
#     else:
#         print('basemodel_name not defined！')
#         raise NameError
#     # concat img feature and linked_column input
#     x = GlobalAveragePooling2D()(basemodel.output)
#     x = concatenate([x, input_lc], name='concat_lc')
#     # add dense layer 1 , not sure if useful
#     x = Dense(512, activation='relu', name='after_dense_1')(x)
#     # output
#     out_diff = Dense(column_num, activation='sigmoid', name='diff')(x)
#     out_count = Dense(column_num, name='count')(x)
#     out = concatenate([out_diff, out_count])
#     return Model([input_img, input_lc], out)
