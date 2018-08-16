#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'loss'
__author__ = 'fangwudi'
__time__ = '18-1-6 14:48'

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
from keras import backend as K


# loss-ave
def myloss(column_num=7, diff_weight=0.5, count_weight=0.5):
    """deal with out_diff and out_count loss in the same time,
    and use loss weights. Add max loss for diff.
    :param column_num:
    :param diff_weight
    :param count_weight
    :return:
    """

    def _myloss(y_true, y_pred):
        y_true_diff = y_true[:, :column_num]
        y_true_count = y_true[:, column_num:]
        y_pred_diff = y_pred[:, :column_num]
        y_pred_count = y_pred[:, column_num:]
        diff_loss = binary_cross_entropy(y_true_diff, y_pred_diff)
        count_loss = mean_squared_error(y_true_count, y_pred_count)
        loss = diff_loss * diff_weight + count_loss * count_weight
        return loss

    return _myloss


# # loss-max
# def myloss(column_num=7, diff_weight=0.5, count_weight=0.5):
#     """deal with out_diff and out_count loss in the same time,
#     and use loss weights. Add max loss for diff.
#     :param column_num:
#     :param diff_weight
#     :param count_weight
#     :return:
#     """
#     def _myloss(y_true, y_pred):
#         y_true_diff  = y_true[:, :column_num]
#         y_true_count = y_true[:, column_num:]
#         y_pred_diff  = y_pred[:, :column_num]
#         y_pred_count = y_pred[:, column_num:]
#         diff_loss  = max_binary_cross_entropy(y_true_diff, y_pred_diff)
#         count_loss = max_mean_squared_error(y_true_count, y_pred_count)
#         loss = diff_loss * diff_weight + count_loss * count_weight
#         return loss
#     return _myloss


def binary_cross_entropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def max_binary_cross_entropy(y_true, y_pred):
    return K.mean(K.max(K.binary_crossentropy(y_true, y_pred), axis=-1))


def max_mean_squared_error(y_true, y_pred):
    return K.mean(K.max(K.square(y_pred - y_true), axis=-1))

# def myloss(column_num=7, diff_weight=0.2, count_weight=0.8):
#     """deal with out_diff and out_count loss in the same time,
#     and use loss weights.
#     :param column_num:
#     :param diff_weight
#     :param count_weight
#     :return:
#     """
#     def _myloss(y_true, y_pred):
#         y_true_diff  = y_true[:, :column_num]
#         y_true_count = y_true[:, column_num:]
#         y_pred_diff  = y_pred[:, :column_num]
#         y_pred_count = y_pred[:, column_num:]
#         diff_loss  = binary_cross_entropy(y_true_diff, y_pred_diff)
#         count_loss = mean_squared_error(y_true_count, y_pred_count)
#         loss = diff_loss * diff_weight + count_loss * count_weight
#         return loss
#     return _myloss


# def myloss(column_num=7, threshold=0.2, coeff_weight=-0.8, diff_weight=0.2, count_weight=0.8):
#     """deal with out_diff and out_count loss in the same time,
#     and use loss weights.
#     :param column_num:
#     :param threshold: below this, loss = 0
#     :param coeff_weight: -0.8 means will rest 0.2 percentege loss
#     :param diff_weight
#     :param count_weight
#     :return:
#     """
#     def _myloss(y_true, y_pred):
#         y_true_diff = y_true[:, :column_num]
#         y_true_count = y_true[:, column_num:]
#         y_pred_diff = y_pred[:, :column_num]
#         y_pred_count = y_pred[:, column_num:]
#         diff_loss = binary_cross_entropy(y_true_diff, y_pred_diff)
#         # coeff used for reduce count loss when diff, maybe can change to mask?
#         coeff = y_true_diff * coeff_weight + 1
#         count_loss = int_regression_loss(threshold)(coeff * y_true_count, coeff * y_pred_count)
#         loss = diff_loss * diff_weight + count_loss * count_weight
#         return loss
#     return _myloss

# def myloss(column_num=7, threshold=0.0025,  diff_weight=0.2, count_weight=0.8):
#     """deal with out_diff and out_count loss in the same time,
#     and use loss weights.
#     :param column_num:
#     :param threshold: below this, loss = 0
#     :param diff_weight
#     :param count_weight
#     :return:
#     """
#     def _myloss(y_true, y_pred):
#         y_true_diff = y_true[:, :column_num]
#         y_true_count = y_true[:, column_num:]
#         y_pred_diff = y_pred[:, :column_num]
#         y_pred_count = y_pred[:, column_num:]
#         diff_loss = binary_cross_entropy(y_true_diff, y_pred_diff)
#         # mask used for count accuracy
#         mask = y_true_diff * (-1) + 1
#         mask_batch = K.repeat_elements(K.min(mask, axis=1, keepdims=True), column_num, axis=1)
#         count_loss = int_regression_loss(threshold)(y_true_count * mask_batch, y_pred_count * mask_batch)
#         loss = diff_loss * diff_weight + count_loss * count_weight
#         return loss
#     return _myloss

# def int_regression_loss(threshold=0.2):
#     """modified mean_squared_error for int regression
#     """
#     def _int_regression_loss(y_true, y_pred):
#         diff = K.maximum(K.abs(y_true - y_pred) - threshold, 0.)
#         return K.mean(K.square(diff), axis=-1)
#     return _int_regression_loss

# def int_regression_loss(threshold=0.0025):
#     """modified mean_squared_error for int regression
#     """
#     def _int_regression_loss(y_true, y_pred):
#         diff = K.maximum(K.square(y_true - y_pred) - threshold, 0.)
#         return K.mean(diff, axis=-1)
#     return _int_regression_loss

# def binary_cross_entropy(y_true, y_pred):
#     return K.mean(K.binary_crossentropy(y_true, y_pred))
