#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'metric'
__author__ = 'fangwudi'
__time__ = '18-1-10 14:27'

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


def diff_accuracy(column_num=7):
    def _diff_accuracy(y_true, y_pred):
        y_true_diff = y_true[:, :column_num]
        y_pred_diff = y_pred[:, :column_num]
        acc_all = K.cast(K.equal(y_true_diff, K.round(y_pred_diff)), 'int8')
        acc_batch = K.min(acc_all, axis=-1, keepdims=False)
        acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)
        return acc
    return _diff_accuracy


def count_accuracy(column_num=7):
    def _count_accuracy(y_true, y_pred):
        y_true_count = y_true[:, column_num:]
        y_pred_count = y_pred[:, column_num:]
        acc_all = K.cast(K.equal(y_true_count, K.round(y_pred_count)), 'int8')
        acc_batch = K.min(acc_all, axis=-1, keepdims=False)
        acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)
        return acc
    return _count_accuracy

# def count_accuracy(column_num=7):
#     def _count_accuracy(y_true, y_pred):
#         y_true_diff = y_true[:, :column_num]
#         y_true_count = y_true[:, column_num:]
#         y_pred_count = y_pred[:, column_num:]
#         # mask used for count accuracy
#         mask = y_true_diff * (-1) + 1
#         mask_batch = K.repeat_elements(K.min(mask, axis=1, keepdims=True), column_num, axis=1)
#         acc_all = K.cast(K.equal(y_true_count * mask_batch, K.round(y_pred_count) * mask_batch), 'int8')
#         acc_batch = K.min(acc_all, axis=-1, keepdims=False)
#         acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)
#         return acc
#     return _count_accuracy
