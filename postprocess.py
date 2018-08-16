#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'postprocess'
__author__ = 'fangwudi'
__time__ = '18-1-22 18:41'

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

import numpy as np


def postprocess(y_pred, column_num=7):
    """
    model output (pred): out_diff, out_count
    """
    y_pred_diff = y_pred[:column_num]
    y_pred_count = y_pred[column_num:]
    diff = np.round(y_pred_diff).astype(int)
    count = np.round(y_pred_count).astype(int)
    diff_flag = np.any(diff)
    return diff, count, diff_flag

