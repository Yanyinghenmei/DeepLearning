#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plot

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
#
# print(x_train.shape)
# print(t_train.shape)
#
# train_size = x_train.shape[0]
# batch_size = 10
#
# # 随机索引
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

x = np.array([
    [1,2,3,0],
    [4,5,6,0],
    [7,8,9,0]
])
batch_size = x.shape[0]
print(batch_size)

res = x[np.arange(batch_size), 2]
print(res)

