#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import  Image
import numpy as np
import matplotlib.pyplot as plot

def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)


img = x_train[0]
label = t_train[0]

# img = img.reshape(28, 28)
# show_img(img)

# plot.imshow(img)
# plot.show()