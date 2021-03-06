#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'



import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
# from learn02.two_layer_net import TwoLayerNet
from learn03.simple_conv_net import SimpleConvNet
from common.optimizer import *

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False, flatten=False)

#network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)
network = SimpleConvNet()

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_tate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 通过误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新
    optimizer = Adam()
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
