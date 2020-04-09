#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle

def sigmod(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# 求梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        f1 = f(x)

        x[idx] = tmp_val - h
        f2 = f(x)
        grad[idx] = (f2 - f1) / 2 * h
        x[idx] = tmp_val
    return grad

# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# 损失函数 -- 均方误差
def mean_squared_error(y, t, one_hot=True):
    if one_hot == True:
        return 0.5 * np.sum((y-t)**2)
    else:
        one_hot_t = np.zeros_like(y)
        for i in range(len(t)):
            one_hot_t[i][t[i]] = 1
        return 0.5 * np.sum((y-one_hot_t)**2)

# 损失函数 -- 交叉熵
# delta 是为了防止计算结果为无穷大, 导致后续无法计算
def cross_entropy_error(y, t, one_hot=True):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    if one_hot == True:
        return np.sum(np.log(y) + delta) / batch_size
    else:
        # y[[...],t] 不同于y[a,b]  是取第0维索引在[...]中的数组, 取数组中索引为t的元素
        return np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False, flatten=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

# 预测
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'],
    b1, b2, b3 = network['b1'], network['b2'], network['b3'],

    a1 = np.dot(x,W1)
    z1 = sigmod(a1)

    a2 = np.dot(z1,W2)
    z2 = sigmod(a2)

    a3 = np.dot(z2,W3)
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0 # 精确度
batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # print(y_batch.shape) # (100, 10)
    p = np.argmax(y_batch, axis=1)  # axis=1, 以第一维(0为第零维)为轴找最大元素索引 结果为 (100, 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # 如果p[i] = t[i] 积一分

print('Accuract:' + str(float(accuracy_cnt)/len(x)))

