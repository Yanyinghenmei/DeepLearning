#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

class Convaolution_L:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # 多个过滤去重叠在一起传进来
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 结果第一维为数据个数
        # 过滤器每一次偏移, 取与过滤器对应的立方体展开为一维, 与过滤器相乘时也将过滤器展开为一维并转至
        # col的行数为: FN * ((X*W)得到的特征图的格子数)
        col = im2col(x, FH, FW, self.stride, self.pad)
        print(col.shape)

        # 对过滤器展开并转至
        col_W = self.W.reshape(FN, -1).T # 录波器展开

        # 虽然 W * x 应该是计算内积, 由于降维的原因, 矩阵相乘, W的行与x的列一对一相乘并求和, 相当于求内积
        # 此处 out.shape = (N * 原计算内积结果元素个数, FN)
        out = np.dot(col, col_W) + self.b
        # 此处 out.shape = (N, C, out_h, out_w), 即结果的四个维度分别取计算前的维度的第(0, 3, 1, 2)位
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H , W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展开
        # im2col: 将x中 h*w 横向展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)

        # 最大值
        out = np.max(col, axis=1)
        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out
