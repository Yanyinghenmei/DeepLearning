#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import sys, os
sys.path.append(os.pardir)
from learn02.layer_native import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backwark(dprice)
dapple_price, dorange_price = add_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backwark(dapple_price)
dorange, dorange_num = mul_orange_layer.backwark(dorange_price)

print(price)
print(dapple_num, dapple, dorange_num, dorange, dtax)

