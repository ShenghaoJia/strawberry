#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2019 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# 执行命令 python3 model/train.py model/model.h5 -e 10

from __future__ import print_function

import argparse

import argcomplete
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
import numpy as np
import tensorflow as tf

def create_parser():
    # create the top-level parser
    # 创建参数解析器, 包括生成的h5文件的位置，以及迭代轮数
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('h5_file',
                        default="output.h5",
                        nargs=argparse.OPTIONAL,
                        help='Output - Trained model in h5 format')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=128,
                        help='training batch size')
    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=3,
                        help='training epochs')
    parser.add_argument('-B', '--batch_norm',
                        action='store_true',
                        help='carry out batch normalization')
    return parser

def train(args):
    # 处理批大小
    batch_size = args.batch_size
    # 类别数
    num_classes = 3
    # 迭代轮数
    epochs = args.epochs

    # input image dimensions
    img_rows, img_cols = 224, 224
    channels = 3

    # the data, split between train and test sets
    data = np.load('./imageData.npz')
    (x_train, y_train, x_test, y_test) = (data['x_train'], data['y_train'], data['x_test'], data['y_test'])

    # # K为keras的地层库，默认为tensorflow
    # # K.image_data_format() 返回默认的图片格式，包括'channels_first' 和 'channels_last'，即通道是第一维还是最后一维
    # # 这一步是根据图片格式对图片再进行一次规格化，好像没有必要，因为本身minist就是单通道图片
    # if K.image_data_format() == 'channels_first':
    #     # x_train.shape[0] 指的是图片的数量，即最后把x_train和x_test规格化为(数量，一维的通道，行，列)的四维数组格式
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     # 这里是把x_train和x_test规格化为(数量，行，列，一维的通道)的四维数组格式
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 这里好像是把图片像素规格化到 (-1,1) 的区间内，tensorflow给出的官方代码好像是给到(0,1)
    x_train = (x_train / 128) - 1
    x_test = (x_test / 128) - 1

    # 打印图片格式和样例个数
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    # 是否要添加批标准化，即把输出结果再次压缩到一个小区间内，但是如果用relu作为激活函数是不是这一步就不太需要?
    if args.batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 数据拉平
    model.add(Flatten())
    # 通过全连接层
    model.add(Dense(num_classes))
    if args.batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # 模型被存储在h5文件中
    model.save(args.h5_file)

def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
