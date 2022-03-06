#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
#

# 执行该python文件的命令是 mkdir -p images; python3 model/save_samples.py -d images -n 5

# 该python文件即从minist数据集中每个数字中都拿出一部分数量的图片，放在指定的文件夹中
import argparse
import os
import random
import importlib

import argcomplete
import numpy as np

def create_parser():
    # create the top-level parser
    # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser = argparse.ArgumentParser(prog='save_samples')
    parser.add_argument('-d', '--directory',
                        default="sample_project/images",
                        help='directory to save samples to')
    parser.add_argument('-n', '--number_of_examples',
                        default=1,
                        type=int,
                        help='number of examples per number')
    parser.add_argument('-s', '--set',
                        help='train or test image set',
                        choices=['train', 'test'],
                        default='test')
    return parser

def write_image(directory, num, img_idx, sample):
    image = importlib.import_module('keras.preprocessing.image')
    # axis=-1表示扩展最后一个维度
    # sample = np.expand_dims(sample, axis=-1)
    # print(sample.shape)
    # 按照 数字_图片编号.pgm 的名字保存sample图片
    image.save_img(os.path.join(directory, "{}_{}.pgm".format(img_idx, num)), sample)

def save_samples(per_number, directory, use_set='test'):
    class_num = 3

    data = np.load('./imageData.npz')
    (x_train, y_train, x_test, y_test) = (data['x_train'], data['y_train'], data['x_test'], data['y_test'])
    # use_set 决定(x,y)是训练集还是测试集
    (x, y) = (x_train, y_train) if use_set == 'train' else (x_test, y_test)
    # 意思好像是生成一个大小为10的通 index
    index = {k: [] for k in range(class_num)}
    # enumerate的意思是把一个数组变成(下标,值)的形式，即这里的idx,value
    # 这里y指的是图片的标签，即(idx,value)=测试集中的(图片编号,图片数字)
    for idx, value in enumerate(y):
        # 这里是在用值来索引下标,即把相同数字的图片编号放在一个桶里
        index[value].append(idx)
    random.seed()
    for idx in range(class_num):
        # 拿出数字为 idx 的所有测试集图片编号
        sample_set = index[idx]
        # 每个类别(数字)都随机拿出per_number张图片
        for _ in range(per_number):
            img_idx = random.randrange(0, len(sample_set))
            # print(x[sample_set[img_idx]].shape)
            write_image(directory, idx, sample_set[img_idx], x[sample_set[img_idx]])
            # 确保拿出的图片不重复
            del sample_set[img_idx]

def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # 把三个参数解析出来以后放到该函数中执行，其中number=5, directory=imaegs,set='test'
    save_samples(args.number_of_examples, args.directory, args.set)

if __name__ == "__main__":
    main()

