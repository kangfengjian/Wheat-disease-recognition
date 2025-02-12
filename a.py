# 引入依赖
from torchvision import datasets
import torch
import os
import skimage
import torchvision.datasets.mnist as mnist
import numpy
from pathlib import Path
root = "./data/Fashion-MNIST"

print(1)
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
 
test_set = (
    mnist.read_image_file(os.path.join(root,'t10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root,'t10k-labels-idx1-ubyte'))
)
print(2)
print("train set:", train_set[0].size())
print("test set:", test_set[0].size())

def convert_to_img(train=True):
    if(train):
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            skimage.io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            skimage.io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()
 
 
convert_to_img(True)
convert_to_img(False)