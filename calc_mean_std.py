import sys
import numpy as np
import cv2

filename = sys.argv[1]

total_sum = 0.0
num_elements = 0
'''
with open(filename, 'r') as dataset:
    for line in dataset:
        (img_name, label) = line.split(',')
        img_name = 'train-tif-v2/' + img_name + '.tif'
        img = cv2.imread(img_name, -1)
        total_sum += img.sum()
        num_elements += 1
        if num_elements % 1000 == 0:
            print(num_elements)
num_elements *= 256*256*4
mean_pixel = total_sum/float(num_elements)
np.save('train_mean_pixel', mean_pixel)
dataset.seek(0)
'''
mean_pixel = np.load('train_mean_pixel.npy')
total_sum = 0.0
num_elements = 0
with open(filename, 'r') as dataset:
    for line in dataset:
        (img_name, label) = line.split(',')
        img_name = 'train-tif-v2/' + img_name + '.tif'
        img = cv2.imread(img_name, -1)
        total_sum += ((img - mean_pixel)**2).sum()

        num_elements += 1
        if num_elements % 1000 == 0:
            print(num_elements)
num_elements *= 256*256*4
var_pixel = total_sum/float(num_elements)
std_pixel = var_pixel**0.5
np.save('train_std_pixel', std_pixel)
