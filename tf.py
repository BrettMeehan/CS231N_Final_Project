import numpy as np
import cv2

im = cv2.imread('train-tif-v2/train_0.tif', -1)
print(im.dtype)

batch_size = 64

X = np.empty((batch_size, 256, 256, 4))

