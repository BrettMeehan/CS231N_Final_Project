import random
import numpy as np
import cv2
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle, mean_pixel, std_pixel, 
                 cache=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch

        -mean_pixel: mean pixel over all dimensions in training set
        -std_pixel: std deviation of each pixel over all dimensions in training                     set
        """
        assert len(X) == len(y), 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.mean_pixel, self.std_pixel = mean_pixel, std_pixel
        self.cache = cache
        if cache: 
            image_list = self.X
            self.cache_data = np.empty((len(image_list), 128, 128, 4))
            for i in range(len(image_list)):
                img_name = 'train-tif-v2/' + image_list[i]
                img = cv2.imread(img_name, -1)
                img = cv2.resize(img, (128, 128))
                self.cache_data[i] = (img -\
                                 self.mean_pixel)/self.std_pixel
    def __iter__(self):
        N, B = len(self.X), self.batch_size
        self.num_batches = N//int(B)
        self.batch_idx = 0
        if self.shuffle:
            combined = list(zip(self.X, self.y))
            random.shuffle(combined)
            self.X[:], self.y[:] = zip(*combined)
        return self

    def __next__(self):
        if self.batch_idx < self.num_batches:
            start = self.batch_idx*self.batch_size
            end = start + self.batch_size
            self.batch_idx += 1
            #batch_imgs = np.empty((self.batch_size, 256, 256, 4))
            batch_imgs = np.empty((self.batch_size, 128, 128, 4))
            if self.cache:
                batch_imgs = self.cache_data[start:end]
            else:
                image_list = self.X[start:end]
                for i in range(len(image_list)):
                    img_name = 'train-tif-v2/' + image_list[i]
                    img = cv2.imread(img_name, -1)
                    img = cv2.resize(img, (128, 128))
                    batch_imgs[i] = (img -\
                                     self.mean_pixel)/self.std_pixel
            return (batch_imgs, np.asarray(self.y[start:end]))
        else:
            raise StopIteration

#train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
#val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
#test_dset = Dataset(X_test, y_test, batch_size=64)tch_size = 10
'''
img_names = []
img_labels = []
with open('weather_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append(img_name)
        img_labels.append(int(label))

dset = Dataset(img_names, img_labels, batch_size=8, shuffle=True)
i = 0
for x, y in dset:
    print(x[0,0,0,0])
    print(x.shape)
    print(y.shape)
    print()
    i += 1
    if i >= 10:
        break
'''
