from Dataset import Dataset
import numpy as np

img_names = []
img_labels = []
with open('weather_val_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append(img_name)
        img_labels.append(int(label))

mean_pixel = np.load('train_mean_pixel.npy')
std_pixel = np.load('train_std_pixel.npy')
dset = Dataset(img_names, img_labels, batch_size=8, shuffle=True, 
               mean_pixel=mean_pixel, std_pixel=std_pixel)
i = 0
for x, y in dset:
    print(x[0,0,0,0])
    print(x.shape)
    print(y.shape)
    print()
    i += 1
    if i >= 1000:
        break
