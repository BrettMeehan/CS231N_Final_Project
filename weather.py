import sys
import random
filename = sys.argv[1]
with open(filename, 'r') as label_file,\
     open('weather_' + filename, 'w') as weather_label_file:
    next(label_file)
    for line in label_file:
        (img_name, labels) = line.strip().split(',')
        img_labels = set(labels.split())
        if 'clear' in img_labels:
           label = 0
        elif 'partly_cloudy' in img_labels:
           label = 1
        elif 'cloudy' in img_labels:
           label = 2
        elif 'haze' in img_labels:
           label = 3
        else:
            print('No label for img: {}'.format(img_name))
            #Add noisy label
            label = random.randint(0, 3)
        weather_label_file.write('{}.tif,{}\n'.format(img_name, label))
