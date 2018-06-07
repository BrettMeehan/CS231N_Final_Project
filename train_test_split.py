import random

with open('labels.csv', 'r') as label_file,\
     open('test_labels.csv', 'w') as test_label_file,\
     open('pre_train_labels.csv', 'w') as train_label_file:
    next(label_file)
    for line in label_file:
        if random.uniform(0, 1) < 0.1:
            test_label_file.write(line)
        else:
            train_label_file.write(line)
