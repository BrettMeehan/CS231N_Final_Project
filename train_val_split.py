import random

with open('pre_train_labels.csv', 'r') as pre_train_label_file,\
     open('train_labels.csv', 'w') as train_label_file,\
     open('val_labels.csv', 'w') as val_label_file:
    for line in pre_train_label_file:
        if random.uniform(0, 1) < 0.03:
            val_label_file.write(line)
        else:
            train_label_file.write(line)
