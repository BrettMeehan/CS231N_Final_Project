
unique_labels = set()
with open('labels.csv', 'r') as label_file:
    next(label_file)
    for line in label_file:
        labels = line.strip().split(',')[1]
        unique_labels = unique_labels.union(labels.split())
print(unique_labels)
