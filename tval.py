


with open('weather_val_labels.csv', 'r') as in_file:
    total = 0
    zeros = 0
    for line in in_file:
        _, label = line.split(',')
        total += 1
        if label.strip() == '0':
            zeros += 1

print(zeros/float(total))
