import tensorflow as tf

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode.image(image_string)
    return image_decoded, label



img_names = []
img_labels = []
with open('weather_labels.csv') as input_file:
    for line in input_file:
    (img_name, label) = line.split(',')
        img_names.append(img_name)
        img_labels.append(int(label))
