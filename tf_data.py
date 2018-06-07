import tensorflow as tf
import cv2
import numpy as np

def _read_py_function(filename, label):
  img = cv2.imread(filename.decode(), -1).astype(np.float32)
  mean_train_pixel = 4684.00444771
  std_train_pixel = 2099.77405513
  normalized_img = (img - mean_train_pixel)/std_train_pixel
  return normalized_img, label
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

def parse_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), -1)
  #print(tf.decode_raw(image_string, tf.uint8))
  #print(cv2.imread(image_string, -1).shape)
  #image_decoded = tf.convert_to_tensor(cv2.imread(image_string, -1))
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_decoded, label#image_resized, label

# A vector of filenames.
filenames = tf.constant(["train-tif-v2/train_0.tif"])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.float32, label.dtype])))
dataset = dataset.map(_resize_function)
iterator = dataset.make_one_shot_iterator()
print('HELLO')
with tf.Session() as sess:
    data = sess.run(iterator.get_next())
    print(data[0].shape)
    print(data[0].dtype)
