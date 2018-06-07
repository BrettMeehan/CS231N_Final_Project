import tensorflow as tf
import numpy as np
import cv2
import sys
from labels_to_int import * 

def multihot_encoding(integer_list, num_classes):
    integer_list = np.asarray(integer_list)
    multihot = np.zeros(num_classes)
    multihot[integer_list] = 1
    return multihot   

def train_input_fn():
    iterator = train_dataset.make_one_shot_iterator()
    return iterator.get_next()

def eval_input_fn():
    iterator = val_dataset.make_one_shot_iterator()
    return iterator.get_next()

def model_fn(features, labels, mode, params):
    num_classes = params['num_classes']
    channels = params['channels']
    lmda = params['lambda']
    initializer = tf.variance_scaling_initializer(scale=2.0)
    conv1 = tf.layers.conv2d(features['x'], channels[0], 3, 
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(lmda))
    downsample1 = tf.layers.conv2d(conv1, channels[0], 3, strides=2,
                                   kernel_initializer=initializer)
    conv2 = tf.layers.conv2d(downsample1, channels[1], 3, activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(lmda))
    downsample2 = tf.layers.conv2d(conv2, channels[1], 3, strides=2,
                                   kernel_initializer=initializer)
    downsample2 = tf.layers.flatten(downsample2)
    fc1 = tf.layers.dense(downsample2, num_classes, 
                          kernel_initializer=initializer)

    logits=fc1

    head = tf.contrib.estimator.multi_label_head(n_classes=num_classes)
    optimizer = tf.train.AdamOptimizer()
    def _train_op_fn(loss):
        tf.summary.scalar('loss', loss)
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=_train_op_fn)

def _read_py_function(filename, label):
    img = cv2.imread(filename.decode(), -1).astype(np.float32)
    mean_train_pixel = 4684.00444771
    std_train_pixel = 2099.77405513
    normalized_img = (img - mean_train_pixel)/std_train_pixel
    return normalized_img, label

def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, num_channels])
    image_resized = tf.image.resize_images(image_decoded, [resize_dim, 
                                                           resize_dim])
    return image_resized, label 

def _parse(image, label):
    features = {'x': image}
    return features, label

device = '/cpu:0'
batch_size = 16
resize_dim = 128
channels = [3, 64, 64, 64, 64]
num_classes = 17
num_channels = 4
lmbda = 0.001
params = {'num_classes': num_classes, 'channels':channels, 'lambda':lmbda} 
model_dir = 'estimator_save_model_RGBI'
feature_columns = [tf.feature_column.numeric_column('x', shape=[resize_dim, 
                                                                resize_dim,
                                                                num_channels])]
convnet_classifier = tf.estimator.Estimator(model_fn=model_fn, 
                                            model_dir=model_dir, 
                                            params=params)

img_names = []
img_labels = []
with open('../train_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append('../train-tif-v2/' + img_name + '.tif')
        labels = labels_to_int(label.split())
        img_labels.append(multihot_encoding(labels, num_classes))

filenames = tf.constant(img_names)
labels = tf.constant(np.asarray(img_labels))

train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
train_dataset = train_dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.float32, label.dtype])))
train_dataset = train_dataset.map(_resize_function)
train_dataset = train_dataset.map(_parse)
train_dataset = train_dataset.batch(batch_size)


img_names = []
img_labels = []
with open('../val_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append('../train-tif-v2/' + img_name + '.tif')
        labels = labels_to_int(label.split())
        img_labels.append(multihot_encoding(labels, num_classes))

filenames = tf.constant(img_names)
labels = tf.constant(np.asarray(img_labels))

val_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
val_dataset = val_dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.float32, label.dtype])))
val_dataset = val_dataset.map(_resize_function)
val_dataset = val_dataset.map(_parse)
val_dataset = val_dataset.batch(batch_size)


convnet_classifier.train(input_fn=train_input_fn, steps=500)
val_accuracy = convnet_classifier.evaluate(input_fn=eval_input_fn)
print('Validation: {}'.format(val_accuracy))
