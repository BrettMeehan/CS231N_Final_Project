import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import tensorflow as tf
import tensorflow_hub as hub
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
    module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1', trainable=True)
    module_input_height, module_input_width = hub.get_expected_image_size(module)
    num_classes = params['num_classes']
    channels = params['channels']
    lmda = params['lambda']
    learning_rate = params['learning_rate']
    initializer = tf.variance_scaling_initializer(scale=2.0)
    conv1 = tf.layers.conv2d(features['x'], channels[0], 3, 
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(lmda))
    
    images = tf.image.resize_images(conv1, [module_input_height, 
                                            module_input_width])
    maxes = tf.reduce_max(images, reduction_indices=[0, 1, 2])
    images = images/maxes

    feature_vectors = module(images)

    fc1 = tf.layers.dense(feature_vectors, num_classes, 
                          kernel_initializer=initializer,
                          kernel_regularizer=tf.keras.regularizers.l2(lmda))
    logits = fc1

    head = tf.contrib.estimator.multi_label_head(n_classes=num_classes)
    optimizer = tf.train.AdamOptimizer(learning_rate)
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
    img = img[:, :, :3]
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

device = '/gpu:0'
alpha = 0.00001
batch_size = 16
resize_dim = 128
channels = [3, 64, 64, 64, 64]
num_classes = 17
num_channels = 3 
lmbda = 0.001
params = {'num_classes': num_classes, 'channels':channels, 'lambda':lmbda, 
          'learning_rate': alpha} 
model_dir = 'resnet-v2_RGB_save_model'
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


convnet_classifier.train(input_fn=train_input_fn, steps=50)
val_accuracy = convnet_classifier.evaluate(input_fn=eval_input_fn)
print('Validation: {}'.format(val_accuracy))
