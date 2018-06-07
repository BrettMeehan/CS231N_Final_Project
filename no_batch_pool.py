import tensorflow as tf
import numpy as np
import cv2
import sys
from Dataset import Dataset

class ConvNet(tf.keras.Model):
    def __init__(self, channels, num_classes, lmda):
        super().__init__()        
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv1 = tf.layers.Conv2D(channels[0], 3, activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(lmda))
        self.downsample1 = tf.layers.Conv2D(channels[0], 3, strides=2,
                              kernel_initializer=initializer)
        self.conv2 = tf.layers.Conv2D(channels[1], 3, activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(lmda))
        self.downsample2 = tf.layers.Conv2D(channels[1], 3, strides=2,
                                   kernel_initializer=initializer)
        self.fc1 = tf.layers.Dense(num_classes, kernel_initializer=initializer)
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.downsample1(x)
        x = self.conv2(x)
        x = self.downsample2(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        return x


def test():
    tf.reset_default_graph()
    device = '/cpu:0'
    channels = [3, 64, 64, 64, 64]
    num_classes = 10
    lmbda = 0.001

    model = ConvNet(channels, num_classes, lmbda)
    with tf.device(device):
        x = tf.zeros((64, 28, 28, 4))
        scores = model(x)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)
       

def train_input_fn():
     
