import tensorflow as tf
import numpy as np
import cv2
import sys
from Dataset import Dataset

def flatten(x):
    """    
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)
    
    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    N = tf.shape(x)[0]
    return tf.reshape(x, (N, -1))

def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        try:
            num_correct += (y_pred == y_batch).sum()
        except Exception:
            print(y_pred)
            print(y_batch)
            print()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def kaiming_normal(shape):
    if len(shape) == 1:
        fan_in = shape[0]
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)


def convnet(x, params):
    (eps, conv_w1, conv_b1, gamma1, beta1, conv_w2, conv_b2, gamma2, beta2, 
     conv_w3, conv_b3, gamma3, beta3, conv_w4, conv_b4, gamma4, beta4, conv_w5,
     conv_b5, fc_w1, fc_b1, fc_w2, fc_b2) = params
    scores = None
    
    conv1 = tf.nn.relu(tf.nn.conv2d(x, conv_w1, [1, 1, 1, 1], 'SAME') + conv_b1)
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
    batch_mean1, batch_var1 = tf.nn.moments(pool1,[0])
    batchnorm1 = tf.nn.batch_normalization(pool1, batch_mean1, batch_var1, beta1, 
                                           gamma1, eps)
    
    conv2 = tf.nn.relu(tf.nn.conv2d(batchnorm1, conv_w2, [1, 1, 1, 1], 'SAME') + conv_b2)
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
    batch_mean2, batch_var2 = tf.nn.moments(pool2,[0])
    batchnorm2 = tf.nn.batch_normalization(pool2, batch_mean2, batch_var2, beta2, 
                                           gamma2, eps)
    
    conv3 = tf.nn.relu(tf.nn.conv2d(batchnorm2, conv_w3, [1, 1, 1, 1], 'SAME') + conv_b3)
    pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
    batch_mean3, batch_var3 = tf.nn.moments(pool3,[0])
    batchnorm3 = tf.nn.batch_normalization(pool3, batch_mean3, batch_var3, beta3, 
                                           gamma3, eps)
    
    conv4 = tf.nn.relu(tf.nn.conv2d(batchnorm3, conv_w4, [1, 1, 1, 1], 'SAME') + conv_b4)
    pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
    batch_mean4, batch_var4 = tf.nn.moments(pool4,[0])
    batchnorm4 = tf.nn.batch_normalization(pool4, batch_mean4, batch_var4, beta4, 
                                           gamma4, eps)

    
    conv5 = tf.nn.relu(tf.nn.conv2d(batchnorm4, conv_w5, [1, 1, 1, 1], 'SAME') + conv_b5)
    
    conv5 = flatten(conv5)
    h1 = tf.nn.relu(tf.matmul(conv5, fc_w1) + fc_b1)
    scores = tf.matmul(h1, fc_w2) + fc_b2
    return scores

def convnet_init():
    params = None
    ############################################################################
    # TODO: Initialize the parameters of the three-layer network.              #
    ############################################################################
    input_channels = 4
    input_dims = 128
    num_classes = 4
    channel_1, channel_2, channel_3, channel_4, channel_5 = 3, 64, 64, 64, 64
    hidden_1 = 500
    eps = 1e-5
    conv_w1 = tf.Variable(kaiming_normal((3, 3, input_channels, channel_1)))
    conv_b1 = tf.Variable(tf.zeros((channel_1,)))
    gamma1 = tf.Variable(kaiming_normal((channel_1,)))
    beta1 = tf.Variable(tf.zeros((channel_1,)))
    
    conv_w2 = tf.Variable(kaiming_normal((3, 3, channel_1, channel_2)))
    conv_b2 = tf.Variable(tf.zeros((channel_2,)))
    gamma2 = tf.Variable(kaiming_normal((channel_2,)))
    beta2 = tf.Variable(tf.zeros((channel_2,)))
    
    conv_w3 = tf.Variable(kaiming_normal((3, 3, channel_2, channel_3)))
    conv_b3 = tf.Variable(tf.zeros((channel_3,)))
    gamma3 = tf.Variable(kaiming_normal((channel_3,)))
    beta3 = tf.Variable(tf.zeros((channel_3,)))
    
    conv_w4 = tf.Variable(kaiming_normal((3, 3, channel_3, channel_4)))
    conv_b4 = tf.Variable(tf.zeros((channel_4,)))
    gamma4 = tf.Variable(kaiming_normal((channel_4,)))
    beta4 = tf.Variable(tf.zeros((channel_4,)))
    
    conv_w5 = tf.Variable(kaiming_normal((3, 3, channel_4, channel_5)))
    conv_b5 = tf.Variable(tf.zeros((channel_5,)))

    fc_w1 = tf.Variable(kaiming_normal((input_dims * input_dims * channel_5, 
                                        hidden_1)))
    fc_b1 = tf.Variable(tf.zeros((hidden_1,)))
    fc_w2 = tf.Variable(kaiming_normal((hidden_1, num_classes)))
    fc_b2 = tf.Variable(tf.zeros((num_classes,)))
    params = [eps, conv_w1, conv_b1, gamma1, beta1, conv_w2, conv_b2, gamma2, 
              beta2, conv_w3, conv_b3, gamma3, beta3, conv_w4, conv_b4, gamma4, 
              beta4, conv_w5, conv_b5, fc_w1, fc_b1, fc_w2, fc_b2]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return params

def train(model_fn, init_fn, learning_rate, num_epochs):
    """
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')
    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        x = tf.placeholder(tf.float32, [None, 128, 128, 4])
        y = tf.placeholder(tf.int32, [None])
        params = init_fn()           # Initialize the model parameters
        scores = model_fn(x, params) # Forward pass of the model
        #loss = training_step(scores, y, params, learning_rate)
        
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(losses)

        # define our optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss)
    

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1




device = '/cpu:0'
batch_size = 16
mean_pixel = np.load('train_mean_pixel.npy')
std_pixel = np.load('train_std_pixel.npy')

img_names = []
img_labels = []
with open('weather_train_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append(img_name)
        img_labels.append(int(label))

train_dset = Dataset(img_names, img_labels, batch_size=batch_size, shuffle=True,
               mean_pixel=mean_pixel, std_pixel=std_pixel)

img_names = []
img_labels = []
with open('weather_val_labels.csv') as input_file:
    for line in input_file:
        (img_name, label) = line.split(',')
        img_names.append(img_name)
        img_labels.append(int(label))

val_dset = Dataset(img_names, img_labels, batch_size=batch_size*10, shuffle=False,
               mean_pixel=mean_pixel, std_pixel=std_pixel, cache=False)



learning_rate = 5e-8#5e-4
num_epochs = 4
print_every = 1
train(convnet, convnet_init, learning_rate, num_epochs=num_epochs)
