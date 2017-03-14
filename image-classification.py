from urllib.request import urlretrieve
from os.path import isfile, isdir
import problem_unittests as tests
import tarfile
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(range(10))

cifar10_dataset_folder_path = 'cifar-10-batches-py'

floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()



import helper
import numpy as np

batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    return x/255

def one_hot_encode(x):
    #try using np.eye as well 
    return encoder.transform(x)


# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


import pickle
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


import tensorflow as tf

def neural_net_image_input(image_shape):
    imw, imh, cc = image_shape
    x = tf.placeholder(tf.float32,
    shape=[None, imw, imh, cc], name='x')
    return x

def neural_net_label_input(n_classes):
    y = tf.placeholder(tf.float32,
    shape=[None, n_classes], name='y')
    return y


def neural_net_keep_prob_input():
    k = tf.placeholder(tf.float32, name='keep_prob')
    return k


tf.reset_default_graph()

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    filter_size_width, filter_size_height = conv_ksize
    color_channels = x_tensor.get_shape().as_list()[-1]
    weight = tf.Variable(tf.truncated_normal(
        [filter_size_width, filter_size_height, color_channels, conv_num_outputs], stddev=0.01))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
   
    # Apply Convolution
    # set the stride for batch and input_channels (i.e. the first and fourth element in the strides array) to be 1.
    a, b = conv_strides
    full_conv_strides = [1, a, b, 1]
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=full_conv_strides, padding='SAME')
    # Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer = tf.nn.relu(conv_layer)
    # Apply Max Pooling
    a, b = pool_strides
    full_pool_strides = [1, a, b, 1]
    a, b = pool_ksize
    full_pool_ksize = [1, a, b, 1]
    conv_layer = tf.nn.max_pool(conv_layer,
        ksize=full_pool_ksize,strides=full_pool_strides,
        padding='SAME')
    return conv_layer 


def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    ns = np.prod(shape[1:])
    flat = tf.reshape(x_tensor, [-1, ns])
    return flat

def fully_conn(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal(
        [x_tensor.get_shape().as_list()[-1] ,num_outputs], stddev=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))
    fc = tf.add(tf.matmul(x_tensor, weight), bias)
    fc = tf.nn.relu(fc)
    return fc

def output(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal(
        [x_tensor.get_shape().as_list()[-1] , num_outputs], stddev=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))
    out = tf.add(tf.matmul(x_tensor, weight), bias)
    return out

def conv_net(x, keep_prob):
    
    conv_num_outputs = 32
    conv_ksize = (5,5)
    conv_strides = (1,1) # tried (3,3) as well
    pool_ksize = (2,2)
    pool_strides = (2,2)
    logits = conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    conv_num_outputs = 64
    conv_ksize = (5,5)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
    logits = conv2d_maxpool(logits, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    conv_num_outputs = 128
    conv_ksize = (5,5)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
    logits = conv2d_maxpool(logits, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    conv_num_outputs = 256
    conv_ksize = (5,5)
    conv_strides = (1,1) # earlier 3,3 
    pool_ksize = (2,2)
    pool_strides = (2,2)
    logits = conv2d_maxpool(logits, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    # gave just 46 % with this layer included at dropout = 0.69 w/0 a third fcc layer
    # increased  validation accuracy with droput at 0.6
    tf.nn.dropout(logits, keep_prob)    
    logits = flatten(logits)
    
    num_outputs = 512
    logits = fully_conn(logits, num_outputs)
    
    num_outputs = 512
    logits = fully_conn(logits, num_outputs)
    
    num_outputs = 512
    logits = fully_conn(logits, num_outputs)
    # not included initialy    
    
    num_outputs = 10
    logits = output(logits, num_outputs)
    
    return logits


# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: keep_probability})
    


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    validation_loss = sess.run(cost, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1})
    validation_accuracy = sess.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1})
    print("validation_loss: {}, validation_accuracy: {}".format(validation_loss, validation_accuracy))

# Tune Parameters
epochs = 25 #trial and error
batch_size = 128
keep_probability = 0.50 #changed from 0.69 works great !

# epochs = 35 #trial and error
# batch_size = 256
# keep_probability = 0.50 #change


print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

#     try one to one convolution just for a try 
#     conv_num_outputs = 128
#     conv_ksize = (1,1)
#     conv_strides = (1,1)
#     pool_ksize = (2,2)
#     pool_strides = (2,2)
#     logits = conv2d_maxpool(logits, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
#horribly bad    
