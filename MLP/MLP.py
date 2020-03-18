from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import time
from sklearn.utils import shuffle

EPOCHS = 10
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 784))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

# Hyperparameters
mu = 0
sigma = 0.1

with tf.name_scope("MLP"):

    with tf.name_scope("Layer1"):
        with tf.name_scope('weights'):
            fc1_w = tf.Variable(tf.truncated_normal(shape = (784,512), mean = mu, stddev = sigma))
        with tf.name_scope('biases'):
            fc1_b = tf.Variable(tf.zeros(512))
        fc1 = tf.matmul(x,fc1_w) + fc1_b
        fc1 = tf.nn.relu(fc1)


    with tf.name_scope("Layer2"):
        with tf.name_scope('weights'):
            fc2_w = tf.Variable(tf.truncated_normal(shape = (512,512), mean = mu , stddev = sigma))
        with tf.name_scope('biases'):
            fc2_b = tf.Variable(tf.zeros(512))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("Layer3"):
        with tf.name_scope('weights'):
            fc3_w = tf.Variable(tf.truncated_normal(shape = (512,10), mean = mu , stddev = sigma))
        with tf.name_scope('biases'):
            fc3_b = tf.Variable(tf.zeros(10))

    logits = tf.matmul(fc2, fc3_w) + fc3_b


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



##read in data

mnist = input_data.read_data_sets("MNIST_data/", reshape=True)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))


#look at shapes of images
print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


#define learning parameters
rate = 0.001

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

RUN_NUMBER =0
CPU = True
session_conf = tf.ConfigProto(
    device_count={'GPU' : 0 if CPU else 1}
)


with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    RUN_NUMBER += 1

    print("Training...\n")
    t0 = time.time()

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, batch_cost, acc = sess.run([fc1_w, fc1_b, fc2_w, fc2_b, fc3_w,fc3_b, loss_operation, training_operation], feed_dict={x: batch_x, y: batch_y})
            

        validation_accuracy = evaluate(X_validation, y_validation)

        print("Step {} ...".format(i+1))
        print("Time since start = {}".format(time.time() - t0))
        print("Validation Accuracy = {:.3f}\n\n".format(validation_accuracy))



np.savetxt('y_test.csv', y_test[0:500])
    
fc1wb = np.vstack((fc1b,fc1w))
fc1wb = np.array(fc1wb,dtype=np.float32)
fc2wb = np.vstack((fc2b,fc2w))
fc2wb = np.array(fc2wb,dtype=np.float32)
fc3wb = np.vstack((fc3b,fc3w))
fc3wb = np.array(fc3wb,dtype=np.float32)

batchtest = X_test[0:500]
batchtest = np.array(batchtest,dtype=np.float32)
batchtest = np.column_stack((np.ones(500),batchtest))
layer1 = np.array((500,512),dtype=np.float32)
layer1 = np.matmul(batchtest, fc1wb,dtype=np.float32)
layer1 = layer1.clip(min=0)
layer1 = np.column_stack((np.ones(500),layer1))
layer2 = np.array((500,512),dtype=np.float32)
layer2 = np.matmul(layer1, fc2wb,dtype=np.float32)
layer2 = layer2.clip(min=0)
layer2 = np.column_stack((np.ones(500),layer2))
layer3 = np.array((500,10),dtype=np.float32)
layer3 = np.matmul(layer2, fc3wb,dtype=np.float32)

batchtest = np.ascontiguousarray(batchtest,dtype=np.float32)
fc1wb = np.ascontiguousarray(fc1wb,dtype=np.float32)
fc2wb = np.ascontiguousarray(fc2wb,dtype=np.float32)
fc3wb = np.ascontiguousarray(fc3wb,dtype=np.float32)

with open('ytest.bin', 'wb') as write_binary:
    write_binary.write(y_test[0:500])

with open('batchtest.bin', 'wb') as write_binary:
    write_binary.write(batchtest)

with open('fc1_wb.bin', 'wb') as write_binary:
    write_binary.write(fc1wb)

with open('fc2_wb.bin', 'wb') as write_binary:
    write_binary.write(fc2wb)

with open('fc3_wb.bin', 'wb') as write_binary:
    write_binary.write(fc3wb)



