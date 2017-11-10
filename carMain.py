'''
Created on Jun 23, 2017

@author: Vincent Malmrose
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import gfile
import collections
import csv
import argparse
import sys

import numpy as np
import tensorflow as tf
from test.test_buffer import numpy_array
import random as rand

# Data sets
CAR_DATA = "/Got rid of absolute path so no one can see my computer's info/car.data.txt"

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
batchSize = 20
countModTest = 20
epochCount = 1000
summaryCount = epochCount

def myLoadCSV(filename, target_dtype, features_dtype, target_column=-1):
    """Load dataset from CSV file without a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        for row in data_file:
            target.append(row.pop(target_column))
            data.append(np.asarray(row, dtype=features_dtype))

        target = np.array(target, dtype=target_dtype)
        data = np.array(data)
    return Dataset(data=data, target=target)

def getNextBatch(data, num, isTest=False):
    testData = data.data
    testTarget = data.target
    retData, targetData = [], []
    for _ in range(num):
        if(isTest):
            index = rand.randint(0, 1600)
        else:
            index = rand.randint(1600, 1727)
        retData.append(testData[index])
        targetData.append(testTarget[index])
    #retData is size 60, shape 10, 6
    #retTarget is size 10, shape 10, blank
    
    return np.array(retData), np.array(targetData)

def initVariableSummary(shape, variableName):
    with tf.name_scope(variableName):
        variable = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        variable_summaries(variable)
        return variable

def variable_summaries(var):
    #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def train():

    # Load datasets.
    #full_set = myLoadCSV(filename=CAR_DATA, target_dtype=np.string_, features_dtype=np.string_)
    
    sess = tf.InteractiveSession()

    full_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=CAR_DATA,
        target_dtype=np.string_,
        features_dtype=np.string_)
    
    with tf.name_scope("input"):
        x = tf.placeholder(tf.string, [None, 6], name="Input")
        x_ = tf.cast(tf.string_to_hash_bucket_fast(x, 4), tf.float32, name="Input_Hashed_and_Casted")
        x_Onehot = tf.one_hot(tf.string_to_hash_bucket_fast(x, 4), 4, 1.0, 0.0, dtype=tf.float32, name="Input_Onehot")
        y_ = tf.placeholder(tf.string, [batchSize], name="Label")
        y_Hashed = tf.string_to_hash_bucket_fast(y_, 4, name="Label_Hashed")
        y_Onehot = tf.one_hot(y_Hashed, 4, 1.0, 0.0, dtype=tf.float32, name="Label_Onehot")
        
    W = initVariableSummary([6, 4], "weight")
    b = initVariableSummary([4], "bias")

    with tf.name_scope("Guesses"):
        y = tf.matmul(x_, W) + b
        yOnehot = tf.one_hot(tf.cast(y, tf.int64), 4, 1.0, 0.0, dtype=tf.float32, name= "Guess_Onehot")
        
    yChosen = y
    tf.summary.histogram('Guessed', yChosen)
    y_Chosen = y_Onehot
    tf.summary.histogram('Labeled', y_Chosen)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_Chosen, logits=yChosen))
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()
    
    for count in range(epochCount):
        #ValueError: Cannot feed value of shape (10,) for Tensor 'Placeholder_1:0', which has shape '(?, 4)'
        #Most likely it is counting the batch size as the number of possible outputs, Unsure how to fix this
        #Should look more into this
        batch_xs, batch_ys = getNextBatch(full_set, batchSize)
        summary, train = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
        if count % countModTest == 0:
            correct_prediction = tf.equal(tf.argmax(yChosen, 1), tf.argmax(y_Chosen, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_batch_xs, test_batch_ys = getNextBatch(full_set, batchSize, True)
            summary, acc = sess.run([merged, accuracy], feed_dict={x: test_batch_xs,
                                                                   y_: test_batch_ys})
            train_writer.add_summary(summary, count)
            print('Accuracy at step %s: %s' % (count, acc))
#         sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#         sess.run([W, b])
    
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(yChosen, 1), tf.argmax(y_Chosen, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
    tf.summary.scalar('accuracy', accuracy)
    
    test_batch_xs, test_batch_ys = getNextBatch(full_set, batchSize, True)

    summary, acc = sess.run([merged, accuracy], feed_dict={x: test_batch_xs,
                                                           y_: test_batch_ys})
    
    test_writer.add_summary(summary, summaryCount)
    print('Accuracy at step %s: %s' % (summaryCount, acc))
#     print(sess.run(accuracy, feed_dict={x: batch_xs,
#                                         y_: batch_ys}))
    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/carTest/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/carTest/logs/car_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)