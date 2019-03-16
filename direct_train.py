# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 00:54:22 2018

@author: Ze
"""
import time

time1 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import tensorflow as tf
import os


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [240, 240, 3])

    #    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #    img = tf.reshape(img, [14400])
    label = tf.cast(features['label'], tf.int64)

    return img, label


def get_batch(image, label, batch_size):
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                 capacity=3000, min_after_dequeue=200)

    return images, tf.reshape(label_batch, [batch_size])


class Network(object):
    def __init__(self, learning_rate, batch, classes_num):
        self.learning_rate = learning_rate
        self.batch_size = batch
        self.classes = classes_num
        with tf.variable_scope("weights"):
            self.weights = {
                'conv1': tf.get_variable('conv1', [3, 3, 3, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2': tf.get_variable('conv2', [3, 3, 64, 32],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv3': tf.get_variable('conv3', [3, 3, 32, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv4': tf.get_variable('conv4', [3, 3, 64, 128],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv5': tf.get_variable('conv5', [3, 3, 128, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv6': tf.get_variable('conv6', [3, 3, 64, 128],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv7': tf.get_variable('conv7', [3, 3, 128, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv8': tf.get_variable('conv8', [3, 3, 64, 32],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable('fc1', [15 * 15 * 32, 512],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                'fc2': tf.get_variable('fc2', [512, self.classes],
                                       initializer=tf.contrib.layers.xavier_initializer()),
            }
        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable('conv1', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [32, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4': tf.get_variable('conv4', [128, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv5': tf.get_variable('conv5', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv6': tf.get_variable('conv6', [128, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv7': tf.get_variable('conv7', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv8': tf.get_variable('conv8', [32, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [512, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [self.classes, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
            }

    def inference(self, images):
        with tf.name_scope(name='inference'):
            images = tf.reshape(images, shape=[-1, 240, 240, 3])
            images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2

            conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv1'])
            bn1 = tf.layers.batch_normalization(conv1)
            relu1 = tf.nn.relu(bn1)
            # pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv2 = tf.nn.bias_add(tf.nn.conv2d(bn1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv2'])
            bn2 = tf.layers.batch_normalization(conv2)
            relu2 = tf.nn.relu(bn2)
            # pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            # pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv3 = tf.nn.bias_add(tf.nn.conv2d(relu2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv3'])
            bn3 = tf.layers.batch_normalization(conv3)
            relu3 = tf.nn.relu(bn3)
            resblock1 = tf.add(relu3, relu1)
            pool1 = tf.nn.max_pool(resblock1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv4 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv4'])
            bn4 = tf.layers.batch_normalization(conv4)
            # pool4 = tf.nn.max_pool(bn4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            relu4 = tf.nn.relu(bn4)
            conv5 = tf.nn.bias_add(tf.nn.conv2d(relu4, self.weights['conv5'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv5'], name='layer_conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            relu5 = tf.nn.relu(bn5)
            conv6 = tf.nn.bias_add(tf.nn.conv2d(relu5, self.weights['conv6'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv6'])
            bn6 = tf.layers.batch_normalization(conv6)
            # pool4 = tf.nn.max_pool(bn4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            relu6 = tf.nn.relu(bn6)
            resblock2 = tf.add(relu6, relu4)
            pool2 = tf.nn.max_pool(resblock2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv7 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv7'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv7'], name='layer_conv7')
            bn7 = tf.layers.batch_normalization(conv7)
            relu8 = tf.nn.relu(bn7)
            pool3 = tf.nn.max_pool(relu8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            # resblock3 = tf.add(relu8, relu3)
            conv8 = tf.nn.bias_add(tf.nn.conv2d(pool3, self.weights['conv8'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv8'], name='layer_conv8')
            bn8 = tf.layers.batch_normalization(conv8)
            relu9 = tf.nn.relu(bn8)
            pool4 = tf.nn.max_pool(relu9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            ##################################
            #################################3
            flatten = tf.reshape(pool4, [-1, self.weights['fc1'].get_shape().as_list()[0]])
            # dropout 正则化
            drop1 = tf.nn.dropout(flatten, 0.5)
            fc1 = tf.nn.bias_add(tf.matmul(drop1, self.weights['fc1']), self.biases['fc1'], name='layer_fc1')
            fc_relu1 = tf.nn.relu(fc1)
            fc2 = tf.nn.bias_add(tf.matmul(fc_relu1, self.weights['fc2']), self.biases['fc2'], name='layer_fc2')

        return fc2

    def sorfmax_loss(self, predicts, labels):
        labels = tf.one_hot(labels, self.weights['fc2'].get_shape().as_list()[1])  # as——list得到第二个维度（2分类为2）
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicts, labels=labels))
        self.cost = loss
        return self.cost

    def optimer(self, loss):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   decay_steps=50, decay_rate=0.9,
                                                   staircase=True)
        train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_optimizer


def train(num_epochs):
    tf.reset_default_graph()
    net = Network(0.0001, 64, classes_num=10)
    # get the train data
    # directly train
    train_data_path = "./casy_train_data_10class.tfrecords"
    test_data_path = "./casy_test_data_10class.tfrecords"
    # train_data_path = "./tongji_train_data_rpm1000.tfrecords"
    # test_data_path = "./tongji_test_data_rpm1000.tfrecords"
    # train_data_path = "./tongji_data_4_class_diff_rpm_train.tfrecords"
    # test_data_path = "./tongji_data_4_class_diff_rpm_test.tfrecords"

    train_image, train_label = read_and_decode(train_data_path)
    train_batch_image, train_batch_label = get_batch(train_image, train_label, batch_size=net.batch_size)
    # get the test data
    test_image, test_label = read_and_decode(test_data_path)
    test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=net.batch_size)

    # inference and calculate the accuracy in train data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [net.batch_size, 240, 240, 3], name='x')
        y = tf.placeholder(tf.int64, [net.batch_size, ], name='y')
    inf = net.inference(x)
    loss = net.sorfmax_loss(inf, y)
    train_step = net.optimer(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(inf, 1), tf.int64), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # inference and calculate the accuracy in test data
    with tf.name_scope('test_input'):
        tx = tf.placeholder(tf.float32, [net.batch_size, 240, 240, 3], name="test_x")
        ty = tf.placeholder(tf.int64, [net.batch_size, ], name="test_y")
    test_inf = net.inference(tx)
    test_prediction = tf.equal(tf.cast(tf.argmax(test_inf, 1), tf.int64), ty)
    test_accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

    saver = tf.train.Saver()
    train_costs = []
    train_acc = []
    test_acc = []
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # session =
    with tf.Session(config=config) as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_epochs):
            batch_x, batch_y = session.run([train_batch_image, train_batch_label])
            test_x, test_y = session.run([test_batch_image, test_batch_label])
            t_acc, acc, loss_np, _ = session.run([test_accuracy, accuracy, loss, train_step],
                                                 feed_dict={x: batch_x, y: batch_y, tx: test_x, ty: test_y})
            train_costs.append(loss_np)
            train_acc.append(acc)
            test_acc.append(t_acc)

            if i % 10 == 0:
                print('trainloss:', loss_np)
                print('***************train accruacy:', acc, '*************')
                print('***************test accruacy:', t_acc, '*************')
                print('***************epochs:', i, '/', num_epochs, '**********')
            # if i > 500 and i % 2 == 0:
            #     saver.save(session, 'model' + "\\" + str(net.learning_rate) + "_" + str(net.batch_size) + '/model.ckpt',
            #                global_step=i + 1)

            if acc > 0.9:
                print("reach 90% cost iters : %d ", i)
            if acc > 0.95:
                print("reach 95% cost iters : %d ", i)
            if acc > 0.98:
                print("reach 90% cost iters : %d ", i)
            if acc > 0.99:
                break
        save_path = "./logs/direct_train_casy/"
        np.savetxt(os.path.join(save_path, "train_loss.txt"), train_costs)
        np.savetxt(os.path.join(save_path, "train_acc.txt"), train_acc)
        np.savetxt(os.path.join(save_path, "test_acc.txt"), test_acc)

        coord.request_stop()
        # queue需要关闭，否则报错
        coord.join(threads)
        time2 = time.time()
        time3 = (time2 - time1) / 60
        time3 = round(time3, 2)
        print("cost time：", time3, "min")
        print("lr: ", net.learning_rate)
        print("batch size: ", net.batch_size)
        # print("")
        return train_costs, train_acc, test_acc, time3


# os.chdir('E:\\transfer_learning\\')
os.chdir(os.path.dirname(__file__))
train_loss, train_accs, test_accs, time = train(num_epochs=1000)












