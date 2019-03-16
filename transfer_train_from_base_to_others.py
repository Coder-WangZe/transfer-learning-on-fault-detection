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
    def __init__(self, learning_rate, batch):
        self.learning_rate = learning_rate
        self.batch_size = batch
        with tf.variable_scope("weights"):
            self.weights = {
                'conv1': tf.get_variable('conv1', [3, 3, 3, 16],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv2': tf.get_variable('conv2', [3, 3, 16, 32],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv3': tf.get_variable('conv3', [3, 3, 32, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv4': tf.get_variable('conv4', [3, 3, 64, 128],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv5': tf.get_variable('conv5', [3, 3, 128, 64],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable('fc1', [7 * 7 * 64, 1024],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                'fc2': tf.get_variable('fc2', [1024, 24],
                                       initializer=tf.contrib.layers.xavier_initializer()),
            }
        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable('conv1', [16, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [32, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4': tf.get_variable('conv4', [128, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv5': tf.get_variable('conv5', [64, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [1024, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [24, ],
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
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv2'])
            bn2 = tf.layers.batch_normalization(conv2)
            relu2 = tf.nn.relu(bn2)
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv3'])
            bn3 = tf.layers.batch_normalization(conv3)
            relu3 = tf.nn.relu(bn3)
            pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv4 = tf.nn.bias_add(tf.nn.conv2d(pool3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv4'])
            bn4 = tf.layers.batch_normalization(conv4)
            pool4 = tf.nn.max_pool(bn4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv5 = tf.nn.bias_add(tf.nn.conv2d(pool4, self.weights['conv5'], strides=[1, 2, 2, 1], padding='VALID'),
                                   self.biases['conv5'], name='layer_conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            flatten = tf.reshape(bn5, [-1, self.weights['fc1'].get_shape().as_list()[0]])
            # dropout 正则化
            drop1 = tf.nn.dropout(flatten, 0.5)
            fc1 = tf.nn.bias_add(tf.matmul(drop1, self.weights['fc1']), self.biases['fc1'], name='layer_fc1')
            fc_relu1 = tf.nn.relu(fc1)
            fc2 = tf.nn.bias_add(tf.matmul(fc_relu1, self.weights['fc2']), self.biases['fc2'], name='layer_fc2')

        return fc2

    def inference_fc1(self, images):
        with tf.name_scope(name='inference_fc1'):
            images = tf.reshape(images, shape=[-1, 240, 240, 3])
            images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2

            conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv1'])
            bn1 = tf.layers.batch_normalization(conv1)
            relu1 = tf.nn.relu(bn1)
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv2'])
            bn2 = tf.layers.batch_normalization(conv2)
            relu2 = tf.nn.relu(bn2)
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv3'])
            bn3 = tf.layers.batch_normalization(conv3)
            relu3 = tf.nn.relu(bn3)
            pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv4 = tf.nn.bias_add(tf.nn.conv2d(pool3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'),
                                   self.biases['conv4'])
            bn4 = tf.layers.batch_normalization(conv4)
            pool4 = tf.nn.max_pool(bn4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv5 = tf.nn.bias_add(tf.nn.conv2d(pool4, self.weights['conv5'], strides=[1, 2, 2, 1], padding='VALID'),
                                   self.biases['conv5'], name='layer_conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            flatten = tf.reshape(bn5, [-1, self.weights['fc1'].get_shape().as_list()[0]])
            # dropout 正则化
            drop1 = tf.nn.dropout(flatten, 0.5)
            fc1 = tf.nn.bias_add(tf.matmul(drop1, self.weights['fc1']), self.biases['fc1'], name='layer_fc1')

        return fc1

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


def transfer_inference(input, num_class):
    with tf.name_scope(name="transfer_layer1"):
        weights = tf.get_variable(shape=[1024, 64], name="layer1_w")
        bias = tf.get_variable(shape=[64, ], name="layer1_b")
        fc_layer1 = tf.nn.relu(tf.add(tf.matmul(input, weights), bias), name="fc_layer1")
    # with tf.name_scope(name="transfer_layer2"):
    #     weights = tf.get_variable(shape=[128, 64], name="layer2_w")
    #     bias = tf.get_variable(shape=[64, ], name="layer2_b")
    #     fc_layer2 = tf.nn.relu(tf.add(tf.matmul(fc_layer1, weights), bias), name="fc_layer2")
    with tf.name_scope(name="transfer_layer3"):
        weights = tf.get_variable(shape=[64, num_class], name="layer3_w")
        bias = tf.get_variable(shape=[num_class, ], name="layer3_b")
        fc_layer3 = tf.add(tf.matmul(fc_layer1, weights), bias, name="y_pred")
    return fc_layer3


def compute_loss(y_pred, y_true, num_class):
    labels = tf.one_hot(y_true, num_class)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y_pred))
    return loss


def train(lr=0.0001, epochs=1000):
    tf.reset_default_graph()
    net = Network(0.0001, 64)
    # inference and calculate the accuracy in train data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [net.batch_size, 240, 240, 3], name='x')
        y = tf.placeholder(tf.int64, [net.batch_size, ], name='y')
        layer_fc1 = net.inference_fc1(x)

    feature_layer = tf.stop_gradient(layer_fc1)
    y_pred = transfer_inference(feature_layer, num_class=10)
    loss = compute_loss(y_pred, y, num_class=10)
    tf.summary.scalar("loss", loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), y)
    train_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", train_acc)

    data_path = "casy_test_data_10class.tfrecords"
    # data_path = "tongji_test_data_rpm1000.tfrecords"
    # data_path = "tongji_data_4_class_diff_rpm_test.tfrecords"
    train_image, train_label = read_and_decode(data_path)
    train_batch_image, train_batch_label = get_batch(train_image, train_label, batch_size=64)

    test_image, test_label = read_and_decode(data_path)
    test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)
    merged = tf.summary.merge_all()
    # saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./model/0.0001_64/')
    # 只加载conv部分参数，其他参数初始化
    var = tf.global_variables()
    var_to_restore = [val for val in var if 'conv' in val.name]
    # var_to_restore = [val for val in var if 'conv1' in val.name or 'conv2' in val.name or 'conv3' in val.name]
    saver = tf.train.Saver(var_to_restore)
    var_to_init = [val for val in var if 'conv' not in val.name]
    tf.variables_initializer(var_to_init)
    train_costs = []
    train_accs = []
    test_accs = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('./model/0.0001_64/'))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        writer = tf.summary.FileWriter('path/to/transfer_log/transfer_train_log')
        writer_test = tf.summary.FileWriter('path/to/transfer_log/transfer_testlog')
        for i in range(epochs):
            batch_x, batch_y = sess.run([train_batch_image, train_batch_label])
            summary, acc, loss_np, _ = sess.run([merged, train_acc, loss, train_step],
                                                feed_dict={x: batch_x, y: batch_y})

            test_x, test_y = sess.run([test_batch_image, test_batch_label])
            summary_test, test_loss, y_pred_, test_acc = sess.run([merged, loss, y_pred, train_acc],
                                                                  feed_dict={x: test_x, y: test_y})

            writer.add_summary(summary, i)
            writer_test.add_summary(summary_test, i)
            
            train_costs.append(loss_np)
            train_accs.append(acc)
            test_accs.append(test_acc)
            if i % 10 == 0:
                print('***************epochs:', i, '*************')
                print('***************train loss:', loss_np)
                print('***************train accruacy:', acc, '*************')
                print("***********test_accuracy:", test_acc, "*********\n")

        save_path = "E:\\tf_learning(2)\\logs\\transfer_from_base_to_casy\\"
        np.savetxt(os.path.join(save_path, "train_loss.txt"), train_costs)
        np.savetxt(os.path.join(save_path, "train_acc.txt"), train_accs)
        np.savetxt(os.path.join(save_path, "test_acc.txt"), test_accs)


            # if i > 80 and i % 10 == 0:
            #     saver.save(sess, './new_transfered_model/model.ckpt', global_step=i + 1)
            # if loss_np < 0.01:
            #     break
            # if test_acc > 0.9:
            #     time2 = round((time.time() - time1) / 60, 2)
            #     print("达到 %.3f cost %d iterations and %.2f min" % (test_acc, i, time2))
            # if test_acc > 0.95:
            #     time2 = round((time.time() - time1) / 60, 2)
            #     print("达到 %.3f cost %d iterations and %.2f min" % (test_acc, i, time2))
            # if test_acc > 0.98:
            #     time2 = round((time.time() - time1) / 60, 2)
            #     print("达到 %.3f cost %d iterations and %.2f min" % (test_acc, i, time2))
            #     break

        writer.close()
        coord.request_stop()
        # queue需要关闭，否则报错
        coord.join(threads)
    time2 = round((time.time() - time1) / 60, 2)
    print("cost time: %.2f min" % time2)
    # return test_y, np.argmax(y_pred_, 1)


# os.chdir('E:\\transfer_learning\\')
os.chdir(os.path.dirname(__file__))
train()












