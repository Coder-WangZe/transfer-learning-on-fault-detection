import os
import tensorflow as tf
from utils import *
from time import time
import numpy as np


def transfer_inference(input):
    with tf.name_scope(name="transfer_layer1"):
        weights = tf.get_variable(shape=[1024, 128], name="layer1_w")
        bias = tf.get_variable(shape=[128, ], name="layer1_b")
        fc_layer1 = tf.nn.relu(tf.add(tf.matmul(input, weights), bias), name="fc_layer1")
    # with tf.name_scope(name="transfer_layer2"):
    #     weights = tf.get_variable(shape=[128, 64], name="layer2_w")
    #     bias = tf.get_variable(shape=[64, ], name="layer2_b")
    #     fc_layer2 = tf.nn.relu(tf.add(tf.matmul(fc_layer1, weights), bias), name="fc_layer2")
    with tf.name_scope(name="transfer_layer3"):
        weights = tf.get_variable(shape=[128, 4], name="layer3_w")
        bias = tf.get_variable(shape=[4, ], name="layer3_b")
        fc_layer3 = tf.add(tf.matmul(fc_layer1, weights), bias, name="y_pred")
    return fc_layer3


def compute_loss(y_pred, y_true):
    labels = tf.one_hot(y_true, 4)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y_pred))
    return loss


def train(lr, epochs):
    if os.path.exists('path/to/log'):
        import shutil
        shutil.rmtree('path/to/log/')
    time1 = time()
    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph('./model_casy/0.0001_64/model.ckpt-581.meta')
        train_x = g.get_tensor_by_name('input/x:0')
        train_y = g.get_tensor_by_name('input/y:0')

        layer_fc1 = g.get_tensor_by_name('inference/layer_fc1:0')
        feature_layer = tf.stop_gradient(layer_fc1)
        y_pred = transfer_inference(feature_layer)
        loss = compute_loss(y_pred, train_y)
        tf.summary.scalar("loss", loss)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), train_y)
        train_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", train_acc)
        train_image, train_label = read_and_decode("tongji_train_data_rpm500.tfrecords")
        train_batch_image, train_batch_label = get_batch(train_image, train_label, batch_size=64)

        test_image, test_label = read_and_decode("tongji_test_data_rpm500.tfrecords")
        test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)
        merged = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model_casy/0.0001_64/'))
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        writer = tf.summary.FileWriter('path/to/log/transfer_train_log', graph=g)
        writer_test = tf.summary.FileWriter('path/to/log/transfer_testlog', graph=g)
        for i in range(epochs):
            batch_x, batch_y = sess.run([train_batch_image, train_batch_label])
            summary, acc, loss_np, _ = sess.run([merged, train_acc, loss, train_step],
                                                feed_dict={train_x: batch_x, train_y: batch_y})

            test_x, test_y = sess.run([test_batch_image, test_batch_label])
            summary_test, test_loss, y_pred_, test_acc = sess.run([merged, loss, y_pred, train_acc],
                                                                  feed_dict={train_x: test_x, train_y: test_y})

            writer.add_summary(summary, i)
            writer_test.add_summary(summary_test, i)
            if i % 10 == 0:
                print('***************epochs:', i, '*************')
                print('***************train loss:', loss_np)
                print('***************train accruacy:', acc, '*************')
                print("***********test_accuracy:",  test_acc, "*********\n")
            # if i > 80 and i % 10 == 0:
            #     saver.save(sess, './new_transfered_model/model.ckpt', global_step=i+1)
        writer.close()
        coord.request_stop()
        # queue需要关闭，否则报错
        coord.join(threads)
    time2 = round((time() - time1)/60, 2)
    print("cost time: %.2f min" % time2)
    return test_y, np.argmax(y_pred_, 1)


y_true, y_prediction = train(0.0001, epochs=800)
# false_id = []
# for i, pred in enumerate(y_prediction):
#     if pred != y_true[i]:
#         false_id.append(i)
#
# if not false_id:
#     print("All prediction are true ! \n")
#     print(y_true, "\n", y_prediction)
# else:
#     print(y_true, "\n", y_prediction)
#     print("false_id: ", false_id)

