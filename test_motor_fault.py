import os
import  tensorflow as tf
from utils import *


def t():
    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph('./new_transfered_model/model.ckpt-791.meta')
        test_x = g.get_tensor_by_name('input/x:0')
        test_y = g.get_tensor_by_name('input/y:0')

        y_pred = g.get_tensor_by_name('transfer_layer2/y_pred:0')
        correct_pred = tf.equal(tf.argmax(y_pred, 1), test_y)
        test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        test_image, test_label = read_and_decode("tongji_train_data.tfrecords")
        test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)

    with tf.Session(graph=g) as sess:
        writer = tf.summary.FileWriter('test_log/', graph=g)
        saver.restore(sess, './new_transfered_model/model.ckpt-791')
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        x, y = sess.run([test_batch_image, test_batch_label])
        accuracy = sess.run([test_acc], feed_dict={test_x: x, test_y: y})
        print("****acc: ", accuracy)
        writer.close()
        coord.request_stop()
        coord.join(threads)


t()
