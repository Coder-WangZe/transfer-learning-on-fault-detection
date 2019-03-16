import tensorflow as tf


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


def get_test_batch(image, label, batch_size):
    images, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=3000)

    return images, tf.reshape(label_batch, [batch_size])
