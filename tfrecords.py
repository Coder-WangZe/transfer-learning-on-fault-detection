import tensorflow as tf
from PIL import Image
import os


current_path = "E:\\tf_learning(2)\\base_data\\"
writer = tf.python_io.TFRecordWriter("casy_train_data_10class.tfrecords")

dirs = os.listdir(current_path)
train_classes = [path for path in dirs if (path.startswith("ball") or path.startswith("normal")
                 or path.startswith("inner")
                 or path.startswith("outer")) and not path.endswith("_test")]
test_classes = [path for path in dirs if (path.startswith("ball") or path.startswith("normal")
                 or path.startswith("inner")
                 or path.startswith("outer")) and path.endswith("_test")]
# test_classes = [path for path in dirs if path.endswith("_test")]
#
# train_classes = [path for path in dirs if not path.endswith("_test")]
# train_classes = dirs

for index, name in enumerate(train_classes):
    class_path = current_path + "/"+name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((240, 240))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

'''制作测试数据集'''

writer = tf.python_io.TFRecordWriter("casy_test_data_10class.tfrecords")


for index, name in enumerate(test_classes):
    class_path = current_path + "/"+name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((240, 240))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串



