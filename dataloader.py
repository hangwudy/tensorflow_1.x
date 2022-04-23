import os
import re
from glob import glob

import tensorflow as tf
from keras.utils import to_categorical

IMAGE_SIZE = 64
BATCH_SIZE = 128
DATA_DIR = "/home/hangwu/code/ResNet-Tensorflow/tiny-imagenet-200/train/"
VAL_DATA_DIR = "/home/hangwu/code/ResNet-Tensorflow/tiny-imagenet-200/val/images"
VAL_LABEL_TXT = "/home/hangwu/code/ResNet-Tensorflow/tiny-imagenet-200/val/val_annotations.txt"


def normalize(image):
    mean = 112.6983871459961
    std = 70.93690490722656

    image = (image - mean) / std

    return image


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    # image = image / 127.5 - 1
    image = normalize(image)
    return image


def load_data(image_list, label_list):
    image = read_image(image_list)
    return image, label_list


def data_generator(image_list, label_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).shuffle(BATCH_SIZE * 2).repeat()
    return dataset


def get_label_ids(img_list):
    label_list = [re.findall(r"n\d+", im_path)[0] for im_path in img_list]
    label_set = sorted(set(label_list))
    label_map = {}
    for i, label in enumerate(label_set):
        label_map[label] = i
    label_ids = [label_map[label] for label in label_list]
    print("label map:\n{}".format(label_map))
    label_ids = to_categorical(label_ids)
    return label_ids, label_map


def get_dataset():
    val_map = {}
    with open(VAL_LABEL_TXT, "r") as fp:
        for line in fp:
            im_id, label_id, _, _, _, _ = line.split()
            val_map[im_id] = label_id
    train_images = sorted(glob(os.path.join(DATA_DIR, "*", "*", "*.JPEG")))
    val_images = sorted(glob(os.path.join(VAL_DATA_DIR, "*.JPEG")))

    print("get train images: {}".format(len(train_images)))
    print("get val images: {}".format(len(val_images)))

    label_ids, label_map = get_label_ids(train_images)

    val_label_list = [val_map[os.path.basename(im_path)] for im_path in val_images]
    val_label_ids = [label_map[lbl] for lbl in val_label_list]
    val_label_ids = to_categorical(val_label_ids)

    train_dataset = data_generator(train_images, label_ids)
    val_dataset = data_generator(val_images, val_label_ids)
    return train_dataset, val_dataset


def data_test(dataset):
    iterator = dataset.make_one_shot_iterator()
    data, label = iterator.get_next()
    with tf.Session() as sess:
        for i in range(1000000):
            out = sess.run(data)
            print(out[0][0][0])


if __name__ == '__main__':
    t, v = get_dataset()
    data_test(t)
