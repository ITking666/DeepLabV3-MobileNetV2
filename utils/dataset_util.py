"""
Utility functions for creating data sets.
Source: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""

import tensorflow as tf
import numpy as np
import cv2
from utils.image_util import preprocess


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_record(raw_record, image_size):
    """
    Parse image and label from a TFRecord
    ----
    Args:
    ----
    :param raw_record: a TFRecord

    :param image_size: [Height, Width, Channels]
    ----
    Return:
    ----
    A tuple consisting of an image tensor and its label tensor
    """

    # Keys to features stored in the TFRecord
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64),
        'image/width':
            tf.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
    }

    # The following lines are used to extract an image and a label from a TFRecord
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    image = tf.to_float(tf.decode_raw(parsed['image/encoded'], tf.uint8))

    image = tf.reshape(image, image_size)

    label = tf.to_int32(tf.decode_raw(parsed['label/encoded'], tf.uint8))

    label = tf.reshape(label, [image_size[0], image_size[1]])

    return image, label


def input_fn_images_labels(data_file, image_size, batch_size=16, num_epochs_to_repeat=1, shuffle=False, buffer_size=128,
                           is_training=False, aug_params=None):
    """
    input_fn in the tf.data input pipeline
    ----
    Args:
    ----
    :param data_file: The file containing the data either a "train" TFRecord file or a "validation" TFRecord file

    :param image_size: [Height, Width, Channels]

    :param batch_size: The number of samples per batch.

    :param num_epochs_to_repeat: The number of epochs to repeat the dataset. Set it to 1, and OutOfRangeError exception will be thrown at the end of each epoch which is used by TFEstimator for example.

    :param shuffle: a boolean to indicate if the data needs to be shuffled

    :param buffer_size: an integer to indicate the size of the buffer. If it equals to the whole dataset size, all of the records will be loaded in memory

    :param is_training: A boolean to indicate whether training is being done or not

    :param aug_params: A dictionary of additional parameters used for augmentation ['aug_delta_brightness', 'aug_delta_contrast']
    ----
    Return:
    ----
    A tuple consisting of an image tensor and its label tensor
    """
    # Create a dataset from the datafile
    dataset = tf.data.Dataset.from_tensor_slices([data_file])
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    # Shuffle if argument passed is True
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Parse the record into an image and its label
    dataset = dataset.map(lambda record: parse_record(record, image_size))

    # Preprocess the data using random brightness, contrast, etc.
    dataset = dataset.map(lambda image, label: preprocess(image, label, is_training, aug_params))

    # Load "buffer_size" records from the disk.
    dataset = dataset.prefetch(buffer_size)

    # Repeat the dataset if train function works for multiple epochs or throw OutOfRangeError exception
    dataset = dataset.repeat(num_epochs_to_repeat)

    # Batch the dataset into portions. The size of each one is equal to batch_size
    dataset = dataset.batch(batch_size)

    # Create an iterator from this dataset to yield [image, label] tuple
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def get_num_records(tf_record_filename):
    """
    Get the number of records stored in a TFRecord file
    ----
    Args:
    ----
    :param tf_record_filename: path to the tfrecord file
    ----
    Return:
    ----
    Number of records (int)
    """
    return np.sum([1 for _ in tf.python_io.tf_record_iterator(tf_record_filename)])


def read_examples_list(path):
    """Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
      path: absolute path to examples list file.

    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]
