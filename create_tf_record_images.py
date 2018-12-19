"""
Create TFRecord from Images

This file is used to convert two directories containing the dataset images and the labels respectively into two TFRecord files (training and validation). This is used to prepare the data as well as to facilitate the training and validation processes. A structure of a dataset should be as follows:
- A directory containing the RGB images.
- A directory containing the segmentation labels.
- A .txt file listing the file names for the training images and labels.
- A .txt file listing the file names for the validation images and labels.
"""

import io
import os
import sys
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import dataset_util
from utils.image_util import LABEL_COLORS
from utils.generic_util import parse_args

args = parse_args()
# Enable only if validation data exists
VALIDATION_EXISTS = args.VALIDATION_EXISTS
# Path to the directory which will store the TFRecord train file
TRAIN_TF_RECORD_NAME = args.TRAIN_TF_RECORD_NAME
# Path to the directory which will store the TFRecord validation file
VAL_TF_RECORD_NAME = args.VAL_TF_RECORD_NAME
# Path to the file containing the training data
TRAIN_DATA_LIST_NAME = args.TRAIN_DATA_LIST_NAME
# Path to the file containing the validation data
VAL_DATA_LIST_NAME = args.VAL_DATA_LIST_NAME
# Path to the directory containing the training images
IMAGE_DATA_DIR = args.IMAGE_DATA_DIR
# Path to the directory containing the training labels
LABEL_DATA_DIR = args.LABEL_DATA_DIR
# Resize Image Height and Width
OUTPUT_HEIGHT = args.OUTPUT_HEIGHT
OUTPUT_WIDTH = args.OUTPUT_WIDTH
# Color coded or label coded. If color coded, then an RGB image with colors will be read. Else, a label coded will be read with labels [0,1,2, etc.]
LABEL_COLOR_CODED = args.LABEL_COLOR_CODED
# This is the boundaries label that should be ignored. If equals -1, no labels are ignored.
IGNORE_LABEL = args.IGNORE_LABEL


def load_image(filename, output_size, is_label=False):
    """
    Read an image from filename, resize it with an interpolation. If the image is a label image, then mapping from colors to labels is done according to a color palette of 0-255 compatible with PASCAL VOC.
    ----
    Args:
    ----
    :param filename: a string representing the file name

    :param output_size: a tuple [height, width]

    :param is_label: a boolean to specify whether the input image is a label image or not
    ----
    Return:
    ----
    Image after the above operations performed
    """
    img = Image.open(filename)
    if is_label:
        if LABEL_COLOR_CODED:
            # Mapping from RGB image to a label image with labels [0, 1, 2, 3, etc.]
            palette = LABEL_COLORS
            img = img.convert('RGB')
            img = np.array(img)

            label_img = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
            unique_colors = np.unique(np.reshape(img, [-1, 3]), axis=0).tolist()
            for color in unique_colors:
                label = palette.index(color)
                if label != IGNORE_LABEL:
                    match_idx = np.argwhere((img == color).all(axis=-1))
                    label_img[match_idx[:, 0], match_idx[:, 1]] = label
        else:
            label_img = np.array(img)
            label_img[label_img == IGNORE_LABEL] = 0
        label_img = cv2.resize(label_img, (output_size[1], output_size[0]), interpolation=cv2.INTER_NEAREST)
        return label_img
    img = np.array(img)
    img = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_CUBIC)
    return img


def dict_to_tf_example(image_path,label_path):
    """Convert image and label to tf.Example proto.
    ----
    Args:
    ----
      image_path: Path to a single image.

      label_path: Path to its corresponding segmentation label.
    ----
    Returns:
    ----
      example: The converted tf.Example.

    Raises:
      ValueError: if the size of image does not match with that of label.
    """
    # Reading an image
    image = load_image(image_path, (OUTPUT_HEIGHT, OUTPUT_WIDTH)).astype(np.uint8)

    label = load_image(label_path, (OUTPUT_HEIGHT, OUTPUT_WIDTH), is_label=True)

    if image.shape[0:2] != label.shape[0:2]:
        raise ValueError('The size of image does not match with that of label.')

    # Create the TFRecord example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image.shape[0]),
        'image/width': dataset_util.int64_feature(image.shape[1]),
        'image/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(image.tostring())),
        'label/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(label.tostring())),
    }))
    return example


def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     examples):
    """Creates a TFRecord file from examples.
    ----
    Args:
    ----
      output_filename: Path to where output file is saved.

      image_dir: Directory where image files are stored.

      label_dir: Directory where label files are stored.

      examples: Examples to parse and save to tf record.

    """
    # Create a TFRecordWriter
    writer = tf.python_io.TFRecordWriter(output_filename)
    # Define supported image extensions
    IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
    # Start reading the images one by one and iterate in this loop
    for idx, example in tqdm(enumerate(examples)):
        found = False
        image_path, label_path = "", ""
        # The following conditions are done to support different extensions with _L and without at the same time.
        for extension in IMAGE_EXTENSIONS:
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, example + '.' + extension)
            if not os.path.exists(label_path):
                label_path = os.path.join(label_dir, example + '_L.' + extension)
            if not os.path.exists(label_path):
                label_path = os.path.join(label_dir, example + '.' + extension)
            # Break when everything is correct!
            if os.path.exists(image_path) and os.path.exists(label_path):
                found = True
                break
        # Try to create the TFRecord example. If it can't be done, ignore the example.
        try:
            if found:
                tf_example = dict_to_tf_example(image_path, label_path)
                writer.write(tf_example.SerializeToString())
        except ValueError:
            found = False

        if not found:
            print('Could not find {} or it is invalid, ignoring example.\n'.format(example), file=sys.stderr)

    # A writer should be closed after writing
    writer.close()


def main():
    print("Processing the dataset...\n")

    if not os.path.isdir(IMAGE_DATA_DIR):
        raise ValueError("Images directory doesn't exist or there is an error in reading it.")

    if not os.path.isdir(LABEL_DATA_DIR):
        raise ValueError("Segmentation labels directory doesn't exist or there is an error in reading it.")

    # Read training and validation images list, usually .txt file.
    train_examples = dataset_util.read_examples_list(TRAIN_DATA_LIST_NAME)
    val_examples = dataset_util.read_examples_list(VAL_DATA_LIST_NAME)

    # Run the create tf record method for the training data.
    print("Processing the training data...")
    create_tf_record(TRAIN_TF_RECORD_NAME, IMAGE_DATA_DIR, LABEL_DATA_DIR, train_examples)
    print("DONE!\n")

    if VALIDATION_EXISTS:
        # Run the create tf record method for the validation data.
        print("Processing the validation data...\n")
        create_tf_record(VAL_TF_RECORD_NAME, IMAGE_DATA_DIR, LABEL_DATA_DIR, val_examples)
        print("DONE!\n")


if __name__ == '__main__':
    # Run the main program
    main()
