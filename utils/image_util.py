import numpy as np
import tensorflow as tf


def generate_palette(N=256):
    """
    Generate N colors compatible with PASCAL VOC labeling and coloring
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    return cmap.tolist()


LABEL_COLORS = generate_palette(256)


def decode_labels(mask):
    """
    Decode a batch of segmentation masks.
    ----
    Args:
    ----
    :param mask: Argmax output. Note that the input should of shape [NHW].

    :param num_classes: number of classes to predict (including background).
    ----
    Return:
    ----
    RGB colored images ready for display [0-255]
    """
    n, h, w = mask.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    unique = np.unique(mask)
    for i in range(n):
        for val in unique:
            outputs[mask == val] = LABEL_COLORS[val]
    return outputs


def mean_image_addition(image, means):
    """
    Adds the dataset mean from each image channel.
    ----
    Args:
    ----
    :param image: a tensor of shape [height, width, channels]

    :param means: an RGB array. For example: means=[123,128,125]
    ----
    Return:
    ----
    The image after the addition
    """
    num_channels = image.shape[2]
    channel_dim = 2

    if len(means) != num_channels:
        raise ValueError('Mean array should be of the same size as the number of channels')

    channels = tf.split(axis=channel_dim, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=channel_dim, values=channels)


def mean_image_subtraction(image, means):
    """
    Subtracts the dataset mean from each image channel.
    ----
    Args:
    ----
    :param image: a tensor of shape [height, width, channels]

    :param means: an RGB array. For example: means=[123,128,125]
    ----
    Return:
    ----
    The image after the subtraction
    """
    num_channels = image.shape[2]
    channel_dim = 2

    if len(means) != num_channels:
        raise ValueError('Mean array should be of the same size as the number of channels')

    channels = tf.split(axis=channel_dim, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=channel_dim, values=channels)


def preprocess(image, label, is_training, params):
    """
    Preprocess an image of shape [height, width, channels]
    ----
    Args:
    ----
    :param image: the input image

    :param label: the corresponding label

    :param is_training: A boolean to indicate whether training is being done or not

    :param params: A dictionary of additional parameters used for augmentation ['aug_delta_brightness', 'aug_delta_contrast', 'aug_flip_left_right']
    ----
    Return:
    ----
    The preprocessed image and label tuple
    """
    channel_dim = 2
    image_num_channels = image.shape[2]

    if is_training:
        # Randomly change the brightness of the image
        if 'aug_delta_brightness' in params:
            image = tf.image.random_brightness(image, params['aug_delta_brightness'])

        # Randomly flip the image and label horizontally.
        if 'aug_flip_left_right' in params:
            if params['aug_flip_left_right']:
                # First, concatenate the image and the label to do the same operation on both of them at the same time
                image_and_label = tf.concat([image, tf.to_float(tf.expand_dims(label, channel_dim))], axis=channel_dim)
                image_and_label_flipped = tf.image.random_flip_left_right(image_and_label)

                # Separate the image and the label from each other
                image = image_and_label_flipped[:, :, :image_num_channels]
                label = image_and_label_flipped[:, :, image_num_channels:]

                # Remove the extra 1-D
                label = tf.squeeze(label, channel_dim)

        if 'aug_delta_scale_pad_crop' in params:
            # Random scale image and label. Note that image is scaled bilinearly
            # while label is scaled in a nearest neighbor fashion.
            original_image_height = tf.to_float(image.shape[0])
            original_image_width = tf.to_float(image.shape[1])

            scale = tf.random_uniform([], minval=params['aug_delta_scale_pad_crop'][0],
                                      maxval=params['aug_delta_scale_pad_crop'][1],
                                      dtype=tf.float32)
            new_dims = tf.to_int32([original_image_height * scale, original_image_width * scale])
            image = tf.image.resize_images(image, new_dims, method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize_images(tf.expand_dims(label, channel_dim), new_dims,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # Random crop or padding is performed on the result of scaling
            # First, concatenate the image and the label to do the same operation on both of them at the same time
            image_and_label = tf.concat([image, tf.to_float(label)], axis=channel_dim)
            # Here, if the image size after random scaling is large than the original. Then, do nothing.
            # If it's smaller, then pad till the size of the original.
            image_and_label_padded = tf.image.pad_to_bounding_box(image_and_label, 0, 0,
                                                                  tf.maximum(tf.to_int32(original_image_height),
                                                                             tf.shape(image)[0]),
                                                                  tf.maximum(tf.to_int32(original_image_width),
                                                                             tf.shape(image)[1]),
                                                                  )
            # Always take the crops with the original size of the input image
            image_and_label_crop = tf.random_crop(image_and_label_padded,
                                                  [tf.to_int32(original_image_height),
                                                   tf.to_int32(original_image_width), 4])

            # Separate the image and the label from each other
            image = image_and_label_crop[:, :, :image_num_channels]
            label = image_and_label_crop[:, :, image_num_channels:]

            # Remove the extra 1-D
            label = tf.squeeze(label, channel_dim)

    if 'dataset_mean_values' in params:
        image = mean_image_subtraction(image, params['dataset_mean_values'])

    return image, tf.to_int32(label)
