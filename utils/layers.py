import tensorflow.contrib.slim as slim
import tensorflow as tf


@slim.add_arg_scope
def inverted_residual_block(net, in_filters, out_filters, expansion_factor, stride, kernel_size=(3, 3)):
    """
    Inverted Residual Block specified in MobileNet-V2 paper
    ----
    Args:
    ----
    :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

    :param in_filters: number of input feature map filters

    :param out_filters: number of output feature map filters

    :param expansion_factor: multiplier to increase number of filters (proposed in the paper)

    :param stride: output stride for the depthwise convolution

    :param kernel_size: is the height and width of the convolution kernel.

    ----
    Return:
    ----
    Returns a block that consists of [conv2d, depthwise_conv2d, conv2d] with residual connection if input stride equals 1 or without if input stride equals 2
    """
    if stride not in [1, 2]:
        raise ValueError("Strides should be either 1 or 2")

    res = slim.conv2d(net, in_filters * expansion_factor, kernel_size, stride=1,
                      activation_fn=tf.nn.relu6)

    res = slim.separable_conv2d(res, None, kernel_size, 1, stride=stride, activation_fn=tf.nn.relu6)

    res = slim.conv2d(res, out_filters, kernel_size, stride=1, activation_fn=None)

    if stride == 2:
        return res
    else:
        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if in_filters != out_filters:
            net = slim.conv2d(net, out_filters, stride=1, kernel_size=1, activation_fn=None)
        return tf.add(res, net)


@slim.add_arg_scope
def inverted_residual_block_sequence(net, in_filters, out_filters, num_units: int, expansion_factor=6, initial_stride=2,
                                     kernel_size=3):
    """
    A group of inverted residual blocks
    ----
    Args:
    ----
    :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

    :param in_filters: number of input feature map filters

    :param out_filters: number of output feature map filters

    :param num_units: is the number of blocks in the sequence

    :param expansion_factor: multiplier to increase number of filters (proposed in the paper)

    :param initial_stride: output stride for the depthwise convolution

    :param kernel_size: is the height and width of the convolution kernel.

    ----
    Return:
    ----
    Returns a sequence of blocks that consists of [conv2d, depthwise_conv2d, conv2d] with residual connection if input stride equals 1 or without if input stride equals 2
    """
    net = inverted_residual_block(net, in_filters, out_filters, expansion_factor, initial_stride, kernel_size)

    for i in range(num_units - 1):
        net = inverted_residual_block(net, in_filters, out_filters, expansion_factor, 1, kernel_size)

    return net


@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, out_filters=256, feature_output_stride=8, data_format='NHWC', is_training=True,
                                   scope="atrous_spp"):
    """
    Atrous Spatial Pyramid Pooling as specified in DeepLab papers
    ----
    Args:
    ----
    :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

    :param out_filters: number of output feature map filters

    :param feature_output_stride: is the downsampling factor. If 16, then an image will be downsampled by factor equals 1/16. If 8, then an image will be downsampled by a factor equals 1/8. It determines the rates for atrous convolution. The rates are (6, 12, 18) when the output stride is 16 and doubled when it is 8

    :param is_training: a boolean used for both [batchnorm and dropout] layers which should be true during training and false during testing

    :param scope: is the name of the variable scope of the block
    ----
    Return:
    ----
    The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope(scope):
        # Create an arg scope for the layers conv2d and separable_conv2d
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_params={'is_training': is_training}):
            if feature_output_stride not in [8, 16]:
                raise ValueError('Currently supported downsampling factors [feature_output_stride] are 8 and 16')

            if data_format not in ['NCHW', 'NHWC']:
                raise ValueError('Data format should be either NCHW or NHWC')

            atrous_rates = [6, 12, 18]
            if feature_output_stride == 8:
                atrous_rates = [2 * rate for rate in atrous_rates]

            input_size = net.shape[1:3] if data_format == 'NHWC' else net.shape[2:]
            average_dims = [1, 2] if data_format == 'NHWC' else [2, 3]
            concat_dim = 3 if data_format == 'NHWC' else 1

            # It consists of one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18)
            # when features output stride = 16. Note that the rates are doubled when features output stride = 8.
            conv_1x1 = slim.conv2d(net, out_filters, 1, stride=1)
            conv_3x3_1 = slim.conv2d(net, out_filters, 3, stride=1, rate=atrous_rates[0])
            conv_3x3_2 = slim.conv2d(net, out_filters, 3, stride=1, rate=atrous_rates[1])
            conv_3x3_3 = slim.conv2d(net, out_filters, 3, stride=1, rate=atrous_rates[2])

            # Global average pooling on height and width of the feature maps
            image_level_features = tf.reduce_mean(net, average_dims, keepdims=True, name='global_avg_pooling')

            # 1x1 convolution on the pooled features
            image_level_features = slim.conv2d(image_level_features, out_filters, 1, stride=1)
            # Upsample the features in a bilinear fashion as proposed by DeepLab-V3 paper
            if data_format == 'NCHW':
                # If the data is NCHW, then a transpose should be done because resize_bilinear works on CPU not GPU.
                image_level_features = tf.transpose(image_level_features, [0, 2, 3, 1])
                image_level_features = tf.image.resize_bilinear(image_level_features, input_size, name='upsample')
                # A transpose is used to retrieve back the original shape
                image_level_features = tf.transpose(image_level_features, [0, 3, 1, 2])
            else:
                image_level_features = tf.image.resize_bilinear(image_level_features, input_size, name='upsample')
            # Concatenate all the extracted features across the channels
            net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=concat_dim,
                            name='concat_all_aspp')
            # Final 1x1 convolution to get the output features
            net = slim.conv2d(net, out_filters, 1, stride=1, scope='conv_1x1_concat_output')

            return net


###################################################################################################################
def create_arg_scope(weight_decay=0.0, dropout_keep_prob=1.0, batchnorm=True, batchnorm_decay=0.999,
                     is_training=True, data_format='NHWC'):
    """
    This creates an argument scope to pass the same parameters for the same layers to avoid having duplicate code
    ----
    Args:
    ----
    :param weight_decay: is the L2 regularization strength hyperparameter

    :param dropout_keep_prob: the probability of keeping neurons for dropout layer

    :param batchnorm: a boolean to enable or disable batch normalization

    :param batchnorm_decay: a float number to represent the decay rate for the moving average of batch normalization

    :param is_training: a boolean used for both [batchnorm and dropout] layers which should be true during training and false during testing

    :param data_format: 'NCHW' or 'NHWC'. It's proved that 'NCHW' provides a performance boost for GPUs. However, 'NHWC' is the only one that can be used for CPU computations

    ----
    Return:
    ----
    A scope after construction with the passed arguments
    """
    # L2 Regularization
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    # Glorot initializer is the default initializer
    initializer = tf.glorot_normal_initializer()
    # Batch normalization is used if enabled
    normalizer_fn = slim.batch_norm if batchnorm else None
    normalizer_params = {'is_training': is_training, 'center': True, 'scale': True,
                         'decay': batchnorm_decay} if batchnorm else None

    # Create an arg scope for the layers conv2d and separable_conv2d
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], data_format=data_format,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer, normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):
        # Create am arg scope for batch_norm layer
        with slim.arg_scope([slim.batch_norm], data_format=data_format):
            # Create an arg scope for the dropout layer
            with slim.arg_scope([slim.dropout], data_format=data_format, is_training=is_training,
                                keep_prob=dropout_keep_prob) as sc:
                return sc


def resize_bilinear(net, shape, is_nchw=False):
    if is_nchw:
        # If the data is NCHW, then a transpose should be done because resize_bilinear works on CPU not GPU.
        net = tf.transpose(net, [0, 2, 3, 1])
        net = tf.image.resize_bilinear(net, shape)
        # A transpose is used to retrieve back the original shape
        net = tf.transpose(net, [0, 3, 1, 2])
        return net
    return tf.image.resize_bilinear(net, shape)


def transpose_from_nchw_if_necessary(net, data_format):
    if data_format == 'NCHW':
        return tf.transpose(net, [0, 2, 3, 1])
    return net


def transpose_to_nchw_if_necessary(net, data_format):
    if data_format == 'NCHW':
        return tf.transpose(net, [0, 3, 1, 2])
    return net
