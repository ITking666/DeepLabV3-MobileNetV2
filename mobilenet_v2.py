import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.layers import inverted_residual_block_sequence, create_arg_scope


class MobileNetV2:
    def __init__(self, net, kernel_size=(3, 3), downsampling_factor=16, width_multiplier=1.0, weight_decay=0.0,
                 dropout_keep_prob=1.0, batchnorm=True, batchnorm_decay=0.999, is_training=True, data_format='NHWC',
                 scope='MobileNetV2'):
        """
        MobileNet-V2 feature extractor class as specified in https://arxiv.org/pdf/1801.04381.pdf
        By constructing this class, the network is built automatically.
        ----
        Args:
        ----
        :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

        :param kernel_size: is the kernel size for the internal layers which aren't fixed to 1x1 convolution

        :param downsampling_factor: is the downsampling factor. If 16, then an image will be downsampled by factor equals 1/16. If 8, then an image will be downsampled by a factor equals 1/8

        :param width_multiplier: is the multiplier for the number of channels as specified in the paper

        :param weight_decay: is the L2 regularization strength hyperparameter

        :param dropout_keep_prob: the probability of keeping neurons for dropout layer

        :param batchnorm: a boolean to enable or disable batch normalization

        :param batchnorm_decay: a float number to represent the decay rate for the moving average of batch normalization

        :param is_training: a boolean used for both [batchnorm and dropout] layers which should be true during training and false during testing

        :param data_format: 'NCHW' or 'NHWC'. It's proved that 'NCHW' provides a performance boost for GPUs. However, 'NHWC' is the only one that can be used for CPU computations

        :param scope: is the name of the variable scope of the network

        """
        if downsampling_factor not in [8, 16]:
            raise ValueError("Currently supported downsampling factors are 8 and 16")

        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError("Data format is either NCHW or NHWC")

        # These are the strides as proposed in the paper if the downsampling factor is 16
        s1 = 2
        if downsampling_factor == 8:
            s1 = 1

        # Network settings:
        # 1. t: expansion factor
        # 2. c: number of output filters
        # 3. n: number of repeated layers in the block
        # 4. s: the initial stride for downsampling
        network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1},
                            {'t': 1, 'c': 16, 'n': 1, 's': 1},
                            {'t': 6, 'c': 24, 'n': 2, 's': 1},
                            {'t': 6, 'c': 32, 'n': 3, 's': 2},
                            {'t': 6, 'c': 64, 'n': 4, 's': 2},
                            {'t': 6, 'c': 96, 'n': 3, 's': 1},
                            {'t': 6, 'c': 160, 'n': 3, 's': 2},
                            {'t': 6, 'c': 320, 'n': 1, 's': 1},
                            {'t': None, 'c': 1280, 'n': 1, 's': 1}]

        with tf.variable_scope(scope):
            with slim.arg_scope(
                    create_arg_scope(weight_decay, dropout_keep_prob, batchnorm, batchnorm_decay, is_training,
                                          data_format)):
                # Feature Extraction part
                # Layer 0
                self.end_point = slim.conv2d(net, int(network_settings[0]['c'] * width_multiplier), kernel_size,
                                                stride=network_settings[0]['s'], activation_fn=tf.nn.relu6)

                # Block sequences 1 to 7. Each one takes settings from the above array.
                for i in range(1, 7):
                    self.end_point = inverted_residual_block_sequence(self.end_point,
                                                                         int(network_settings[i - 1][
                                                                             'c'] * width_multiplier),
                                                                         network_settings[i]['c'],
                                                                         network_settings[i]['n'],
                                                                         network_settings[i]['t'],
                                                                         network_settings[i]['s'],
                                                                         kernel_size)
                    if i == 3:
                        self.mid_point = self.end_point
                # Last Layer
                self.end_point = slim.conv2d(self.end_point, int(network_settings[8]['c'] * width_multiplier), 1,
                                                stride=network_settings[8]['s'], activation_fn=tf.nn.relu6)


# Test drive for MobileNet-V2
if __name__ == '__main__':
    tf.reset_default_graph()
    # Creating a dummy placeholder
    x = tf.placeholder(tf.float32, [None, 3, 200, 200])
    # Class instantiation
    print("Building MobileNet-V2...")
    mobilenetv2 = MobileNetV2(x, 3, 16, 1, 0.00001, 0.999)
    # Success message
    print("MobileNet-V2 is built successfully.")
    print(mobilenetv2.end_point.shape, mobilenetv2.mid_point.shape)
