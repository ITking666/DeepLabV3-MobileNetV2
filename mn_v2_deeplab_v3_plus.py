from os.path import join
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework.errors_impl import NotFoundError
from utils.layers import create_arg_scope, atrous_spatial_pyramid_pooling, resize_bilinear, \
    transpose_from_nchw_if_necessary, \
    transpose_to_nchw_if_necessary
from utils.image_util import decode_labels, mean_image_addition
from utils.generic_util import compute_mean_iou
from mobilenet_v2 import MobileNetV2


class MN2DeepLabV3Plus:
    def __init__(self, input, labels, params, mode='train', scope='DeepLabV3PlusNetwork'):
        """
        DeepLab-V3+ with MobileNet V2 as proposed in the paper https://arxiv.org/pdf/1802.02611.pdf
        ----
        Args:
        ----
        :param input: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

        :param labels: an array or a tuple consisting of [height, width, channels] or [channels, height, width] of the label tensor depending on the data format [NHWC] or [NCHW]

        'params' is a dictionary containing parameters such as:
        -------------------------------------------------------
        :param num_classes: number of labels in the segmentation maps

        :param downsampling_factor: is the downsampling factor. If 16, then an image will be downsampled by factor equals 1/16. If 8, then an image will be downsampled by a factor equals 1/8

        :param width_multiplier: is the multiplier for the number of channels as specified in the paper

        :param weight_decay: is the L2 regularization strength hyperparameter

        :param dropout_keep_prob: the probability of keeping neurons for dropout layer

        :param batchnorm: a boolean to enable or disable batch normalization

        :param batchnorm_decay: a float number to represent the decay rate for the moving average of batch normalization

        :param data_format: 'NCHW' or 'NHWC'. It's proved that 'NCHW' provides a performance boost for GPUs. However, 'NHWC' is the only one that can be used for CPU computations

        ------------------

        :param mode: 'train', 'val', or 'predict'

        :param scope: is the name of the variable scope of the network
        """
        if params['downsampling_factor'] not in [8, 16]:
            raise ValueError("Currently supported downsampling factors are 8 and 16")

        if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            raise ValueError("Modes are \'train\', \'eval\', or \'infer\'")

        if params['data_format'] not in ['NCHW', 'NHWC']:
            raise ValueError("Data format is either NCHW or NHWC")

        # Transpose if the data format is NCHW
        self.input = transpose_to_nchw_if_necessary(input, params['data_format'])

        # Private attributes
        self.__NORMALIZATION_FACTOR = 255

        if mode != tf.estimator.ModeKeys.PREDICT:
            self.__num_epochs = params['num_epochs']
            self.__num_iterations = params['num_iterations']
        self.__input_size = self.input.shape[1:]
        self.__num_classes = params['num_classes']
        self.__downsampling_factor = params['downsampling_factor']
        self.__width_multiplier = params['width_multiplier']
        self.__weight_decay = params['weight_decay']
        self.__dropout_keep_prob = params['dropout_keep_prob']
        self.__batchnorm = params['batchnorm']
        self.__batchnorm_decay = params['batchnorm_decay']
        self.__initial_learning_rate = params['initial_learning_rate']
        self.__final_learning_rate = params['final_learning_rate']
        self.__learning_rate_power = params['learning_rate_power']
        self.__data_format = params['data_format']
        self.__export = params['export'] if 'export' in params else False
        self.__scope = scope
        # Height and width extracted from the input size
        self.__input_2d_sizes = self.__input_size[0:2] if self.__data_format == 'NHWC' else self.__input_size[1:]
        self.__channel_dim = 3 if self.__data_format == 'NHWC' else 1
        self.__mode = mode
        self.__is_training = True if self.__mode == 'train' else False

        # Public attributes
        self.labels = labels
        self.logits = None
        self.y_pred = None
        self.y_pred_decoded = None
        self.loss = None
        self.learning_rate = None
        self.global_step = None
        self.train_op = None
        self.metrics = None

        # Perform architecture building..
        self.__network()
        self.__output()

    def __network(self):
        """
        Define the network itself (DeepLab-V3+) from the input to the logits
        """
        # Network Construction begins here
        with tf.variable_scope(self.__scope):
            # Create the feature extractor which is MobileNet-V2 in this case
            features = MobileNetV2(self.input / self.__NORMALIZATION_FACTOR, 3, self.__downsampling_factor,
                                   self.__width_multiplier, self.__weight_decay,
                                   self.__dropout_keep_prob,
                                   self.__batchnorm,
                                   self.__batchnorm_decay, self.__is_training, self.__data_format)
            with slim.arg_scope(
                    create_arg_scope(self.__weight_decay, self.__dropout_keep_prob, self.__batchnorm,
                                     self.__batchnorm_decay,
                                     self.__is_training,
                                     self.__data_format)):
                # Call the atrous spatial pyramid pooling function which works on the end point
                # of the extracted features
                encoder_output = atrous_spatial_pyramid_pooling(features.end_point, out_filters=256,
                                                                feature_output_stride=self.__downsampling_factor,
                                                                is_training=self.__is_training,
                                                                data_format=self.__data_format)

                with tf.variable_scope('decoder'):
                    with tf.variable_scope("low_level_features"):
                        # Extract low level features from the downsampled (usually 4x) feature maps
                        low_level_features = slim.conv2d(features.mid_point, 48, 1, stride=1)
                        low_level_features_size = low_level_features.shape[
                                                  1:3] if self.__data_format == 'NHWC' else low_level_features.shape[2:]

                    with tf.variable_scope("upsampling"):
                        # Resize the output of the ASPP to be of the same size as the low level feature maps
                        net = resize_bilinear(encoder_output, low_level_features_size, self.__data_format == 'NCHW')

                        # Concatenate the low level feature maps and the output of the ASPP after resizing
                        net = tf.concat([net, low_level_features], axis=self.__channel_dim, name='concat')
                        # Perform two 3x3 convolution operations on them
                        net = slim.conv2d(net, 256, 3, stride=1)
                        net = slim.conv2d(net, 256, 3, stride=1)
                        # Perform last convolution without any activation or normalization
                        net = slim.conv2d(net, self.__num_classes, 1, activation_fn=None, normalizer_fn=None)

                        # Resize the output directly to match the size of the input
                        self.logits = resize_bilinear(net, self.__input_2d_sizes, self.__data_format == 'NCHW')
        # Network Construction ends here

    def __output(self):
        # Predicted labels is the argmax of the logits.
        # Note that softmax is not needed at all because it's only used during loss computation.
        # This provides a performance boost.
        self.y_pred = tf.argmax(self.logits, axis=self.__channel_dim, output_type=tf.int32)
        # Call the decode_label function, as a part of the graph, to produce output images
        self.y_pred_decoded = tf.py_func(decode_labels, [self.y_pred], tf.uint8)
        if self.__mode == 'predict' or self.__export == True:
            return
        self.y_decoded = tf.py_func(decode_labels, [self.labels], tf.uint8)

        # Create loss and optimizer with adaptive learning rate
        # Collect regularization losses
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Create a cross entropy loss
        # Losses in TensorFlow support ONLY NHWC
        self.logits = transpose_from_nchw_if_necessary(self.logits, self.__data_format)
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
            logits=self.logits, labels=self.labels)

        # Total loss is the summation of both losses
        self.loss = cross_entropy_loss + regularization_loss

        # Create a global step for training
        self.global_step = tf.train.get_or_create_global_step()

        # Create a learning rate tensor which supports decaying
        self.learning_rate = tf.train.polynomial_decay(self.__initial_learning_rate,
                                                       tf.cast(self.global_step, tf.int32),
                                                       self.__num_iterations * self.__num_epochs,
                                                       self.__final_learning_rate, power=self.__learning_rate_power)
        # Create Adam optimizer [This optimizer can be changed according to the needs]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, self.global_step)

        # Construct extra needed metrics for training and validation
        accuracy = tf.metrics.accuracy(self.labels, self.y_pred)
        mean_iou = tf.metrics.mean_iou(self.labels, self.y_pred, self.__num_classes)
        self.metrics = {'pixel_wise_accuracy': accuracy, 'mean_iou': mean_iou}


def deeplab_v3_plus_estimator_fn(features, labels, mode, params):
    """
    This is the model function needed for TensorFlow Estimator API. ALL of its parameters are passed by the estimator automatically.
    ----
    Return:
    ----
    A TFEstimator spec either for training or evaluation
    """
    if isinstance(features, dict):
        features = features['feature']

    # Construct the whole network graph
    network_graph = MN2DeepLabV3Plus(features, labels, params, mode)

    predictions = {
        'classes': network_graph.y_pred,
        'decoded_labels': network_graph.y_pred_decoded
    }

    # Do the following only if the TF estimator is in PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            # These are the predictions that are needed from the model
            predictions=predictions,
            # This is very important for TensorFlow serving API. It's the response from a TensorFlow server.
            export_outputs={
                'classes': tf.estimator.export.PredictOutput(network_graph.y_pred),
            })

    # Restore variables from a pretrained model (with the same names) except those in the last layer.
    # This works only in training and in validation modes ONLY.
    try:
        if params['pretrained_model_dir'] != "":
            output_head_scope = 'DeepLabV3PlusNetwork/decoder/upsampling/Conv_2/'
            variables = tf.trainable_variables(scope='DeepLabV3PlusNetwork')
            names = {v.name.split(":")[0]: v.name.split(":")[0] for v in variables}
            names.pop(output_head_scope + 'weights')
            names.pop(output_head_scope + 'biases')
            tf.train.init_from_checkpoint(params['pretrained_model_dir'], names)
    except NotFoundError:
        tf.logging.warning("No pretrained model directory exists. Skipping.")

    def create_summaries_and_logs():
        """
        Construct summaries and logs during training and evaluation
        ----
        Return:
        ----
        a logging hook object and a summary hook object
        """
        # Construct extra metrics for Training and Evaluation
        images = tf.cast(tf.map_fn(lambda f: mean_image_addition(f, params['dataset_mean_values']), features), tf.uint8)
        summary_images = [images, network_graph.y_decoded, network_graph.y_pred_decoded]

        extra_summary_ops = [tf.summary.scalar('loss', network_graph.loss),
                             tf.summary.scalar('pixel_wise_accuracy', network_graph.metrics['pixel_wise_accuracy'][1]),
                             tf.summary.scalar('mean_iou', compute_mean_iou(network_graph.metrics['mean_iou'][1])),
                             # Concatenate them on width axis
                             tf.summary.image('images', tf.concat(axis=2, values=summary_images),
                                              max_outputs=params['max_num_tensorboard_images'])]

        # TFEstimator automatically creates a summary hook during training. So, no need to create one.
        if mode == tf.estimator.ModeKeys.TRAIN:
            extra_summary_ops.append(tf.summary.scalar('learning_rate', network_graph.learning_rate))

            # Construct tf.logging tensors
            train_tensors_to_log = {'epoch': network_graph.global_step // params['num_iterations'],
                                    'learning_rate': network_graph.learning_rate,
                                    'train_px_acc': network_graph.metrics['pixel_wise_accuracy'][1],
                                    'train_mean_iou': compute_mean_iou(network_graph.metrics['mean_iou'][1])}
            logging_hook = tf.train.LoggingTensorHook(tensors=train_tensors_to_log,
                                                      every_n_iter=params['log_every'])

            return [logging_hook]

        summary_output_dir = join(params['experiment_dir'], 'eval')

        # Construct tf.logging tensors
        val_tensors_to_log = {'epoch': network_graph.global_step // params['num_iterations'] - 1,
                              'global_step': network_graph.global_step,
                              'val_loss': network_graph.loss,
                              'val_px_acc': network_graph.metrics['pixel_wise_accuracy'][1],
                              'val_mean_iou': compute_mean_iou(network_graph.metrics['mean_iou'][1])}
        logging_hook = tf.train.LoggingTensorHook(tensors=val_tensors_to_log, every_n_iter=params['log_every'])

        summary_hook = tf.train.SummarySaverHook(params['tensorboard_update_every'], output_dir=summary_output_dir,
                                                 summary_op=tf.summary.merge(extra_summary_ops))

        return [logging_hook, summary_hook]

    hooks = create_summaries_and_logs()

    # Do the following only if the TF estimator is in TRAIN or EVAL modes
    # It computes the loss and optimizes it using the train_op.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=network_graph.loss,
        train_op=network_graph.train_op,
        training_hooks=hooks,
        evaluation_hooks=hooks)
