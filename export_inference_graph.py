"""
This script is used to export a graph for inference. It reads the same .json config file that was used previously for training.
Then, it outputs the exported timestamped graph to the same experiment directory.

Usage: python export_inference_graph.py --config [config_filename]
Example: python export_inference_graph.py --config config/train_exp1_apple.json
"""
import os
import tensorflow as tf
from mn_v2_deeplab_v3_plus import deeplab_v3_plus_estimator_fn
from utils.generic_util import parse_args
from utils.image_util import mean_image_subtraction
import logging


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    # Call TFEstimator and pass the model function to it
    model = tf.estimator.Estimator(
        model_fn=deeplab_v3_plus_estimator_fn,
        model_dir=args.experiment_dir,
        params={
            'experiment_dir': args.experiment_dir,
            'num_classes': args.num_classes,
            'downsampling_factor': args.output_stride,
            'width_multiplier': args.width_multiplier,
            'weight_decay': args.weight_decay,
            'dropout_keep_prob': 1.0,
            'batchnorm': args.enable_batchnorm,
            'batchnorm_decay': args.batchnorm_decay,
            'initial_learning_rate': args.initial_learning_rate,
            'final_learning_rate': args.final_learning_rate,
            'learning_rate_power': args.learning_rate_decay_power,
            'num_epochs': None,
            'num_iterations': None,
            'data_format': args.data_format,
            'max_num_tensorboard_images': None,
            'log_every': None,
            'tensorboard_update_every': None,
            'export': True
        })

    # Export the model
    def serving_input_receiver_fn():
        features = tf.placeholder(tf.float32, [None, *args.image_size], name='image_tensor')
        receiver_tensors = {'input': features}
        features = tf.map_fn(lambda image: mean_image_subtraction(image, args.dataset_mean_values), features)
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=receiver_tensors)

    tf.logging.info("Exporting the model to {} ...".format(args.experiment_dir))
    model.export_savedmodel(args.experiment_dir, serving_input_receiver_fn)
    tf.logging.info("Exported successfully!")


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()
