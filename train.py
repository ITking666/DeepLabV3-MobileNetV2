"""
This script is used to train the model. It reads a .json config file that was used previously for training.
Then, training loop is executed automatically.

Usage: python train.py --config [config_filename]
Example: python train.py --config config/train_exp1_apple.json
"""
import os
import tensorflow as tf
from mn_v2_deeplab_v3_plus import deeplab_v3_plus_estimator_fn
from utils.generic_util import parse_args
from utils.dataset_util import input_fn_images_labels, get_num_records
import logging


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    train_data_size = get_num_records(args.train_data_file)
    num_iterations = train_data_size // args.batch_size

    # Make args.log_every equal to -1 to print every epoch
    log_every = args.log_every if args.log_every != -1 else num_iterations
    # Make a RunConfig to save a checkpoint per training epoch
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=num_iterations,
                                                  keep_checkpoint_max=args.max_num_checkpoints,
                                                  log_step_count_steps=log_every, session_config=config)
    # Check if there is a checkpoint from which a model will be loaded or not
    warm_start_from = args.experiment_dir if tf.train.get_checkpoint_state(args.experiment_dir) is not None else None

    # Call TFEstimator and pass the model function to it
    model = tf.estimator.Estimator(
        model_fn=deeplab_v3_plus_estimator_fn,
        model_dir=args.experiment_dir,
        config=run_config,
        params={
            'experiment_dir': args.experiment_dir,
            'pretrained_model_dir': args.pretrained_model_dir,
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
            'num_epochs': args.num_epochs,
            'num_iterations': num_iterations,
            'data_format': args.data_format,
            'max_num_tensorboard_images': args.max_num_tensorboard_images,
            'log_every': log_every,
            'tensorboard_update_every': args.tensorboard_update_every,
            'dataset_mean_values': args.dataset_mean_values,
        }, warm_start_from=warm_start_from)

    if args.train_data_file == "" or args.val_data_file == "":
        raise ValueError("Train and Validation data files must exist")

    # Create a train specification object and evaluation specification object and pass the input function to both of them respectively
    aug_params = {'aug_delta_brightness': args.aug_delta_brightness,
                  'aug_flip_left_right': args.aug_flip_left_right,
                  "aug_delta_scale_pad_crop": args.aug_delta_scale_pad_crop,
                  "dataset_mean_values": args.dataset_mean_values}
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn_images_labels(args.train_data_file, args.image_size, args.batch_size, args.num_epochs,
                                                args.shuffle, args.buffer_size, is_training=True, aug_params=aug_params),
        max_steps=(args.num_epochs-1) * num_iterations)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn_images_labels(args.val_data_file, args.image_size, args.batch_size, 1,
                                                args.shuffle, args.buffer_size, is_training=False,
                                                aug_params=aug_params), throttle_secs=1)

    tf.estimator.train_and_evaluate(model, train_spec=train_spec, eval_spec=eval_spec)

    tf.logging.info("Training completed successfully!")


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()
