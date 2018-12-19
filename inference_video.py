import os
import logging
import cv2
import skvideo.io
import numpy as np
import watershed
from utils.generic_util import parse_args
from tqdm import tqdm
from inference_api import ExportedModel


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    model = ExportedModel(os.path.join(args.experiment_dir, args.test_model_timestamp_directory), args.image_size)
    videogen = skvideo.io.vreader(args.test_video_file)
    videosav = skvideo.io.FFmpegWriter(args.output_video_file)

    # Allocation is done before the loop to make it as fast as possible
    dump_every = args.dump_frames_every
    output_frame = np.zeros((dump_every, args.image_size[0], args.image_size[1] * 2, args.image_size[2])).astype(
        np.uint8)

    frame = 0
    for rgb_input in tqdm(videogen):
        # Resize to the size provided in the config file
        rgb_input, predictions, predictions_decoded = model.predict(rgb_input)
        # add the watershed algorithm to locate each apple of the frame
        predictions_decoded, fruit_centers, fruit_size = watershed.fruit_center_size(predictions_decoded)
        print(fruit_centers)
        print(fruit_size)
        # Fast hack as stated before. Add both images to the width axis.
        output_frame[frame, :, :args.image_size[1]] = rgb_input
        output_frame[frame, :, args.image_size[1]:] = predictions_decoded
        cv2.imshow('window', cv2.cvtColor(output_frame[frame], cv2.COLOR_RGB2BGR))
        frame += 1

        if frame == dump_every:
            if args.save_output_video:
                videosav.writeFrame(output_frame)
            frame = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    videosav.close()


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()
