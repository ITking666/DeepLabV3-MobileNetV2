import os
import logging
import cv2
import watershed
from utils.generic_util import parse_args
from inference_api import ExportedModel
import pyzed.sl as sl
import numpy as np
import math

def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    # initialize zed
    zed = sl.Camera()

    # Set configuration parameters
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
    init.coordinate_units = sl.UNIT.UNIT_MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_resolution()
    new_width = image_size.width / 2
    new_height = image_size.height / 2
    print("The depth range is from {0} to {1}".format(zed.get_depth_min_range_value(), zed.get_depth_max_range_value()))

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
    point_cloud = sl.Mat()

    # Allocation is done before the loop to make it as fast as possible
    output_frame = np.zeros((args.image_size[0], args.image_size[1] * 2, args.image_size[2])).astype(np.uint8)

    model = ExportedModel(os.path.join(args.experiment_dir, args.test_model_timestamp_directory), args.image_size)

    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(new_width), int(new_height))
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)
        if image_zed is None:
            raise ValueError("Camera is not connected or not detected properly.")

        frame = image_zed.get_data()
        # Resize to the size provided in the config file
        rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_input, predictions, predictions_decoded = model.predict(rgb_input)

        # add the watershed algorithm to locate each apple of the frame
        predictions_decoded, fruit_centers, fruit_size = watershed.fruit_center_size(predictions_decoded)

        # calculate the depth of each fruit center based on the point clouds
        for i in range(len(fruit_centers)):
            err, point_cloud_value = point_cloud.get_value(round(fruit_centers[i][0]), round(fruit_centers[i][1]))
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])
            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], distance)
                print("Distance to Camera at fruit {0}: {1} mm\n".format(i, distance))
            else:
                fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], None)
                print("Can't estimate distance at fruit {0}".format(i))

        # Fast hack as stated before. Add both images to the width axis.
        output_frame[:, :args.image_size[1]] = rgb_input
        output_frame[:, args.image_size[1]:] = predictions_decoded
        cv2.imshow('window', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cv2.destroyAllWindows()
    zed.close()


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()