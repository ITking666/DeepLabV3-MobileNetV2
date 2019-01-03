import os
import logging
import cv2
import watershed
from utils.generic_util import parse_args
from inference_api import ExportedModel
import pyzed.sl as sl
import numpy as np
import math
import sys


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    # initialize zed
    zed = sl.Camera()

    # Set configuration parameters
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_QUALITY
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

    # Prepare image size to retrieve images
    new_width = args.image_size[1]
    new_height = args.image_size[0]
    print("The depth range is from {0} to {1}".format(zed.get_depth_min_range_value(), zed.get_depth_max_range_value()))

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
    point_cloud = sl.Mat()

    # Allocation is done before the loop to make it as fast as possible
    output_frame = np.zeros((args.image_size[0], args.image_size[1] * 2, args.image_size[2])).astype(np.uint8)

    model = ExportedModel(os.path.join(args.experiment_dir, args.test_model_timestamp_directory), args.image_size)

    # mode can be "center" or "median", in the median mode, number of samples should be specified
    mode = "median"
    num_sample = 10

    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(new_width), int(new_height))
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA, sl.MEM.MEM_CPU, int(new_width), int(new_height))
        if image_zed is None:
            raise ValueError("Camera is not connected or not detected properly.")

        frame = image_zed.get_data()
        # Resize to the size provided in the config file
        rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_input, predictions, predictions_decoded = model.predict(rgb_input)

        # add the watershed algorithm to locate each apple of the frame
        predictions_decoded, fruit_centers, radius, fruit_size = watershed.fruit_center_size(predictions_decoded)

        if mode == "center":
            # calculate the depth of each fruit center based on the point clouds
            for i in range(len(fruit_centers)):
                distance = get_distance(fruit_centers[i][0], fruit_centers[i][1], point_cloud)
                if not np.isnan(distance) and not np.isinf(distance):
                    distance = round(distance)
                    fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], distance)
                    cv2.putText(predictions_decoded, "{}mm".format(distance), (int(fruit_centers[i][0]) + 4, int(fruit_centers[i][1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    print("Distance to Camera at fruit {0}: {1} mm\n".format(i, distance))
                else:
                    fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], None)
                    cv2.putText(predictions_decoded, "N/A",
                                (int(fruit_centers[i][0]) + 4, int(fruit_centers[i][1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    print("Can't estimate distance at fruit {0}".format(i))
        elif mode == "median":
            # calculate the depth of each fruit based on median
            for i in range(len(fruit_centers)):
                distance = get_sampled_distances(num_sample, fruit_centers[i], radius[i], point_cloud)
                if not len(distance) == 0:
                    distance = round(np.median(distance))
                    fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], distance)
                    cv2.putText(predictions_decoded, "{}mm".format(distance),
                                (int(fruit_centers[i][0]) - 4 , int(fruit_centers[i][1]) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    print("Distance to Camera at fruit {0}: {1} mm\n".format(i, distance))
                else:
                    fruit_centers[i] = (fruit_centers[i][0], fruit_centers[i][1], None)
                    cv2.putText(predictions_decoded, "N/A",
                                (int(fruit_centers[i][0]) + 4, int(fruit_centers[i][1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    print("Can't estimate distance at fruit {0}".format(i))

        # Fast hack as stated before. Add both images to the width axis.
        output_frame[:, :args.image_size[1]] = rgb_input
        output_frame[:, args.image_size[1]:] = predictions_decoded
        cv2.imshow('window', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sys.stdout.flush()

    # When everything done, release the capture
    cv2.destroyAllWindows()
    zed.close()


# get the distance of a specific point
def get_distance(x, y, point_cloud):
    err, point_cloud_value = point_cloud.get_value(round(x), round(y))
    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                         point_cloud_value[1] * point_cloud_value[1] +
                         point_cloud_value[2] * point_cloud_value[2])
    return distance


# sample from the circle randomly and return a list of distances
def get_sampled_distances(num_sample, fruit_center, radius, point_cloud):
    j = 0
    distance = []
    while j < num_sample:
        rand_r = np.random.rand(1) * radius
        rand_theta = np.random.rand(1) * 2 * np.pi
        x = int(fruit_center[0] + rand_r * np.cos(rand_theta))
        y = int(fruit_center[1] + rand_r * np.sin(rand_theta))
        if x < 0 or x > 308 or y < 0 or y > 202:
            continue
        j = j + 1
        d = get_distance(x, y, point_cloud)
        if not np.isnan(d) and not np.isinf(d):
            distance.append(d)
    return distance


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()
