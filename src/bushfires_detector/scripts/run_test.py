"""Command line script to upload a whole directory tree."""
from __future__ import print_function

from builtins import input
import argparse
import cProfile as profile
import sys
from pathlib import Path
from loguru import logger as logging
from tensorflow import keras
import numpy as np

# local
from bushfires_detector.data_management import DataManagement
from bushfires_detector.plot_helpers import display_digits_with_boxes
from bushfires_detector.perf_helpers import intersection_over_union
from bushfires_detector.utils import load_conf

__author__ = ["Ariel Hernandez <ahestevenz@bleiben.ar>"]
__copyright__ = "Copyright 2022 Bleiben. All rights reserved."
__license__ = """General Public License"""


def _main(args):
    """Actual program (without command line parsing). This is so we can profile.
    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """

    # 1. Load configuration
    logging.info(f"Loading configuration file from {args['json_file']}")
    if not Path(args['json_file']).exists():
        logging.error(
            f'{args["json_file"]} does not exist. Please check config.json path and try again')
        return -1
    conf = load_conf(args['json_file'])
    experiment_path = Path(
        conf['main']['artefacts']+f'/run_{str(conf["test"]["run"])}')
    if not experiment_path.exists():
        logging.error(
            f'{experiment_path} does not exist. Please check config.json file')
        return -1
    logging.info(f'Experiment directory: {experiment_path}')

    # 2. Load model
    exp_conf = load_conf(
        experiment_path/Path("config.json"), from_string=True)
    model_name = f"smoke_detector_mobilenet_{exp_conf['model']['mobilenet']}_ep_{exp_conf['train']['epochs']}_bs_{exp_conf['train']['batch_size']}_dataset_{exp_conf['data']['release']}_len_data_{exp_conf['data']['len_train_dataset']}"
    model_name = f'{model_name}_augmented_data.h5' if exp_conf[
        'data']['need_augmentation'] else f'{model_name}.h5'
    logging.info(f'Loading model: {model_name}')
    model = keras.models.load_model(experiment_path/Path(model_name))
    logging.debug(model.summary())

    # 3. Testing
    logging.info(f'Testing model using {exp_conf["data"]["release"]} dataset')
    data_mgt = DataManagement(
        exp_conf['data']['path'], exp_conf['train']['batch_size'], exp_conf['data']['release'])
    _, _, ds_test = data_mgt.get_dataset('test')
    visualization_test_dataset = ds_test.map(
        data_mgt.original_image_and_bboxes_from_path, num_parallel_calls=16)
    (visualization_test_images, visualization_test_bboxes) = data_mgt.dataset_to_numpy_arrays(
        visualization_test_dataset, N=10)
    display_digits_with_boxes(np.array(visualization_test_images),
                            np.array([]),
                            np.array(visualization_test_bboxes),
                            np.array([]),
                            "Validation images and their bboxes",
                            experiment_path)

    original_images, normalized_images, normalized_bboxes = data_mgt.dataset_to_numpy_arrays_with_original_bboxes(
        visualization_test_dataset, N=500)
    predicted_bboxes = model.predict(normalized_images, batch_size=32)
    iou = intersection_over_union(
        predicted_bboxes, normalized_bboxes)
    logging.info(f"IOU: {iou}")
    logging.info(
        f"Number of predictions where iou > threshold({conf['perf']['iou_threshold']}): {(iou >= conf['perf']['iou_threshold']).sum()}")
    logging.info(
        f"Number of predictions where iou < threshold({conf['perf']['iou_threshold']}): {(iou < conf['perf']['iou_threshold']).sum()})")

    n = exp_conf["test"]["num_samples_to_show"]
    indexes = np.random.choice(len(predicted_bboxes), size=n)
    display_digits_with_boxes(original_images[indexes],
                            predicted_bboxes[indexes],
                            normalized_bboxes[indexes],
                            iou[indexes],
                            "True and Predicted values",
                            experiment_path,
                            bboxes_normalized=True)
    return 0


def main():
    """CLI for upload the encripted files"""

    # Module specific
    argparser = argparse.ArgumentParser(
        description='Welcome to the Bushfires Detector testing script')
    argparser.add_argument('-j', '--json_file', help='JSON configuration (default: "%(default)s")', required=False,
                           default='/Users/ahestevenz/Desktop/tech-projects/1_code/bushfires-object-detection/config.json')

    # Default Args
    argparser.add_argument('-v', '--verbose', help='Increase logging output  (default: INFO)'
                           '(can be specified several times)', action='count', default=0)
    argparser.add_argument('-p', '--profile', help='Run with profiling and store '
                           'output in given file', metavar='output.prof')
    args = vars(argparser.parse_args())

    _V_LEVELS = ["INFO", "DEBUG"]
    loglevel = min(len(_V_LEVELS)-1, args['verbose'])
    logging.remove()
    logging.add(sys.stdout, level=_V_LEVELS[loglevel])

    if args['profile'] is not None:
        logging.info("Start profiling")
        r = 1
        profile.runctx("r = _main(args)", globals(),
                       locals(), filename=args['profile'])
        logging.info("Done profiling")
    else:
        logging.info("Running without profiling")
        r = _main(args)
    return r


if __name__ == '__main__':
    exit(main())
