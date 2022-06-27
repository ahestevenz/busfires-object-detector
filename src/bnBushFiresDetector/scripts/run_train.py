"""Command line script to upload a whole directory tree."""
from __future__ import print_function

from builtins import input
import argparse
import cProfile as profile
import sys
from pathlib import Path
import shutil 
from loguru import logger as logging
import numpy as np

# local 
from bnBushFiresDetector import dataset as dataset_mgt
from bnBushFiresDetector import BushFiresDetector
from bnBushFiresDetector import plot_helpers 
from bnBushFiresDetector import perf_helpers 
from bnBushFiresDetector import utils 

__author__ = [ "Ariel Hernandez <ahestevenz@bleiben.ar>" ]
__copyright__ = "Copyright 2022 Bleiben. All rights reserved."
__license__ = """Proprietary"""


def _main(args):
    """Actual program (without command line parsing). This is so we can profile.
    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """

    # 1. Load datasets
    logging.info(f"Loading configuration file from {args['json_file']}")
    if not Path(args['json_file']).exists():
        logging.error(f'{args["json_file"]} does not exist. Please check config.json path and try again')
        return -1
    conf = utils.load_conf(args['json_file'])

    i = 0
    while Path(conf['main']['artefacts']+f'/run_{str(i)}').is_dir():
        i += 1
    experiment_path = Path(conf['main']['artefacts']+f'/run_{str(i)}')
    experiment_path.mkdir(parents=False, exist_ok=True)
    logging.info(f'Experiment directory: {experiment_path}')
    data_mgt = dataset_mgt.DataManagement(conf['data']['path'], conf['train']['batch_size'])
    training_dataset, length_of_training_dataset, _ = data_mgt.get_dataset('train', conf['data']['need_augmentation'])
    validation_dataset, length_of_validation_dataset, _ = data_mgt.get_dataset('valid')

    # 2. Build model
    logging.info(f'Building model...')
    detector = BushFiresDetector.BushFiresDetector(conf['model']['mobilenet'], conf['model']['units_dense_layers'])
    model = detector.define_and_compile_model()
    logging.debug(model.summary())

    steps_per_epoch = length_of_training_dataset//conf['train']['batch_size']

    if length_of_training_dataset % conf['train']['batch_size'] > 0:
        steps_per_epoch += 1
    validation_steps = length_of_validation_dataset//conf['train']['batch_size']
    if length_of_validation_dataset % conf['train']['batch_size'] > 0:
        validation_steps += 1

    # 3. Training
    logging.info(f'Training model...')
    history = model.fit(
                training_dataset,
                validation_data=validation_dataset,
                epochs=conf['train']['epochs'],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=2)
    model_name = f"smoke_detector_mobilenet_{conf['model']['mobilenet']}_ep_{conf['train']['epochs']}_bs_{conf['train']['batch_size']}_len_data_{length_of_training_dataset}"
    model_name = f'{model_name}_augmented_data.h5' if conf['data']['need_augmentation'] else f'{model_name}.h5'
    model.save(experiment_path/Path(model_name))
    
    # 4. Testing
    logging.info(f'Testing model...')
    _, _, ds_test = data_mgt.get_dataset('test')
    visualization_test_dataset = ds_test.map(data_mgt.original_image_and_bboxes_from_path, num_parallel_calls=16)
    (visualization_test_images, visualization_test_bboxes) = data_mgt.dataset_to_numpy_arrays(visualization_test_dataset, N=10)
    plot_helpers.display_digits_with_boxes(np.array(visualization_test_images), 
                                           np.array([]), 
                                           np.array(visualization_test_bboxes), 
                                           np.array([]), 
                                           "Validation images and their bboxes",
                                           experiment_path)

    original_images, normalized_images, normalized_bboxes = data_mgt.dataset_to_numpy_arrays_with_original_bboxes(visualization_test_dataset, N=500)
    predicted_bboxes = model.predict(normalized_images, batch_size=32)
    iou = perf_helpers.intersection_over_union(predicted_bboxes, normalized_bboxes)
    logging.debug(f"IOU: {iou}")
    logging.debug(f"Number of predictions where iou > threshold({conf['perf']['iou_threshold']}): {(iou >= conf['perf']['iou_threshold']).sum()}")
    logging.debug(f"Number of predictions where iou < threshold({conf['perf']['iou_threshold']}): {(iou < conf['perf']['iou_threshold']).sum()})")

    n = conf["test"]["num_samples_to_show"]
    indexes = np.random.choice(len(predicted_bboxes), size=n)
    plot_helpers. display_digits_with_boxes(original_images[indexes], 
                                            predicted_bboxes[indexes], 
                                            normalized_bboxes[indexes], 
                                            iou[indexes], 
                                            "True and Predicted values",
                                            experiment_path,
                                            bboxes_normalized=True)
    # 5. Final tasks
    conf['data']['len_train_dataset'] = length_of_training_dataset
    utils.save_conf(conf, experiment_path/Path("config.json"))
    return 0

def main():
    """CLI for upload the encripted files"""

    # Module specific
    argparser = argparse.ArgumentParser(description='Welcome to the Bushfires Detector training script')
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
        profile.runctx("r = _main(args)", globals(), locals(), filename=args['profile'])
        logging.info("Done profiling")
    else:
        logging.info("Running without profiling")
        r = _main(args)
    return r

if __name__ == '__main__':
    exit(main())
