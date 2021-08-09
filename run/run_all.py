import argparse
import json
import logging
import os
import tempfile
from zipfile import ZipFile

from run_anomaly_detection_model import run_drain, run_logkey_model
from run_parameter_detection_model import run_parameter_detection_model
from . import *


def run_all(logger, **kwargs):
    with tempfile.TemporaryDirectory() as td:
        run_drain(logger, args.input_file, td, args.config_file,
                  args.window_size)

        run_logkey_model(logger, args.window_size,
                         td, args.lstm_units, args.max_epochs,
                         args.train_ratio, args.val_ratio, args.early_stop,
                         args.batch_size, args.out_tensorboard_path, args.top_k)

        # Excluded at the moment from run-all script:
        # run_parameter_detection_model(logger, ...)

        zip_directory(td, args.output_path)

def zip_directory(temp_path, zip_file_name):
    with ZipFile(zip_file_name, 'w') as zip_obj:
        # Iterate over all the files in the directory
        for folder_name, sub_folders, filenames in os.walk(temp_path):
            for filename in filenames:
                if filename == 'data.json':
                    continue
                filepath = os.path.join(folder_name, filename)
                # Add file to zipfile
                zip_obj.write(filepath, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_workflows_runner_args(parser)
    add_logkey_model_runner_args(parser)
    # Excluded at the moment from run-all script:
    # add_parameters_model_runner_args(parser)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    try:
        run_all(logger, *args)
    except Exception as e:
        logger.error(e)
        exit(1)
