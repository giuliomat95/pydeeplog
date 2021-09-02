import argparse
import tempfile
from zipfile import ZipFile

from run.run_drain import drain_runner
from run.run_log_key_detection_model import run_logkey_model
from run.run_parameter_detection_model import run_parameter_detection_model
from run.run_workflow import run_workflows
from . import *


def run_deeplog(logger, input_file, output_path, config_file, window_size,
                lstm_units, max_epochs, train_ratio, val_ratio, early_stop,
                batch_size, out_tensorboard_path, top_k, threshold, back_steps):
    with tempfile.TemporaryDirectory() as td:
        drain_runner(logger, input_file, td, config_file, window_size)
        input_file = os.path.join(td, 'data.json')
        train_dataset, val_dataset, test_dataset, data_preprocess = \
            create_datasets(logger, input_file, window_size, train_ratio,
                            val_ratio)
        run_logkey_model(logger, td, window_size, lstm_units,
                         max_epochs, train_dataset, val_dataset,
                         data_preprocess, early_stop, batch_size,
                         out_tensorboard_path, top_k)
        run_workflows(logger, td, train_dataset, test_dataset, threshold,
                      back_steps)
        # Excluded at the moment from run-all script:
        # run_parameter_detection_model(logger, ...)

        zip_directory(td, os.path.join(output_path, ZIPFILE_NAME))


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
    add_deeplog_runner_args(parser)
    # Excluded at the moment from run-all script:
    # add_parameters_model_runner_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.error("Directory {} can not be created".format(args.output_path))
        exit(1)
    run_deeplog(logger, args.input_file, args.output_path,
                args.config_file, args.window_size, args.lstm_units,
                args.max_epochs, args.train_ratio, args.val_ratio,
                args.early_stop, args.batch_size, args.out_tensorboard_path,
                args.top_k, args.threshold, args.back_steps)
