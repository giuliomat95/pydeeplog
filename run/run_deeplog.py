import argparse
import tempfile

from run.run_drain import run_drain as drain_runner
from run.run_log_key_detection_model import run_log_key_model
from run.run_parameter_detection_model import run_parameter_detection_model
from run.run_workflow import run_workflows
from . import *


def run_deeplog(logger, input_file, output_path, config_file, window_size,
                lstm_units, max_epochs, train_ratio, val_ratio, early_stop,
                batch_size, out_tensorboard_path, top_k):
    with tempfile.TemporaryDirectory() as td:
        drain_runner(logger, input_file, td, config_file, window_size)
        train_dataset, val_dataset, test_dataset, data_preprocess = \
            create_datasets(logger, td, window_size, train_ratio,
                            val_ratio)
        run_log_key_model(logger, td, window_size, lstm_units,
                          max_epochs, train_dataset, val_dataset,
                          data_preprocess, early_stop, batch_size,
                          out_tensorboard_path, top_k)
        # Save empty workflow in json file
        with open(os.path.join(td, 'workflows.json'), 'w') as f:
            network_dict = {"root": {"value": None,
                                     "children": {},
                                     "parents": [],
                                     "is_start": False,
                                     "is_end": False}}
            json.dump(network_dict, f)

        # run_workflows(logger, td, train_dataset, test_dataset, threshold,
        # back_steps)
        # Excluded at the moment from run-all script:
        # run_parameter_detection_model(logger, ...)

        zip_directory(td, os.path.join(output_path, ZIPFILE_APP_MODEL),
                      skip_files=['data.json'])
        zip_directory(td, os.path.join(output_path, ZIPFILE_CORE_MODEL),
                      skip_files=['data.json', 'session_grouper_conf.json',
                                  'drain.json', 'workflows.json',
                                  ZIPFILE_APP_MODEL])
        os.replace(os.path.join(td, 'drain.json'),
                   os.path.join(output_path, 'drain.json'))
        os.replace(os.path.join(td, 'logkey_model.h5'),
                   os.path.join(output_path, 'logkey_model.h5'))
        os.replace(os.path.join(td, 'workflows.json'),
                   os.path.join(output_path, 'workflows.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    add_drain_runner_args(parser)
    add_log_key_model_runner_args(parser)
    # add_workflows_runner_args(parser)
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
                args.top_k)
