import argparse
import ast
import json
import logging
import os
import sys
import tempfile
from zipfile import ZipFile

import configparser
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from deeplog_trainer.log_parser.adapter import AdapterFactory, ParseMethods
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ModelTrainer
from . import create_datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_drain(logger, input_file, output_path, config_file, window_size):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config = TemplateMinerConfig()
    config.load(config_file)
    template_miner = TemplateMiner(config=config)
    adapter_factory = AdapterFactory()
    parser = configparser.ConfigParser()
    parser.read(os.path.join(config_file))
    adapter_params = dict(parser['ADAPTER_PARAMS'])
    adapter_params.setdefault('anomaly_labels', '[]')
    adapter_params['anomaly_labels'] = ast.literal_eval(
        adapter_params['anomaly_labels'])

    if 'regex' in parser.options('ADAPTER_PARAMS'):
        adapter_params['regex'] = ast.literal_eval(adapter_params['regex'])
    elif 'delta' in parser.options('ADAPTER_PARAMS'):
        adapter_params['delta'] = ast.literal_eval(adapter_params['delta'])

    adapter = adapter_factory.build_adapter(**adapter_params)
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    logger.info(f"Drain3 started reading from {args.input_file}")
    line_count = 0
    logformat = parser.get('ADAPTER_PARAMS', 'logformat')
    headers, text_regex = ParseMethods.generate_logformat_regex(
        logformat=logformat)
    with open(os.path.join(root_path, input_file), 'r') as f:
        for line in f:
            sess_id, anomaly_flag = adapter.get_session_id(log=line)
            match = text_regex.search(line.strip())
            message = match.group('Content')
            drain_result = drain.add_message(message)
            sessions = session_storage.get_sessions(sess_id,
                                                    drain_result['template_id'])
            templates = session_storage.get_templates(
                drain_result['template_id'], drain_result['template'])
            parameters = session_storage.get_parameters(sess_id,
                                                        drain_result['params'])
            line_count += 1

            if line_count % 1000 == 0:
                logger.info('Processed {} log lines.'.format(line_count))

        logger.info(f'Finished Drain3 process. Total of lines: {line_count}')
    # Import the results in json format
    result = {'data': []}

    if anomaly_flag:
        for sess_id in sessions:
            result['data'].append(dict(template_seq=sessions[sess_id],
                                       template_params=parameters[sess_id],
                                       is_abnormal=anomaly_flag[sess_id],
                                       session_id=sess_id))
    else:
        for sess_id in sessions:
            result['data'].append(dict(template_seq=sessions[sess_id],
                                       template_params=parameters[sess_id],
                                       session_id=sess_id))
    with open(os.path.join(output_path, 'data.json'), 'w') as f:
        json.dump(result, f)
    with open(os.path.join(output_path, 'templates.json'), 'w') as g:
        json.dump(templates, g)
    # Save the Drain tree object in a JSON file
    with open(os.path.join(output_path, 'drain.json'), 'w') as h:
        drain_serialized = drain.serialize_drain()
        json.dump(drain_serialized, h)
    with open(os.path.join(output_path, 'session_grouper_conf.json'),
              'w') as sg:
        sg_dict = {'sessionGrouperType': "RegexSessionGrouper",
                   'textRegex': text_regex.pattern.replace('?P', '?'),
                   'sessionIdRegex': adapter_params.setdefault('regex', ''),
                   'sessionDelimiters': [adapter_params['delimiter']] if
                   'delimiter' in adapter_params else [],
                   'windowSize': window_size
                   }
        json.dump(sg_dict, sg)


def run_model(logger, window_size, output_path,
              LSTM_units, max_epochs, train_ratio, val_ratio,
              early_stop, batch_size, out_tensorboard_path, top_k):
    input_file = os.path.join(output_path, 'data.json')
    train_dataset, val_dataset, test_dataset, data_preprocess = create_datasets(
        logger, input_file, window_size, train_ratio, val_ratio)
    num_tokens = data_preprocess.get_num_tokens()
    model_manager = ModelManager()
    model = model_manager.build(ModelManager.MODEL_TYPE_LOG_KEYS,
                                input_size=window_size,
                                lstm_units=LSTM_units,
                                num_tokens=num_tokens)
    model.summary()
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset, window_size=window_size),
        add_padding=window_size
    )
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=window_size),
        add_padding=window_size
    )
    model_trainer = ModelTrainer(logger, epochs=max_epochs,
                                 early_stop=early_stop,
                                 batch_size=batch_size)
    # Run training and validation to fit the model
    model_trainer.train(model, [X_train, y_train], [X_val, y_val],
                        out_tensorboard_path=out_tensorboard_path)
    # Save the model
    model_manager.save(model, output_path, 'logkey_model.h5')
    # Calculate scores for different K values in the validation set
    model_evaluator = ModelEvaluator(model, top_k=top_k)
    scores = model_evaluator.compute_scores(X_val, y_val)
    logger.info('-' * 10 + ' K = ' + str(top_k) + ' ' + '-' * 10)
    logger.info('- Num. items: {}'.format(scores['n_items']))
    logger.info('- Num. normal: {}'.format(scores['n_correct']))
    logger.info('- Accuracy: {:.4f}'.format(scores['accuracy']))
    # Save config values in a json file:
    with open(os.path.join(output_path, 'deeplog_conf.json'), 'w') as f:
        par = dict(numTemplates=num_tokens - 3, topCandidates=top_k,
                   windowSize=window_size)
        json.dump(par, f)
    # Save empty workflow in json file
    with open(os.path.join(output_path, 'workflows.json'), 'w') as f:
        network_dict = {"root": {"value": None,
                                 "children": {},
                                 "parents": [],
                                 "is_start": False,
                                 "is_end": False}}
        json.dump(network_dict, f)


def zip_directory(dirname, zip_file_name):
    with ZipFile(zip_file_name, 'w') as zip_obj:
        # Iterate over all the files in the directory
        for folder_name, sub_folders, filenames in os.walk(dirname):
            for filename in filenames:
                if filename == 'data.json':
                    continue
                filepath = os.path.join(folder_name, filename)
                # Add file to zipfile
                zip_obj.write(filepath, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input filepath from root folder")
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory")
    parser.add_argument("--config_file", type=str,
                        help="Put the filepath of the config file")
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--LSTM_units", type=int,
                        help="Put the number of units in each LSTM layer",
                        default=64)
    parser.add_argument("--max_epochs", type=int,
                        help="Put the maximum number of epochs if the process "
                             "is not stopped before by the early_stop",
                        default=50)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.7)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             " validation set", default=0.85)
    parser.add_argument("--early_stop", type=int,
                        help="Put the number of epochs with no improvement "
                             "after which training will be stopped", default=7)
    parser.add_argument("--batch_size", type=int,
                        help="Put the number of samples that will be propagated"
                             " through the network", default=512)
    parser.add_argument("--out_tensorboard_path", type=str,
                        help="Put the name of the folder where to save the "
                             "tensorboard results if desired", default=None)
    parser.add_argument("--top_k", type=int,
                        help="Put the number of top candidates to estimate the "
                             "number of anomalies", default=10)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    with tempfile.TemporaryDirectory() as td:
        run_drain(logger, args.input_file, td, args.config_file,
                  args.window_size)

        run_model(logger, args.window_size,
                  td, args.LSTM_units, args.max_epochs,
                  args.train_ratio, args.val_ratio, args.early_stop,
                  args.batch_size, args.out_tensorboard_path, args.top_k)

        zip_directory(td, args.output_path)
