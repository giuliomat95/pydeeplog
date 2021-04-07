import argparse
import json
import logging
import os

from . import create_datasets
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ModelTrainer


def run_model(logger, input_file, window_size, min_length, output_path,
              LSTM_units, max_epochs, train_ratio, val_ratio,
              early_stop, batch_size, out_tensorboard_path):
    train_dataset, val_dataset, test_dataset, data_preprocess = create_datasets(
        logger, input_file, min_length, train_ratio, val_ratio)
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
    model_manager.save(model, output_path, 'log_key_model.h5')
    # Calculate scores for different K values in the validation set
    for k in range(1, 5):
        model_evaluator = ModelEvaluator(model, top_k=k)
        scores = model_evaluator.compute_scores(X_val, y_val)
        logger.info('-' * 10 + ' K = ' + str(k) + ' ' + '-' * 10)
        logger.info('- Num. items: {}'.format(scores['n_items']))
        logger.info('- Num. normal: {}'.format(scores['n_correct']))
        logger.info('- Accuracy: {:.4f}'.format(scores['accuracy']))
    # Save config values in a json file:
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        par = dict(window_size=window_size, min_length=min_length)
        json.dump(par, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input json dataset filepath from root "
                             "folder",
                        default='artifacts/drain_result/data.json')
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--min_length", type=int,
                        help="Put the minimum length of a sequence to be "
                             "parsed", default=4)
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory",
                        default='artifacts/log_key_model_result')
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
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.info("Directory '%s' can not be created")
        exit(1)

    run_model(logger, args.input_file, args.window_size,
              args.min_length, args.output_path, args.LSTM_units,
              args.max_epochs, args.train_ratio, args.val_ratio,
              args.early_stop, args.batch_size, args.out_tensorboard_path)
