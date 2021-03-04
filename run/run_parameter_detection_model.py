import os
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging as logger
import argparse
import json

from deeplog_trainer.parameter_detection.data_preprocess import DataPreprocess
from deeplog_trainer.model.training import ModelTrainer
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.parameter_detection.model_evaluator import ModelEvaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger.basicConfig(stream=sys.stdout, level=logger.INFO, format='%(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input json dataset filepath from root "
                             "folder")
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=5)
    parser.add_argument("--LSTM_units", type=int,
                        help="Put the number of units in each LSTM layer",
                        default=64)
    parser.add_argument("--max_epochs", type=int,
                        help="Put the maximum number of epochs if the process "
                             "is not stopped before by the early_stop",
                        default=100)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.5)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             " validation set", default=0.75)
    parser.add_argument("--early_stop", type=int,
                        help="Put the number of epochs with no improvement "
                             "after which training will be stopped", default=7)
    parser.add_argument("--batch_size", type=int,
                        help="Put the number of samples that will be propagated"
                             " through the network", default=16)
    parser.add_argument("--out_tensorboard_path", type=str,
                        help="Put the name of the folder where to save the "
                             "tensorboard results if desired", default=None)
    parser.add_argument("--alpha", type=float, help="confidence level of the "
                                                    "confidence interval",
                        default=0.95)
    args = parser.parse_args()
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Upload the data to be evaluated
    with open(os.path.join(root_path, args.input_file), 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    num_params = dataset.shape[1]
    data_preprocess = DataPreprocess()
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=args.train_ratio,
                                  val_ratio=args.val_ratio)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    sc_X = MinMaxScaler()
    train_dataset = sc_X.fit_transform(train_dataset)
    val_dataset = sc_X.transform(val_dataset)
    test_dataset = sc_X.transform(test_dataset)
    logger.info('Datasets sizes: {}, {}, {}'.format(len(train_idx),
                                                    len(val_idx),
                                                    len(test_idx)))
    model_manager = ModelManager()
    model = model_manager.build(ModelManager.MODEL_TYPE_LOG_PARAMS,
                                input_size=args.window_size,
                                lstm_units=args.LSTM_units,
                                num_params=num_params)
    X_train, y_train = data_preprocess.generate(train_dataset, args.window_size)
    X_val, y_val = data_preprocess.generate(val_dataset, args.window_size)
    X_test, y_test = data_preprocess.generate(test_dataset, args.window_size)
    model_trainer = ModelTrainer(logger, epochs=args.max_epochs,
                                 early_stop=args.early_stop,
                                 batch_size=args.batch_size)
    # Run training and validation to fit the model
    history = model_trainer.train(model, [X_train, y_train], [X_val, y_val],
                                  metric_index='mse',
                                  out_tensorboard_path=
                                  args.out_tensorboard_path)
    model.summary()
    y_val_pred = model.predict(X_val)
    # Model Evaluation:
    model_evaluator = ModelEvaluator(logger)
    interval, mse_val = model_evaluator.build_confidence_intervals(y_val,
                                                                   y_val_pred,
                                                                   args.alpha)
    model_evaluator.plot_confidence_intervals(interval, mse_val, args.alpha)
    y_test_pred = model.predict(X_test)
    anomalies_idx, mse_test = model_evaluator.get_anomalies_idx(y_test,
                                                                y_test_pred,
                                                                interval)
    model_evaluator.plot_test_errors(interval, mse_test, args.alpha)
    model_evaluator.plot_time_series(sc_X, val_idx[args.window_size-1:],
                                     y_val, y_val_pred,
                                     label='Forecasting results of validation '
                                           'set: ')
    model_evaluator.plot_time_series(sc_X, test_idx[args.window_size-1:],
                                     y_test, y_test_pred,
                                     label='Forecasting result of the testing '
                                           'set: ')
    logger.info("Number of anomalies: {}".format(len(anomalies_idx)))
