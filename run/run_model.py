import argparse
import json
import logging
import os
import sys
import numpy as np

from deeplog_trainer.model.data_preprocess import DataPreprocess
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ModelTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_model(logger, input_file, window_size, min_length, output_path,
              output_file, LSTM_units, max_epochs, train_ratio, val_ratio,
              early_stop, batch_size, out_tensorboard_path):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset = []
    # Load the data parsed from Drain:
    with open(os.path.join(root_path, input_file), 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            dataset.append(seq)
    dataset = np.array(dataset, dtype=object)
    # List of unique keys in the training file
    vocab = list(set([x for seq in dataset for x in seq]))
    data_preprocess = DataPreprocess(vocab=vocab)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=train_ratio,
                                  val_ratio=val_ratio)
    train_dataset = dataset[train_idx]
    # Save train dataset for building workflows
    with open(os.path.join(output_path, 'train_dataset.json'), 'w') as f:
        json.dump({'train_dataset': train_dataset.tolist()}, f)
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    num_tokens = data_preprocess.get_num_tokens()
    logger.info(
        'Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx),
                                            len(test_idx)))
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
    model_manager.save(model, output_path, output_file)
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
                             "folder")
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--min_length", type=int,
                        help="Put the minimum length of a sequence to be "
                             "parsed", default=4)
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory")
    parser.add_argument("--output_file", type=str,
                        help="Put the the name of the output model file")
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
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    run_model(logging.getLogger(__name__), args.input_file, args.window_size,
              args.min_length, args.output_path, args.output_file,
              args.LSTM_units, args.max_epochs, args.train_ratio,
              args.val_ratio,
              args.early_stop, args.batch_size, args.out_tensorboard_path)

