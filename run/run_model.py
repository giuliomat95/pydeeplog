import os
import sys
import numpy as np
from deeplog_trainer.model.data_preprocess import DataPreprocess
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ModelTrainer
from deeplog_trainer.model.model_evaluator import ModelEvaluator
import argparse
import json
import logging as logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger.basicConfig(level=logger.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        help="Put the input json dataset filepath from root "
                             "folder")
    parser.add_argument("--window_size", type=int,
                        help="Put the window_size parameter", default=10)
    parser.add_argument("--min_length", type=int,
                        help="Put the minimum length of a sequence to be "
                             "parsed", default=4)
    parser.add_argument("--output", type=str,
                        help="Put the the filepath of the output model file")
    parser.add_argument("--filename", type=str,
                        help="Put the name of the h5 file containing the model")
    parser.add_argument("--LSTM_units", type=int,
                        help="Put the number of units in each LSTM layer",
                        default=64)
    parser.add_argument("--n_epochs", type=int,
                        help="Put the number of epochs", default=50)
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
    args = parser.parse_args()
    dataset = []
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # load the data parsed from Drain:
    with open(os.path.join(root_path, args.filepath), 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq'][
                  1:]  # the first element is skipped since in batrasio data all
            # the sequences start with the same encoding number
            if len(seq) < args.min_length:
                # Skip short sequences
                continue
            dataset.append(seq)

    dataset = np.array(dataset, dtype=object)
    vocab = list(set([x for seq in dataset for x in
                      seq]))  # list of unique keys in the training file
    data_preprocess = DataPreprocess(vocab=vocab)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=args.train_ratio,
                                  val_ratio=args.val_ratio)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    num_tokens = data_preprocess.get_num_tokens()
    logger.info(
        'Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx),
                                            len(test_idx)))
    model_manager = ModelManager(input_size=args.window_size,
                                 num_tokens=num_tokens,
                                 lstm_units=args.LSTM_units)

    model = model_manager.build()
    model.summary()
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset, window_size=args.window_size),
        add_padding=args.window_size
    )
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=args.window_size),
        add_padding=args.window_size
    )
    model_trainer = ModelTrainer(logger, epochs=args.n_epochs,
                                 early_stop=args.early_stop,
                                 batch_size=args.batch_size)
    # Run training and validation to fit the model
    model_trainer.train(model, [X_train, y_train], [X_val, y_val])
    # Save the model
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    model_manager.save(model, args.output, args.filename)
    # Calculate scores for different K values in the validation set
    for k in range(1, 5):
        model_evaluator = ModelEvaluator(model, top_k=k)
        scores = model_evaluator.compute_scores(X_val, y_val)
        print('-' * 10, 'K = %d' % k, '-' * 10)
        print('- Num. items: %d' % scores['n_items'])
        print('- Num. normal: %d' % scores['n_normal'])
        print('- Accuracy: %.4f' % scores['accuracy'])
