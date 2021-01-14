import os
import sys
import numpy as np
from deeplog_trainer.model.data_preprocess import DataPreprocess
from deeplog_trainer.model.model_manager import ModelManager
from deeplog_trainer.model.training import ValLossLogger, ModelTrainer
from deeplog_trainer.model.model_evaluator import ModelEvaluator
import argparse
import json
import logging as logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger.basicConfig(level=logger.DEBUG)

if __name__ == '__main__':
    WINDOW_SIZE = 10
    MIN_LENGTH = 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Put the path to the file to be parsed")
    args = parser.parse_args()
    INPUT_FILE = 'data.json'
    dataset = []
    # load the data parsed from Drain:
    with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + args.input_dir + '/' + INPUT_FILE,
              'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq'][1:]  # the first element is skipped since in batrasio data all the sequences start
            # with the same encoding number
            if len(seq) < MIN_LENGTH:
                # Skip short sequences
                continue
            dataset.append(seq)

    dataset = np.array(dataset, dtype=object)
    vocab = list(set([x for seq in dataset for x in seq]))  # list of unique keys in the training file
    data_preprocess = DataPreprocess(start_token=1, vocab=vocab, window_size=WINDOW_SIZE)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = data_preprocess.split_idx(len(dataset))
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    num_tokens = data_preprocess.get_num_tokens()
    logger.info('Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx), len(test_idx)))
    model_manager = ModelManager(WINDOW_SIZE, num_tokens)
    model = model_manager.build()
    model.summary()
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset),
        add_padding=WINDOW_SIZE
    )
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset),
        add_padding=WINDOW_SIZE
    )
    model_trainer = ModelTrainer(logger, epochs=100)
    # Run training and validation to fit the model
    model_trainer.train(model, [X_train, y_train], [X_val, y_val])
    # Save the model
    filepath = 'run/model_result/'
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    filename = 'LSTM'
    model_manager.save(model, filepath, filename)
    # Calculate scores for different K values in the validation set
    for k in range(1, 5):
        model_evaluator = ModelEvaluator(model, X_val, y_val, top_k=k)
        scores = model_evaluator.compute_scores()
        print('-' * 10, 'K = %d' % k, '-' * 10)
        print('- Num. items: %d' % scores['n_items'])
        print('- Num. normal: %d' % scores['n_normal'])
        print('- Accuracy: %.4f' % scores['accuracy'])


