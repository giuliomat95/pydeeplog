import logging
import sys
import os
import json
import numpy as np

from deeplog_trainer.model.data_preprocess import DataPreprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def create_datasets(logger, input_file, min_length, train_ratio, val_ratio):
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
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    logger.info(
        'Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx),
                                            len(test_idx)))
    return train_dataset, val_dataset, test_dataset, data_preprocess
