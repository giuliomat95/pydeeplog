import os
import sys
import numpy as np
from deeplog_trainer.model.data_preprocess import DataPreprocess
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    WINDOW_SIZE = 10
    MIN_LENGTH = 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Put the path to the file to be parsed")
    args = parser.parse_args()
    INPUT_FILE = 'data.json'
    dataset = []
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
    vocab_size = len(vocab)
    data_preprocess = DataPreprocess(start_token=1, vocab_size=vocab_size, vocab=vocab)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = data_preprocess.split_idx(len(dataset))
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    num_tokens = data_preprocess.get_num_tokens()
    print('Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx), len(test_idx)))