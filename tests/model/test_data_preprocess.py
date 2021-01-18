from deeplog_trainer.model.data_preprocess import DataPreprocess
import pytest
import argparse
import json
import numpy as np
import pdb

dataset = []
WINDOW_SIZE = 7
@pytest.fixture(scope='function')
def setup(seq):
    dataset.append(seq)
    data_preprocess = DataPreprocess(np.array(dataset, dtype=object),
                                     window_size=WINDOW_SIZE)
    return data_preprocess

def get_data():
    filepath = 'data/data.json'
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        MIN_LENGTH = 4
        for d in data['data']:
            seq = d['template_seq'][1:]
            # the first element is skipped since in batrasio data all
            # the sequences start with the same encoding number
            if len(seq) < MIN_LENGTH:
                # Skip short sequences
                continue
            yield seq


@pytest.mark.parametrize("seq", get_data())
def test_preprocess(seq, setup):
    vocab = setup.vocab()
    # vocab must be a list of integers:
    assert set(map(type, vocab)) == {int}
    setup.encode_dataset()
    assert setup.dict_token2idx()['[PAD]'] == 0
    list_of_chunks = setup.chunks_seq(seq)
    if len(seq) > WINDOW_SIZE:
        assert np.shape(list_of_chunks) == (len(seq)-WINDOW_SIZE+1, WINDOW_SIZE)
    else:
        assert np.shape(list_of_chunks) == (1, len(seq))
    train_dataset, val_dataset, test_dataset = \
        setup.split_data(train_ratio=0.7, val_ratio=0.85)
    assert len(train_dataset) == int(len(dataset)*0.7)
    assert len(train_dataset) + len(val_dataset) == int(len(dataset)*0.85)
    train_chunks = setup.chunks(train_dataset)
    X_train, y_train = setup.transform(
        train_chunks,
        add_padding=WINDOW_SIZE
    )
    assert np.shape(X_train) == (len(train_chunks), WINDOW_SIZE,
                                 setup.get_num_tokens())
    assert np.shape(y_train) == (len(train_chunks), setup.get_num_tokens())


