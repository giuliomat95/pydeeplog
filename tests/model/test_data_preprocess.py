from deeplog_trainer.model.data_preprocess import DataPreprocess
import pytest
import argparse
import json
import numpy as np

def get_data():
    filepath = 'batrasio_result/data.json'
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


dataset = []
@pytest.mark.parametrize("seq", get_data())
def test_preprocess(seq):
    dataset.append(seq)
    WINDOW_SIZE = 7
    data_preprocess = DataPreprocess(dataset, window_size=WINDOW_SIZE)
    vocab = data_preprocess.vocab()
    # vocab must be a list of integers:
    assert set(map(type, vocab)) == {int}
    data_preprocess.encode_dataset()
    assert data_preprocess.dict_token2idx()['[PAD]'] == 0
    list_of_chunks = data_preprocess.chunks_seq(seq)
    if len(seq) > WINDOW_SIZE:
        assert np.shape(list_of_chunks) == (len(seq)-WINDOW_SIZE+1, WINDOW_SIZE)
    else:
        assert np.shape(list_of_chunks) == (1, len(seq))

    train_dataset, val_dataset, test_dataset = \
        data_preprocess.split_data(train_ratio=0.7, val_ratio=0.85)
    assert len(train_dataset) == int(len(dataset)*0.7)
    assert len(val_dataset) == int(len(dataset)*0.85)






