from deeplog_trainer.model.data_preprocess import DataPreprocess
import pytest
import json
import numpy as np


@pytest.fixture(scope='function')
def setup(dataset):
    # List of unique keys in the training file
    vocab = list(set([x for seq in dataset for x in seq]))
    data_preprocess = DataPreprocess(vocab=vocab)
    return data_preprocess

def get_data(min_length=4):
    filepath = 'data/data.json'
    dataset = []
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            dataset.append(seq)
            yield seq, dataset

@pytest.mark.parametrize("seq,dataset", get_data())
def test_preprocess(seq, dataset, setup, window_size=7):
    dataset = np.array(dataset, dtype=object)
    dataset = setup.encode_dataset(dataset)
    assert setup.get_dictionaries()[1]['[PAD]'] == 0
    list_of_chunks = setup.chunks_seq(seq, window_size=window_size)
    if len(seq) > window_size:
        assert np.shape(list_of_chunks) == (len(seq)-window_size+1, window_size)
    else:
        assert np.shape(list_of_chunks) == (1, len(seq))
    train_idx, val_idx, test_idx = \
        setup.split_idx(len(dataset), train_ratio=0.7, val_ratio=0.85)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    assert len(train_dataset) == int(len(dataset)*0.7)
    assert len(train_dataset) + len(val_dataset) == int(len(dataset)*0.85)
    train_chunks = setup.chunks(train_dataset, window_size=window_size)
    X_train, y_train = setup.transform(
        train_chunks,
        add_padding=window_size
    )
    assert np.shape(X_train) == (len(train_chunks), window_size,
                                 setup.get_num_tokens())
    assert np.shape(y_train) == (len(train_chunks), setup.get_num_tokens())
