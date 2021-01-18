from deeplog_trainer.model.data_preprocess import DataPreprocess
import pytest
import json
import numpy as np
import pdb
WINDOW_SIZE = 7
@pytest.fixture(scope='function')
def setup(dataset):
    vocab = list(set([x for seq in dataset for x in
                      seq]))  # list of unique keys in the training file
    data_preprocess = DataPreprocess(vocab=vocab)
    return data_preprocess

def get_data():
    filepath = 'data/data.json'
    dataset = []
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
            dataset.append(seq)
            yield seq, dataset


@pytest.mark.parametrize("seq,dataset", get_data())
def test_preprocess(seq, dataset, setup):
    dataset = np.array(dataset, dtype=object)
    dataset = setup.encode_dataset(dataset)
    assert setup.get_dictionaries()[1]['[PAD]'] == 0
    list_of_chunks = setup.chunks_seq(seq, window_size=WINDOW_SIZE)
    # pdb.set_trace()
    if len(seq) > WINDOW_SIZE:
        assert np.shape(list_of_chunks) == (len(seq)-WINDOW_SIZE+1, WINDOW_SIZE)
    else:
        assert np.shape(list_of_chunks) == (1, len(seq))
    train_idx, val_idx, test_idx = \
        setup.split_idx(len(dataset), train_ratio=0.7, val_ratio=0.85)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    assert len(train_dataset) == int(len(dataset)*0.7)
    assert len(train_dataset) + len(val_dataset) == int(len(dataset)*0.85)
    train_chunks = setup.chunks(train_dataset, window_size=WINDOW_SIZE)
    X_train, y_train = setup.transform(
        train_chunks,
        add_padding=WINDOW_SIZE
    )
    assert np.shape(X_train) == (len(train_chunks), WINDOW_SIZE,
                                 setup.get_num_tokens())
    assert np.shape(y_train) == (len(train_chunks), setup.get_num_tokens())


