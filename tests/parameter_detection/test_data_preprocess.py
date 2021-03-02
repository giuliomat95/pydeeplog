import pytest
import json
import numpy as np
from deeplog_trainer.parameter_detection.data_preprocess import DataPreprocess

def get_dataset():
    filepath = 'data/dataset.json'
    with open(filepath, 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    return dataset

@pytest.mark.parametrize("dataset, train_ratio, val_ratio, window_size",
                         [(get_dataset(), 0.7, 0.85, 5)])
def test_data_preprocess(dataset, train_ratio, val_ratio, window_size):
    data_preprocess = DataPreprocess()
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio, val_ratio)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    assert len(train_dataset) == int(len(dataset) * train_ratio)
    assert len(train_dataset) + len(val_dataset) == int(
        len(dataset) * val_ratio)
    X_train, y_train = data_preprocess.generate(train_dataset, window_size)
    assert X_train.shape == (len(train_dataset)-window_size+1, window_size,
                             dataset.shape[1])
    assert y_train.shape == (len(train_dataset)-window_size+1, dataset.shape[1])
    for i in range(len(X_train)):
        assert X_train[i, -1, 0] == y_train[i, 0]
