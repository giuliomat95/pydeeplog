import pytest
from deeplog_trainer.model.training import ValLossLogger, ModelTrainer
from deeplog_trainer.model.data_preprocess import DataPreprocess
import logging as logger
from tests.model.mocks import MockModel
import json
import numpy as np
from testfixtures import LogCapture


@pytest.fixture(scope='function')
def setup(dataset):
    vocab = list(set([x for seq in dataset for x in
                      seq]))  # list of unique keys in the training file
    data_preprocess = DataPreprocess(vocab=vocab)
    train_logger = ValLossLogger(logger)
    model_trainer = ModelTrainer(logger, epochs=50, early_stop=7,
                                 batch_size=512)
    return data_preprocess, train_logger, model_trainer

@pytest.fixture(autouse=True)
def capture():
    with LogCapture() as capture:
        yield capture

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
        yield dataset

@pytest.mark.parametrize("dataset", get_data())
def test_model_trainer(setup, dataset, capture):
    WINDOW_SIZE = 7
    data_preprocess, train_logger, model_trainer = setup
    dataset = np.array(dataset, dtype=object)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=0.7,
                                  val_ratio=0.85)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    X_train, y_train = data_preprocess.transform(
        data_preprocess.chunks(train_dataset, window_size=WINDOW_SIZE),
        add_padding=WINDOW_SIZE
    )
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=WINDOW_SIZE),
        add_padding=WINDOW_SIZE
    )
    mocked_model = MockModel()
    train_logger.on_train_begin()
    history = model_trainer.train(mocked_model, [X_train, y_train],
                                  [X_val, y_val])
    train_logger.on_train_end()
    capture.check(
        ('root', 'INFO', 'Start training'),
        ('root', 'INFO', 'Loss: inf (acc.: 0.0000) - Val. loss:  inf '
                         '(acc.: 0.0000)'),
        ('root', 'INFO', 'Training finished')
    )
    assert isinstance(history, MockModel)
