import pytest
from tests.model.mocks import MockModel
from deeplog_trainer.model.model_evaluator import ModelEvaluator
from deeplog_trainer.model.data_preprocess import DataPreprocess
import json
import numpy as np

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
        yield dataset

@pytest.mark.parametrize("dataset", get_data())
def test_predict(setup, dataset):
    WINDOW_SIZE = 7
    data_preprocess = setup
    dataset = np.array(dataset, dtype=object)
    dataset = data_preprocess.encode_dataset(dataset)
    train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=0.7,
                                  val_ratio=0.85)
    val_dataset = dataset[val_idx]
    X_val, y_val = data_preprocess.transform(
        data_preprocess.chunks(val_dataset, window_size=WINDOW_SIZE),
        add_padding=WINDOW_SIZE
    )
    mocked_model = MockModel(predict_output=
                             np.array([[0.15, 0, 0.05, 0, 0.1, 0, 0.7, 0],
                                      [0.1, 0.4, 0, 0, 0.3, 0, 0.2, 0]]))
    model_evaluator = ModelEvaluator(mocked_model, top_k=4)
    assert (model_evaluator.predict(X_val) == np.array([[2, 4, 0, 6],
                                                       [0, 6, 4, 1]])).all()





