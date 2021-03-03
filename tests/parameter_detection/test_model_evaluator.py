import pytest
import numpy as np
import logging as logger
from sklearn.preprocessing import MinMaxScaler
from deeplog_trainer.parameter_detection.model_evaluator import ModelEvaluator

def get_data():
    y_true = np.random.rand(15, 3)
    y_pred = np.random.rand(15, 3)
    yield y_true, y_pred

@pytest.mark.parametrize("y_true, y_pred", get_data())
def test_model_evaluator(mocker, y_true, y_pred):
    model_evaluator = ModelEvaluator(logger)
    sc_X = MinMaxScaler()
    label = 'Test plot'
    timestamps = np.arange(len(y_pred))
    # Mock inverse_transform method of the class MinMaxScaler in sklearn
    mock_scaler = mocker.patch(
        'sklearn.preprocessing.MinMaxScaler.inverse_transform')
    # Random transformation
    mock_scaler.side_effect = [y_true*10, y_pred*10]
    # Mock pyplot.show of the library matplotlib
    mocker.patch('matplotlib.pyplot.show')
    model_evaluator.plot_time_series(sc_X, timestamps, y_true, y_pred, label)
    # Mock norm.interval method of the library scipy.stats and return a random
    # list with two numbers between -1 and 1
    mocked_interval = [np.random.uniform(-1, 0), np.random.uniform(0, 1)]
    mock_stats_normal = mocker.patch('scipy.stats.norm.interval')
    mock_stats_normal.return_value = mocked_interval
    interval, mse_val = model_evaluator.build_confidence_intervals(y_true,
                                                                   y_pred)
    assert interval == mocked_interval
    assert len(mse_val) == len(y_true)
    model_evaluator.plot_confidence_intervals(interval, mse_val)
    anomalies_idx, mse_test = model_evaluator.get_anomalies_idx(y_true, y_pred,
                                                                interval)
    for idx in anomalies_idx:
        assert mse_test[idx] > interval[1] or mse_test[idx] < interval[0]
    model_evaluator.plot_test_errors(interval, mse_test)
