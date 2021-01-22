from deeplog_trainer.model.model_manager import ModelManager
import pytest

@pytest.fixture(scope='class')
def setup():
    window_size = 10
    model_manager = ModelManager(input_size=window_size, lstm_units=64,
                                 num_tokens=27)
    return model_manager

def test_model_manager(setup):
    model = setup.build()
    assert model.output_shape[1] == setup.num_tokens

