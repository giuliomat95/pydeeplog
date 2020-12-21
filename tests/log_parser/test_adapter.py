from deeplog_trainer.log_parser.adapter import BatrasioAdapter
import pytest

@pytest.fixture(scope='session')
def adapter():
    return BatrasioAdapter()

def get_data():
    expected_sess_id = [1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3]
    with open('data/sample_test.log') as f:
        for i, line in enumerate(f):
            yield (line.strip(), expected_sess_id[i])

@pytest.mark.parametrize("logs,expected", get_data())
def test_get_sessionId(logs, expected, adapter):
    sess_id, anomaly_flag = adapter.get_session_id(log_msg=logs)
    assert sess_id == expected
    assert isinstance(anomaly_flag[sess_id], bool)
