import sys
sys.path.append('../../')
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
import pytest

adapter = BatrasioAdapter()
expected_sess_id = [1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3]
with open('../../data/sample_test.log') as f:
        list_of_tuples = [(line.strip(), expected_sess_id[i]) for i, line in enumerate(f)]

@pytest.mark.parametrize("logs,expected", list_of_tuples)
def test_get_sessionId(logs, expected):
    sess_id, anomaly_flag = adapter.get_session_id(log_msg=logs)
    assert sess_id == expected
    assert isinstance(anomaly_flag[sess_id], bool)
