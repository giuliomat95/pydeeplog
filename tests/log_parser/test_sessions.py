import sys
sys.path.append('../../')
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage
import pytest

adapter = BatrasioAdapter()
template_miner = TemplateMiner()
drain = Drain(template_miner)
session_storage = SessionStorage()

with open('../../data/sample_test.log') as f:
    list_of_logs = [line.strip() for line in f]

@pytest.mark.parametrize("logs", list_of_logs)
def test_get_sessions(logs):
    sess_id, anomaly_flag = adapter.get_session_id(log_msg=logs)
    template_id = drain.add_message(logs)['template_id']
    sessions = session_storage.get_sessions(sess_id, template_id)
    assert len(sessions) == len(anomaly_flag)
