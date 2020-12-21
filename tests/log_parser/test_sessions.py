from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage
import pytest

@pytest.fixture(scope='session')
def setup():
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    return adapter, drain, session_storage

def get_data():
    with open('../../data/sample_test.log') as f:
        for line in f:
            yield line.strip()

@pytest.mark.parametrize("logs", get_data())
def test_get_sessions(logs, setup):
    adapter, drain, session_storage = setup
    sess_id, anomaly_flag = adapter.get_session_id(log_msg=logs)
    template_id = drain.add_message(logs)['template_id']
    sessions = session_storage.get_sessions(sess_id, template_id)
    assert len(sessions) == len(anomaly_flag)
