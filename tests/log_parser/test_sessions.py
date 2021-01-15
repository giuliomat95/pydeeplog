from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage
import pytest
import re


@pytest.fixture(scope='session')
def setup():
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    return adapter, drain, session_storage


def get_data():
    with open('data/sample_test.log') as f:
        for line in f:
            yield line.strip()


@pytest.mark.parametrize("logs", get_data())
def test_dict(logs, setup):
    adapter, drain, session_storage = setup
    sess_id, anomaly_flag = adapter.get_session_id(log_msg=logs)
    procid = re.search(r"^(\d+)", logs)[0]
    content = logs.split(procid)[1].strip()
    drain_result = drain.add_message(content)
    sessions = session_storage.get_sessions(
        sess_id, drain_result['template_id'])
    parameters = session_storage.get_parameters(sess_id, drain_result['params'])
    templates = session_storage.get_templates(drain_result['template_id'],
                                              drain_result['template'])
    # The 3 dictionaries must have the same length equivalent to the number of
    # sessions
    assert len(sessions) == len(parameters) == len(anomaly_flag)
    # The templates Id must be integers
    assert set(map(type, templates)) == {int}
