from deeplog_trainer.log_parser.adapter import SessionAdapter
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
from deeplog_trainer.log_parser.sessions import SessionStorage
import pytest
import re


@pytest.fixture(scope='session')
def setup():
    adapter = SessionAdapter(logformat='<Pid>  <Content>',
                             delimiter='TCP source connection created',
                             anomaly_labels=['TCP source SSL error',
                                             'TCP source socket error'],
                             regex=r"^(\d+)")
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    return adapter, drain, session_storage

def get_data():
    with open('data/sample_test_batrasio.log') as f:
        for line in f:
            yield line.strip()

@pytest.mark.parametrize("logs", get_data())
def test_dict(logs, setup):
    adapter, drain, session_storage = setup
    sess_id, anomaly_flag = adapter.get_session_id(log=logs)
    headers, regex = adapter.generate_logformat_regex()
    match = regex.search(logs.strip())
    message = match.group('Content')
    drain_result = drain.add_message(message)
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
