from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
import pytest
import re

@pytest.fixture(scope='session')
def drain():
    template_miner = TemplateMiner()
    return Drain(template_miner)

def get_data():
    with open('data/sample_test.log') as f:
        for line in f:
            yield line.strip()

@pytest.mark.parametrize("logs", get_data())
def test_get_parameters(logs, drain):
    result = drain.add_message(logs)
    template = result['template']
    params = result['params']
    assert len(re.findall(r'<[^<>]+>', template)) == len(params)