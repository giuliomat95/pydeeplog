import sys
sys.path.append('../../')
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.drain import Drain
import pytest
import re

template_miner = TemplateMiner()
drain = Drain(template_miner)
with open('../../data/sample_test.log') as f:
    list_of_logs = [line.strip() for line in f]

@pytest.mark.parametrize("logs", list_of_logs)
def test_get_parameters(logs):
    result = drain.add_message(logs)
    template = result['template']
    params = result['params']
    assert len(re.findall(r'<[^<>]+>', template)) == len(params)