from drain3 import TemplateMiner
import json
import sys
sys.path.append('../')
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain
import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

input_dir = '../data/'
in_log_file = "sample_test.log"
line_count = 0
with open(input_dir + in_log_file) as f:
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    logger.info(f"Drain3 started reading from {in_log_file}")

    for line in f:
        sess_id, anomaly_flag = adapter.get_session_id(log_msg=line)
        procid = re.search(r"^(\d+)", line)[0]
        content = line.split(procid)[1].strip()
        drain_result = drain.add_message(content)
        sessions = session_storage.get_sessions(sess_id, drain_result['template_id'])
        templates = session_storage.get_templates(drain_result['template_id'], drain_result['template'])
        parameters = session_storage.get_parameters(sess_id, drain_result['params'])
        line_count += 1
        if line_count % 10000 == 0:
            logger.info('Processed {} log lines.'.format(line_count))
    logger.info(f'Finished Drain3 process. Total of lines: {line_count}')
    normal_sessions, abnormal_sessions = session_storage.split_sessions(anomaly_flag)
    f.close()
#let's import the results in json format
result = {'data': []}
for sess_id in sessions:
    result['data'].append(dict(template_seq=sessions[sess_id], template_params=parameters[sess_id],
                               is_normal=anomaly_flag[sess_id], session_id=sess_id))

with open('data.json', 'w') as f:
    json.dump(result, f)
with open('templates.json', 'w') as g:
    json.dump(templates, g)

