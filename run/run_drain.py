from drain3 import TemplateMiner
import json
import sys
sys.path.append('../')
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain

input_dir = '../data/'
in_log_file = "sample.log"

with open(input_dir + in_log_file) as f:
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    print(f"Drain3 started with reading from {in_log_file}")
    count = 0
    for line in f:
        sess_id, anomaly_flag = adapter.get_sessionId(log_msg=line)
        content = line.split('\t')[1]
        drain_result = drain.add_message(content)
        sessions = session_storage.get_sessions(sess_id, drain_result['template_id'])
        templates = session_storage.get_templates(drain_result['template_id'], drain_result['template'])
        parameters = session_storage.get_parameters(sess_id, drain_result['params'])
        count += 1
        if count % 1000 == 0:
            print('Processed {} log lines.'.format(count))
    print('Finished Drain3 process')
    normal_sessions, abnormal_sessions = session_storage.split_sessions(anomaly_flag)
    f.close()
#let's import the results in json format
result = {'data': []}
for sess_id in sessions:
    result['data'].append(dict(template_seq=sessions[sess_id], template_params=parameters[sess_id],
                               is_normal=anomaly_flag[sess_id], session_id=sess_id))

json_dataset = json.dumps(result)
json_templates = json.dumps(templates)


