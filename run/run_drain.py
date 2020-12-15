from drain3 import TemplateMiner
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain

input_dir = '../data/'
in_log_file = "sample_batrasio.log"
output_dir = 'drainResult/'



with open(input_dir + in_log_file) as f:
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    session_storage = SessionStorage()
    print(f"Drain3 started with reading from {in_log_file}")
    for line in f:
        sess_id, anomaly_flag = adapter.get_sessionId(log_msg=line)
        content = line.split('\t')[1]
        drain_result = drain.add_message(content)
        sessions = session_storage.get_sessions(sess_id, drain_result['template_id'])
        templates = session_storage.get_templates(drain_result['template_id'], drain_result['template'])
    normal_sessions, abnormal_sessions = session_storage.split_sessions(anomaly_flag)

print(sessions)
print(templates)
print(normal_sessions)
print(abnormal_sessions)