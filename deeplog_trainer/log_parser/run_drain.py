import json
import re
from drain3 import TemplateMiner

input_dir = '../../data/'
in_log_file = "sample.log"
output_dir = 'drainResult/'

class BatrasioAdapter:
    def __init__(self):
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}

    def get_sessionId(self, log_msg: str):
        log_msg = log_msg.rstrip()
        procid = re.search(r"^(\d+)", log_msg)[0]
        if procid not in self.d.keys():
            self.d[procid] = self.last_session_id + 1
            self.anomaly_flag[self.d[procid]] = False
        delimiter = 'TCP source connection created'
        anomaly1 = 'TCP source SSL error'
        anomaly2 = 'TCP source socket error'
        if delimiter in log_msg:
            self.d[procid] = self.last_session_id + 1
            self.last_session_id += 1
            self.anomaly_flag[self.d[procid]] = False
        if anomaly1 in log_msg or anomaly2 in log_msg:
            self.anomaly_flag[self.d[procid]] = True
        return self.d[procid], self.anomaly_flag

class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner

    def get_parameters(self, content, template):
        template_regex = re.sub(r"<.{1,5}>", "<*>", template)
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r' ', template_regex)  # replace any '\ ' by ' '
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, content)
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    @staticmethod
    def clusterId_to_num(cluster_id):
        num = int(cluster_id[1:])-1
        return num

    def add_message(self, msg):
        msg = msg.rstrip()
        cluster = self.template_miner.add_log_message(msg)
        template = cluster['template_mined']
        template_id = cluster['cluster_id']
        template_id = self.clusterId_to_num(template_id)
        parameter_list = self.get_parameters(msg, template)
        result = {
                'template_id': template_id,
                'template': template,
                'params': parameter_list
                }
        return result

class SessionStorage:
    def __init__(self):
        self.sessions = {}
        self.templates = {}
        self.normal_sessions = {}
        self.abnormal_sessions = {}

    def get_sessions(self, sess_id, template_id):
        if sess_id not in self.sessions.keys():
            self.sessions[sess_id] = [template_id]
        else:
            self.sessions[sess_id].append(template_id)
        return self.sessions

    def get_templates(self, template_id, template):
        self.templates[template_id] = template
        return self.templates

    def split_sessions(self, anomaly_flag):
        for i in range(1, len(sessions)+1):
            if anomaly_flag[i] is False:
                self.normal_sessions[i] = self.sessions[i]
            else:
                self.abnormal_sessions[i] = self.sessions[i]
        return self.normal_sessions, self.abnormal_sessions




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