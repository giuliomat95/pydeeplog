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
    def get_sessionId(self, log_msg: str):
        log_msg = log_msg.rstrip()
        procid = re.search(r"^(\d+)", log_msg)[0]
        if procid not in self.d.keys():
            self.d[procid] = self.last_session_id + 1
        delimiter = 'TCP source connection created'
        if delimiter in log_msg:
            self.d[procid] = self.last_session_id + 1
            self.last_session_id += 1

        return self.d[procid]

class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner
    def add_message(self, msg):
        msg = msg.rstrip()
        cluster = self.template_miner.add_log_message(msg)
        template = cluster['template_mined']
        template_id = cluster['cluster_id']
        #cluster = self.template_miner.drain.clusters[0]
        #template = cluster.get_template()
        #template_id = cluster.cluster_id
        template_regex = re.sub(r"<.{1,5}>", "<*>", template)
        if "<*>" not in template_regex:
            result = {'template_id': template_id,
                      'template': template,
                      'params': []}
            return result
        else:
            template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
            template_regex = re.sub(r'\\\s+', r' ', template_regex)  # replace any '\ ' by ' '
            template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
            parameter_list = re.findall(template_regex, msg)
            parameter_list = parameter_list[0] if parameter_list else ()
            parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
            result = {
                'template_id': template_id,
                'template': template,
                'params': parameter_list
            }

        return result

with open(input_dir + in_log_file) as f:
    adapter = BatrasioAdapter()
    template_miner = TemplateMiner()
    drain = Drain(template_miner)
    print(f"Drain3 started with reading from {in_log_file}")
    for line in f:

        sess_id = adapter.get_sessionId(log_msg=line)
        content = line.split('\t')[1]
        print(content)
        result = drain.add_message(content)
        print('---------------')
        print(result['template_id'])
        print(result['template'])
        print(result['params'])
        print('---------------')