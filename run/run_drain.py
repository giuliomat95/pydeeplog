import json
import re
from drain3 import TemplateMiner


def tokenizer(msg, extra_delimiters=[]):
    msg = msg.strip()
    for delimiter in extra_delimiters:
        msg = msg.replace(delimiter, " ")
    msg_tokens = msg.split()
    return msg_tokens

def get_msg_params(msg, template, extra_delimiters=[]):
    print()
    print(msg)
    print(template)
    msg = tokenizer(msg, extra_delimiters)
    template = tokenizer(template)
    print(msg)
    print(template)
    print(len(msg), len(template))
    assert len(msg) == len(template)

    params = []
    for i in range(len(msg)):
        if re.search(r"^\<.+?\>$", template[i]):
            params.append(msg[i])

    return params



template_miner = TemplateMiner()
log_file = [
    'connector=52541013: Closing multi target connection',
    'connector=52540976, kind=unpiped, path=52.86.47.45:58170<->10.11.24.33:443<=>10.11.24.33:41904<->172.17.12.211:660: Connector closed',
    'connector=52541014, kind=unpiped, path=52.19.12.121:49440<->10.11.24.33:1515<=>10.11.24.33:54193<->172.17.12.65:1515: Connector closed',
]
extra_delimiters = template_miner.config.get('DRAIN', 'extra_delimiters', fallback='[]')

for log_line in log_file:
    drain_cluster = template_miner.add_log_message(log_line)

    """cluster_id = drain_cluster['cluster_id']
    template = drain_cluster['template_mined']
    
    result = {
        'cluster_id': cluster_id,
        'template': template,
        'params': get_msg_params(log_line, template, extra_delimiters)
    }"""
    print(json.dumps(drain_cluster))

print('-' * 10)
print("Clusters:")
for cluster in template_miner.drain.clusters:
    print(cluster)
