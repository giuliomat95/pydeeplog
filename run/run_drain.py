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

input_dir = '../data/'
in_log_file = "sample_batrasio.log"
output_dir = 'drainResult/'

def Drain(input_dir, in_log_file, output_dir):
    template_miner = TemplateMiner()
    extra_delimiters = template_miner.config.get('DRAIN', 'extra_delimiters', fallback='[]')
    with open(input_dir + in_log_file) as f:
        for line in f:
            line = line.rstrip()
            line = line.partition("\t")[2]
        template_miner.add_log_message(line)
        #print(json.dumps(drain_cluster))
    msg_templates = []
    templates_id =[]
    """cluster_id = drain_cluster['cluster_id']
    template = drain_cluster['template_mined']
    
    result = {
        'cluster_id': cluster_id,
        'template': template,
        'params': get_msg_params(log_line, template, extra_delimiters)
    }"""

    with open(output_dir+in_log_file+'_templates.csv', 'w') as f:
        for cluster in template_miner.drain.clusters:
            f.write('{}\n'.format(cluster.get_template()))
            templates_id.append(cluster.cluster_id)
            msg_templates.append(cluster.get_template())
    return msg_templates, templates_id