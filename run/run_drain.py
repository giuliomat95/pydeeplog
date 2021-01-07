from drain3 import TemplateMiner
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain
import re
import logging
import argparse
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
input_dir = '/data/'


def run_drain(logger, input_file, output_file):
    with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + input_dir + input_file) as f:
        adapter = BatrasioAdapter()
        template_miner = TemplateMiner()
        drain = Drain(template_miner)
        session_storage = SessionStorage()
        logger.info(f"Drain3 started reading from {args.input}")
        line_count = 0
        for line in f:
            sess_id, anomaly_flag = adapter.get_session_id(log_msg=line)
            procid = re.search(r"^(\d+)", line)[0]
            content = line.split(procid)[1].strip()
            drain_result = drain.add_message(content)
            sessions = session_storage.get_sessions(sess_id, drain_result['template_id'])
            templates = session_storage.get_templates(drain_result['template_id'], drain_result['template'])
            parameters = session_storage.get_parameters(sess_id, drain_result['params'])
            line_count += 1
            if line_count % 1000 == 0:
                logger.info('Processed {} log lines.'.format(line_count))
        logger.info(f'Finished Drain3 process. Total of lines: {line_count}')
        f.close()
    # let's import the results in json format
    result = {'data': []}
    for sess_id in sessions:
        result['data'].append(dict(template_seq=sessions[sess_id], template_params=parameters[sess_id],
                                   is_normal=anomaly_flag[sess_id], session_id=sess_id))

    with open(output_file + '/data.json', 'w') as f:
        json.dump(result, f)
        f.close()
    with open(output_file + '/templates.json', 'w') as g:
        json.dump(templates, g)
        g.close()
    # let's save the Drain tree object in a pickle file
    with open(output_file + '/drain_tree.pickle', 'wb') as h:
        pickle.dump(template_miner.drain.root_node, h)
        h.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Put the name of the file to be parsed")
    parser.add_argument("--output", type=str, help="Put the name of the directory where the results will be saved")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run_drain(logging.getLogger(__name__), args.input, args.output)