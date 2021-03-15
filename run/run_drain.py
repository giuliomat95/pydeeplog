from drain3 import TemplateMiner
import json
import os
import sys
from deeplog_trainer.log_parser.adapter import AdapterFactory, ParseMethods
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain
import logging
import argparse
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_drain(logger, input_file, output_path):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_path, input_file), 'r') as f:
        adapter_factory = AdapterFactory()
        adapter = adapter_factory.instantiate_product(
            adapter_type=AdapterFactory.ADAPTER_TYPE_DELIMITER_AND_REGEX,
            delimiter='TCP source connection created',
            anomaly_labels=['TCP source SSL error', 'TCP source socket error'],
            regex=r"^(\d+)")
        template_miner = TemplateMiner()
        drain = Drain(template_miner)
        session_storage = SessionStorage()
        logger.info(f"Drain3 started reading from {args.input_file}")
        line_count = 0
        for line in f:
            sess_id, anomaly_flag = adapter.get_session_id(log=line)
            headers, regex = ParseMethods.generate_logformat_regex(
                logformat='<Pid>  <Content>')
            match = regex.search(line.strip())
            message = match.group('Content')
            drain_result = drain.add_message(message)
            sessions = session_storage.get_sessions(
                sess_id, drain_result['template_id'])
            templates = session_storage.get_templates(
                drain_result['template_id'], drain_result['template'])
            parameters = session_storage.get_parameters(sess_id,
                                                        drain_result['params'])
            line_count += 1
            if line_count % 1000 == 0:
                logger.info('Processed {} log lines.'.format(line_count))
        logger.info(f'Finished Drain3 process. Total of lines: {line_count}')
    # Import the results in json format
    result = {'data': []}
    for sess_id in sessions:
        result['data'].append(dict(template_seq=sessions[sess_id],
                                   template_params=parameters[sess_id],
                                   is_normal=anomaly_flag[sess_id],
                                   session_id=sess_id))

    with open(os.path.join(output_path, 'data.json'), 'w') as f:
        json.dump(result, f)
    with open(os.path.join(output_path, 'templates.json'), 'w') as g:
        json.dump(templates, g)
    # Save the Drain tree object in a pickle file
    with open(os.path.join(output_path, 'drain_tree.pickle'), 'wb') as h:
        pickle.dump(template_miner.drain.root_node, h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input filepath from root folder")
    parser.add_argument("--output_path", type=str,
                        help="Put the name of the directory where the results "
                             "will be saved")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    run_drain(logging.getLogger(__name__), args.input_file, args.output_path)
