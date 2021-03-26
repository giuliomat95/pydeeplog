import logging
import sys
import os
import json
import argparse

from deeplog_trainer.workflow.workflow import WorkflowBuilder

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def run_workflows(logger, input_dir):
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    workflow_builder = WorkflowBuilder(logger)
    with open(os.path.join(root_path, input_dir, 'train_dataset.json'),
              'r') as fp:
        train_dataset = json.load(fp)['train_dataset']
    workflows = workflow_builder.build_workflows(train_dataset, verbose=1)
    return workflows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        help="Put the name of the directory to retrieve the"
                             "dataset from")
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    workflows = run_workflows(logger, args.input_dir)
    network = workflows['network']
    logger.info('Number of nodes created: {}'.format(len(network.get_nodes())))
