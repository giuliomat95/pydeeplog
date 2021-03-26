import logging
import argparse

from deeplog_trainer.workflow.workflow import WorkflowBuilder
from . import create_datasets


def run_workflows(logger, input_file, min_length, train_ratio, val_ratio):
    workflow_builder = WorkflowBuilder(logger)
    train_dataset, val_dataset, test_dataset, data_preprocess = create_datasets(
        logger, input_file, min_length, train_ratio, val_ratio)
    workflows = workflow_builder.build_workflows(train_dataset.tolist(),
                                                 verbose=1)
    return workflows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input json dataset filepath from root "
                             "folder",
                        default='artifacts/drain_result/data.json')
    parser.add_argument("--min_length", type=int,
                        help="Put the minimum length of a sequence to be "
                             "parsed", default=4)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.7)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             " validation set", default=0.85)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    workflows = run_workflows(logger, args.input_file, args.min_length,
                              args.train_ratio, args.val_ratio)
    network = workflows['network']
    logger.info('Number of nodes created: {}'.format(len(network.get_nodes())))
