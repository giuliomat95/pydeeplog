import logging
import argparse
import os
import json

from deeplog_trainer.workflow.workflow import WorkflowBuilder, WorkflowEvaluator
from . import create_datasets


def run_workflows(logger, input_file, output_path, min_length, train_ratio,
                  val_ratio, threshold, back_steps):
    workflow_builder = WorkflowBuilder(logger)
    train_dataset, val_dataset, test_dataset, data_preprocess = create_datasets(
        logger, input_file, min_length, train_ratio, val_ratio)
    workflows = workflow_builder.build_workflows(train_dataset.tolist(),
                                                 threshold=threshold,
                                                 back_steps=back_steps)
    network = workflows['network']
    logger.info('Number of nodes created: {}'.format(len(network.get_nodes())))
    network_dict = {}
    for node_idx, node in network.get_nodes().items():
        network_dict[node_idx] = {'value': node.get_value(),
                                  'children': node.get_children(),
                                  'parents': node.get_parents(),
                                  'is_start': node.is_start(),
                                  'is_end': node.is_end()}
    workflow_evaluator = WorkflowEvaluator(logger, network_dict)
    matches = workflow_evaluator.evaluate(test_dataset)
    scores = workflow_evaluator.compute_scores(matches)
    logger.info('Workflow evaluation results: \n- accuracy: {}\n- n_correct: {}'
                '\n- n_items: {}'.format(scores['accuracy'],
                                         scores['n_correct'], scores['n_items'])
                )
    # Save workflows in json file
    with open(os.path.join(output_path, 'workflows.json'), 'w') as f:
        json.dump(network_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Put the input json dataset filepath from root "
                             "folder",
                        default='artifacts/drain_result/data.json')
    parser.add_argument("--output_path", type=str,
                        help="Put the path of the output directory",
                        default='artifacts/workflows')
    parser.add_argument("--min_length", type=int,
                        help="Put the minimum length of a sequence to be "
                             "parsed", default=4)
    parser.add_argument("--train_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "train set", default=0.7)
    parser.add_argument("--val_ratio", type=float,
                        help="Put the percentage of dataset size to define the"
                             "validation set", default=0.85)
    parser.add_argument("--threshold", type=float,
                        help="Put the similarity threshold", default=0.8)
    parser.add_argument("--back-steps", type=int,
                        help="Put the number of steps backwards to research"
                             "similar workflows", default=1)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except OSError as error:
        logger.error("Directory {} can not be created".format(args.output_path))
        exit(1)

    run_workflows(logger, args.input_file, args.output_path,
                  args.min_length, args.train_ratio, args.val_ratio,
                  args.threshold, args.back_steps)
