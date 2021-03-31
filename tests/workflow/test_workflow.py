import pytest
from deeplog_trainer.workflow.workflow import WorkflowBuilder, WorkflowEvaluator
import logging as logger
import numpy as np

def get_dataset():
    train_dataset = [
        [1, 2, 8, 12, 10, 10, 9, 9, 16, 8, 13, 1, 15, 14, 9, 9, 16, 14],
        [1, 2, 3, 5, 12, 10, 13, 9, 14, 8, 10, 15, 1, 9, 14, 8, 18, 2],
        [1, 2, 6, 7, 2, 8, 9, 9, 10, 11, 8, 12, 10],
        [1, 4, 5, 12, 10, 8, 9, 8],
        [1, 2, 3, 5, 12, 10, 13, 9, 14, 8, 10, 15, 1, 9, 14, 8, 18, 2],
        [1, 2, 3, 4, 5, 3, 5, 20, 19, 1, 21, 15, 2],
        [1, 2, 8, 12, 10, 10, 9, 9, 16, 8, 13, 1, 15, 14, 9, 9, 16, 14]]
    expected_seqs = [
        [1, 2, 8, 12, 10, 10, 9, 9, 16, 8, 13, 1, 15, 14, 9, 9, 16, 14],
        [1, 2, 3, 5, 12, 10, 13, 9, 14, 8, 10, 15, 1, 9, 14, 8, 18, 2],
        [1, 2, 6, 7, 2, 8, 9, 9, 10, 11, 8, 12, 10],
        [1, 4, 5, 12, 10, 8, 9, 8],
        [1, 2, 3, 4, 5, 3, 5, 20, 19, 1, 21, 15, 2]]
    test_dataset = [
        [1, 4, 5, 12],
        [1, 3, 6, 7, 4, 2],
        [1, 4, 5, 12, 10, 8, 9, 8, 9]
    ]

    return train_dataset, expected_seqs, test_dataset

@pytest.mark.parametrize("train_dataset, threshold, expected_seqs",
                         [(get_dataset()[0], 0.2, get_dataset()[1])])
def test_build_workflows(train_dataset, threshold, expected_seqs):
    workflow_builder = WorkflowBuilder(logger)
    workflow = workflow_builder.build_workflows(train_dataset,
                                                threshold=threshold,
                                                back_steps=1)
    assert workflow['data'] == expected_seqs
    unique_values = []
    for node in workflow['network'].get_nodes().values():
        if node == workflow['network'].get_root_node():
            continue
        unique_values.append(node.get_value())
    unique_values = np.unique(unique_values)
    for seq in expected_seqs:
        for value in seq:
            assert value in unique_values

@pytest.mark.parametrize("train_dataset, test_dataset, threshold,"
                         "expected_matches, expected_accuracy, "
                         "expected_n_correct, expected_n_items",
                         [(get_dataset()[0], get_dataset()[2], 0.2,
                           [True, False, True], 2/3, 2, 3)])
def test_workflow_evaluator(train_dataset, test_dataset, threshold,
                            expected_matches, expected_accuracy,
                            expected_n_correct, expected_n_items):
    workflow_builder = WorkflowBuilder(logger)
    workflow = workflow_builder.build_workflows(train_dataset,
                                                threshold=threshold,
                                                back_steps=1)
    network = workflow['network']
    workflow_evaluator = WorkflowEvaluator(logger, network)
    matches = workflow_evaluator.evaluate(test_dataset)
    assert matches == expected_matches
    scores = workflow_evaluator.compute_scores(matches)
    assert scores['n_correct'] == expected_n_correct
    assert scores['n_items'] == expected_n_items
    assert scores['accuracy'] == expected_accuracy
