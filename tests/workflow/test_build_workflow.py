import pytest
from deeplog_trainer.workflow.build_workflow import WorkflowBuilder
import logging as logger


def get_dataset():
    test_dataset = [
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
    return test_dataset, expected_seqs

@pytest.mark.parametrize("test_dataset, threshold, verbose, expected_seqs",
                         [(get_dataset()[0], 0.2, 1, get_dataset()[1])])
def test_build_workflows(test_dataset, threshold, verbose, expected_seqs):
    workflow_builder = WorkflowBuilder(logger)
    workflow = workflow_builder.build_workflows(test_dataset,
                                                threshold=threshold,
                                                verbose=verbose,
                                                back_steps=1)

    assert workflow['data'] == expected_seqs
    assert expected_seqs[0][0] in \
           workflow['network'].get_root_node().get_children()
