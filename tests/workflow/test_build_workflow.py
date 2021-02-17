import pytest
from deeplog_trainer.workflow.build_workflow import WorkflowBuilder
from deeplog_trainer.workflow.build_network import Node, Network
import json
import numpy as np
import logging as logger


def get_dataset(min_length=4):
    filepath = 'data/data.json'
    dataset = []
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        for d in data['data']:
            seq = d['template_seq']
            if len(seq) < min_length:
                # Skip short sequences
                continue
            dataset.append(seq)
        return dataset

@pytest.mark.parametrize("dataset, threshold, verbose",
                         [(get_dataset(), 0.2, 1)])
def test_build_workflows(dataset, threshold, verbose):
    workflow_builder = WorkflowBuilder(logger)
    workflow = workflow_builder.build_workflows(dataset, threshold=threshold,
                                                verbose=verbose, back_steps=1)
    assert 6 in workflow['network'].get_node(5).get_children(only_node_idx=True)

