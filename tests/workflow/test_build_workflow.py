import pytest
from deeplog_trainer.workflow.build_workflow import WorkflowBuilder
import json
import numpy as np
import logging as logger
import pdb


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
def test_get_similar_seqs(dataset, threshold, verbose):
    wb = WorkflowBuilder(logger)
    max_length = max([len(s) for s in dataset])
    similar_seqs = wb._get_similar_sequences(dataset, threshold, verbose)
    assert len(similar_seqs) == len(dataset)
    assert len(similar_seqs[0][0]) == max_length
    for seqs in similar_seqs:
        if len(seqs) == 1:
            continue
        original_seqs = seqs
        seqs = np.random.rand(len(seqs), max_length)
        for i in range(len(seqs)):
            seqs[i, :len(original_seqs[i])] = original_seqs[i]
        seqs_ref = np.repeat([seqs[0]], len(seqs)-1, axis=0)
        seqs_compare = seqs[1:]
        scores = np.sum(seqs_ref == seqs_compare, axis=-1)/max_length
        assert all(scores >= threshold) is True
