import pytest
from deeplog_trainer.workflow.build_workflow import Node, RootNode, Network
import inspect


def get_data():
    # The function provides a tuple of data and expected values for each test
    return [(None, 0, None, False, False, []),
            ([dict(value=1, is_start=True, parent_idx=0), dict(value=5),
              dict(value=5, parent_idx=1),
              dict(value=4, is_end=True, parent_idx=1), dict(value=7)],
             [1, 2, 3, 4, 5],
             [1, 5, 5, 4, 7],
             [True, False, False, False, False],
             [False, False, False, True, False],
             [[0], [], [1], [1], []])]

@pytest.mark.parametrize("nodes_data, expected_new_idx, expected_value, "
                         "expected_start, expected_end, expected_parent_idx",
                         get_data())
def test_build_workflow(nodes_data, expected_new_idx, expected_value,
                        expected_start, expected_end, expected_parent_idx):
    network = Network()
    assert isinstance(network.get_root_node(), RootNode)
    # Test Network methods in 2 cases: when the class in empty (no node added)
    # and when few nodes are added
    if nodes_data is None:
        assert network.get_last_index() == expected_new_idx
        assert network.get_node(expected_new_idx) is network.get_root_node()
        assert network.get_root_node().get_value() == expected_value
        assert network.get_root_node().get_parents() == expected_parent_idx
        assert network.get_root_node().is_start() == expected_start
        assert network.get_root_node().is_end() == expected_end
        assert network.get_nodes() == \
               {expected_new_idx: network.get_root_node()}
    else:
        # Add all the nodes to the network
        for i, node_data in enumerate(nodes_data):
            network.add_node(**node_data)
            assert network.get_last_index() == expected_new_idx[i]
        for i in range(len(nodes_data)):
            node = network.get_node(expected_new_idx[i])
            assert node.get_value() == expected_value[i]
            assert node.is_start() == expected_start[i]
            assert node.is_end() == expected_end[i]
            assert node.get_parents() == expected_parent_idx[i]
        assert len(network.get_nodes()) == len(nodes_data)+1
