import pytest
from deeplog_trainer.workflow.build_workflow import Network, Node, RootNode


@pytest.fixture(scope='session')
def network():
    return Network()

def test_build_workflow(network):
    root_node = network.get_root_node()
    assert isinstance(root_node, RootNode)
    # Create a new node with a random value
    idx1 = network.add_node(value=3, is_start=True, is_end=False)
    assert idx1 == 1
    node1 = network.get_node(idx1)
    assert isinstance(node1, Node)
    assert isinstance(network.get_nodes(), dict)
    assert network.get_last_index() == 1
    assert node1.get_value() == 3
    # Test on the class Node
    assert isinstance(node1.get_network(), Network)
    # Add 3 other random nodes to the network and check they are all different
    # objects
    idx2 = network.add_node(value=5)
    node2 = network.get_node(idx2)
    idx3 = network.add_node(value=5)
    node3 = network.get_node(idx3)
    idx4 = network.add_node(value=4)
    node4 = network.get_node(idx4)
    # Add node2 to the children of node 1 and node4 to the children of node3
    node1.add_child(child_idx=idx2)
    node3.add_child(child_idx=idx4)
    assert node1.get_children() == {node2.get_value(): idx2}
    assert node3.get_children() == {node4.get_value(): idx4}
    # Add also node3 to the children of node1. Since node2 and node3 have the
    # same value the method add_child must combine node3 and node2 and return
    # the old idx (idx2)
    assert node1.add_child(child_idx=idx3) == idx2
    # Test node2 and node 3 are the same object
    assert network.get_node(idx2) is network.get_node(idx3)
    # Test the children of node3 added to the children of node2
    assert node2.get_children() == {node4.get_value(): idx4}
    # Add node2 and node3 as parents of node4. Since the 2 nodes are the same
    # object, only one node must be added
    node4.add_parents([idx2, idx3])
    assert len(node4.get_parents()) == 1
