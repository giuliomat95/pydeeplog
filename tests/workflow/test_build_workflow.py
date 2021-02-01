import pytest
from deeplog_trainer.workflow.build_workflow import Network, Node, RootNode


@pytest.fixture(scope='session')
def network():
    return Network()

def get_data():
    nodes_data = [dict(value=3, is_start=True), dict(value=5), dict(value=5),
                  dict(value=4, is_end=True)]
    for node_data in nodes_data:
        yield node_data

@pytest.mark.parametrize("node_data", get_data())
def test_build_workflow(network, node_data):
    # Test state of the network when it is empty
    if network.get_last_index() == 0:
        root_node = network.get_root_node()
        assert isinstance(root_node, RootNode)
        assert network.get_nodes() == {0: root_node}
        # Raise an error if you try to add a child with an unknown index Node
        with pytest.raises(Exception):
            root_node.add_child(child_idx=5)
    # Create a new node
    last_index = network.get_last_index()
    idx = network.add_node(**node_data)
    assert idx - last_index == 1
    node = network.get_node(idx)
    assert isinstance(node, Node)
    assert list(network.get_nodes().keys()) == list(range(idx+1))
    assert node.get_value() == node_data['value']
    assert node.get_idx() == idx
    # The first node starts the sequence
    if idx == 1:
        assert node.is_start() is True
    # Test on the class Node
    assert isinstance(node.get_network(), Network)
    # Add node2 to the children of node 1
    if idx == 2:
        network.get_node(1).add_child(child_idx=idx)
        assert network.get_node(1).get_children() == {node.get_value(): idx}
    # Add also node3 to the children of node1. Since node2 and node3 have the
    # same value the method add_child must combine node3 and node2 and return
    # the old idx (2)
    if idx == 3:
        assert network.get_node(1).add_child(child_idx=idx) == 2
        # Test node2 and node 3 are the same object
        assert network.get_node(2) is network.get_node(idx)
    if idx == 4:
        # The fourth Node ends the sequence
        assert node.is_end() is True
        # Add node4 to the children of node3
        network.get_node(3).add_child(child_idx=idx)
        assert network.get_node(3).get_children() == {node.get_value(): idx}
        # Test the children of node3 added to the children of node2
        assert network.get_node(2).get_children() == {node.get_value(): idx}
        # Add node2 and node3 as parents of node4. Since the 2 nodes are the
        # same object, only one node must be added
        node.add_parents([2, 3])
        assert len(node.get_parents()) == 1
        # Raise an error if you try to combine two nodes with different value
        with pytest.raises(Exception):
            network.get_node(3).combine(node)

