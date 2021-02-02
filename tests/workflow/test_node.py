import pytest
from deeplog_trainer.workflow.build_workflow import Node, RootNode


def get_data():
    nodes_data = [dict(value=3, is_start=True, is_end=False, idx=1,
                       parent_idx=0),
                  dict(value=5, is_start=False, is_end=False, idx=2,
                       parent_idx=2),
                  dict(value=5, is_start=False, is_end=False, idx=3,
                       parent_idx=1),
                  dict(value=4, is_start=False, is_end=True, idx=4,
                       parent_idx=1)]
    for node_data in nodes_data:
        yield node_data


@pytest.mark.parametrize("node_data", get_data())
def test_node_get_methods(mocker, node_data):
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.build_workflow.Network',
                     autospec=True)
    node = Node(mocked_network, **node_data)
    assert node.get_value() == node_data['value']
    assert node.get_network() == mocked_network
    assert node.get_idx() == node_data['idx']
    assert node.is_start() == node_data['is_start']
    assert node.is_end() == node_data['is_end']
    assert node.get_parents() == [node_data['parent_idx']]


@pytest.mark.parametrize(
    "value, is_start, is_end, idx, parents_idx",
    [(2, True, False, 2, [0, 1])]
)
def test_node_set_methods(mocker, value, is_start, is_end, idx, parents_idx):
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.build_workflow.Network',
                     autospec=True)
    node = Node(mocked_network, value)
    node.set_start(is_start)
    assert node.is_start() is is_start
    node.set_end(is_end)
    assert node.is_end() is is_end
    node.set_idx(idx)
    assert node.get_idx() == idx
    # Mock method get_node()
    root_node = RootNode(mocked_network, idx=0)
    # random node to add to thh mocked network
    sample_node = Node(mocked_network, 3, idx=1)
    mocked_network.get_node.side_effect = [{parents_idx[0]: root_node},
                                           {parents_idx[1]: sample_node}]
    node.add_parents(parents_idx)
    assert node.get_parents() == parents_idx


@pytest.mark.parametrize(
    "value, is_start, idx",
    [(3, True, 1)]
)
def test_node_add_children(mocker, value, is_start, idx):
    mocked_network = \
        mocker.patch('deeplog_trainer.workflow.build_workflow.Network',
                     autospec=True)
    instance = mocked_network.return_value
    node = Node(instance, value)
    instance.get_nodes.return_value = {2: Node(instance, value=5),
                                       3: Node(instance, value=5)}
    # Raise an error if you try to add a child with an unknown index Node
    with pytest.raises(Exception):
        node.add_child(child_idx=5)
    my_side_effect = lambda index: Node(instance, value=5, idx=index)
    with mocker.patch.object(instance, 'get_node', side_effect=my_side_effect):
        node.add_children([2, 3])
        # Since the 2 nodes have the same value the method add_child must
        # combine node3 and node2 and return the old idx
        assert node.get_children() == {5: 2}

