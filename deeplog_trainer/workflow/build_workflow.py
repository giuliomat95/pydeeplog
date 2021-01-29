class Network:
    def __init__(self):
        self._root_node = RootNode(self, idx=0)
        self._nodes = {0: self._root_node}
        self._last_idx = 0

    def get_new_idx(self):
        return self._last_idx + 1

    def get_root_node(self):
        return self._root_node

    def get_nodes(self):
        return self._nodes

    def get_node(self, idx):
        return self._nodes[idx]

    def get_last_index(self):
        return self._last_idx

    def add_node(self, value, is_start=False, is_end=False, parent_idx=None):
        """
        Given in input the properties of a Node, it adds it to the collection of
        nodes and returns the newer index
        """
        idx = self.get_new_idx()
        node = Node(self, value, is_start=is_start, is_end=is_end,
                    parent_idx=parent_idx, idx=idx)
        self._nodes[idx] = node
        self._last_idx += 1
        return idx

    def replace_node(self, replace_idx, by_idx):
        self._nodes[replace_idx] = self._nodes[by_idx]


class Node:
    def __init__(self, network: Network, value, is_start=False,
                 is_end=False, parent_idx=None, idx=None):
        """
        Attributes:
        :param network: Object of type Network Class
        :param value: the value of the node. 
        :param is_start: boolean variable to set if the Node starts the workflow
        :param is_end: boolean variable to set if the Node ends the workflow
        :param parent_idx: index nodes of the parents
        :param idx: index of the Node
        """
        self._network = network
        self._idx = idx
        self._value = value
        self._is_start = is_start
        self._is_end = is_end
        self._parents = [] if parent_idx is None else [parent_idx]
        # Dictionary of children nodes
        self._children = {}

    def get_network(self):
        return self._network

    def get_idx(self):
        return self._idx

    def set_idx(self, idx):
        self._idx = idx

    def get_value(self):
        return self._value

    def is_start(self):
        return self._is_start

    def set_start(self, is_start):
        self._is_start = is_start

    def is_end(self):
        return self._is_end

    def set_end(self, is_end):
        self._is_end = is_end

    def get_parents(self):
        return self._parents

    def set_parents(self, parents):
        self._parents = parents

    def get_children(self, only_node_idx=False):
        return list(
            self._children.values()) if only_node_idx else self._children

    def set_children(self, children_idx):
        self._children_idx = children_idx

    def add_parents(self, parents_idx):
        for idx in parents_idx:
            self.add_parent(idx)
        return parents_idx

    def add_parent(self, parent_idx):
        exists = [x for x in self._parents if
                  self._network.get_node(x) is self._network.get_node(
                      parent_idx)]
        if len(exists) == 0:
            self._parents.append(parent_idx)
            return parent_idx

    def add_children(self, children_idx):
        for idx in children_idx:
            self.add_child(idx)
        return children_idx

    def add_child(self, child_idx):
        child = self._network.get_node(child_idx)
        if child.get_value() in self._children:
            old_idx = self._children[child.get_value()]
            old_child = self._network.get_node(old_idx)
            # Combine if they are different nodes
            if old_child.get_idx() != child.get_idx():
                self._network.replace_node(child_idx, old_idx)
                old_child.combine(child)
            return old_idx
        else:
            self._children[child.get_value()] = child_idx
            return child_idx

    def combine(self, node):
        """
        :param node: object of type Node
        """
        if node.get_value() != self._value:
            raise Exception('Nodes cannot be combined: values are different!')
        elif node.get_idx() == self._idx:
            # Skip combinations of the same node
            return
        # self._network.replace_node(node.get_idx(), self._idx)
        self._is_start = self._is_start or node.is_start()
        self._is_end = self._is_end or node.is_end()
        self.add_parents(node.get_parents())
        self.add_children(node.get_children(only_node_idx=True))


class RootNode(Node):
    def __init__(self, network, idx=None):
        super().__init__(network, value=None, is_start=False, is_end=False,
                         parent_idx=None, idx=idx)



