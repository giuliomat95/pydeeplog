class Node:
    def __init__(self, network, value, is_start=False, is_end=False,
                 parent_idx=None, idx=None):
        self._network = network
        self._idx = idx
        self._value = value
        self._is_start = is_start
        self._is_end = is_end
        self._parents = [] if parent_idx is None else [parent_idx]
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
    return list(self._children.values()) if only_node_idx else self._children

def set_children(self, children_idx):
    self._children_idx = children_idx

def add_parents(self, parents_idx):
    added = []
    for idx in parents_idx:
        self.add_parent(idx)
        added.append(idx)
    return added

def add_parent(self, parent_idx):
    exists = [x for x in self._parents if
              self._network.get_node(x) is self._network.get_node(parent_idx)]
    if len(exists) == 0:
        self._parents.append(parent_idx)
        return parent_idx

def add_children(self, children_idx):
    added = []
    for idx in children_idx:
        self.add_child(idx)
        added.append(idx)
    return added

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
