'''
agents communicate in binary strings (motor output, actions)
environments communicate in binary strings (state rep, observaiton)
encoded binary strings are translated to integers (symbols) then mapped to 
integers (actual affects)
action pipeline example: 00000010 -> 2 -> -1
state pipeline example: 00000011 -> 3, 
state change pipeline: 3 + -1 = 2, 2 -> 00000010

Of course, if the environment can manipulate binary strings directly, using a nn
or something, then the user of this class can simply override the decode and
encode functions to be the identity function.
'''


class Environment():
    ''' the agent can move back and forth on the number line '''

    def __init__(
        self,
        initial_state: str = None, initial_action: str = None,
        initial_decoded_state: str = None, initial_decoded_action: str = None
    ):
        ''' initial_state: binary string, initial_action: binary string,
            initial_decoded_state: int, initial_decoded_action: int '''
        super(Environment, self).__init__()
        self.initial_state = initial_decoded_state or self._decode_state(
            initial_state)
        self.initial_action = initial_decoded_action or self._decode_action(
            initial_action)
        self.reset()

    def reset(self):
        self.state = self.initial_state
        self.action = self.initial_action

    def execute(self, actions: str):
        for action in actions:
            self.step(action)
        return self.state

    def step(self, action):
        return self._request(self._decode_action(action))

    def render(self):
        print("{}\t{}\t".format(self.action, self.state))

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _decode_action(self, bin: str):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(bin, 2)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _encode_action(self, integer: int, bin_size: int = 8):
        ''' converts an integer to a binary string of bin_size '''
        return bin(integer)[2:].zfill(bin_size)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _decode_state(self, bin: str):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(bin, 2)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _encode_state(self, integer: int, bin_size: int = 8):
        ''' converts an integer to a binary string of bin_size '''
        return bin(integer)[2:].zfill(bin_size)

    # once the action is translated to an symbol (like an integer)
    # it can be simply mapped to a translation of the state
    def _action_affect(self, action: int):
        ''' returns what we're going to do to the state '''
        if isinstance(action, int):
            return {0: 0, 1: 1, 2: -1}.get(action, 0)
        else:
            return 0

    # here we apply the action to the state
    def _state_behavior(self, affect: int):
        affect = (affect or 0)
        if self.state is None:
            return 0 + affect
        return self._decode_state(self.state) + affect

    def _request(self, action: int):
        self.state = self._encode_state(
            self._state_behavior(
                self._action_affect(action)))
        return self.state


# I'm not using anytree correctly. nodes are the entire path, very inefficient.
# Perhaps I can use these graphs to map it... idk.
# otherwise just go back to table structure.
# I see now why I was using anytree like that, every node is a path from a node,
# maybe the root, to every node, making lookup times on getting a path to any
# node very fast.

class Graph:
    def __init__(self):
        self.pairs: dict[tuple[str, str], list[str | int]] = {}

    def add_edge(self, parent: str, child: str, edge: str):
        if (parent, child) not in self.pairs:
            self.pairs[(parent, child)] = []
        if edge not in self.pairs[(parent, child)]:
            self.pairs[(parent, child)].append(edge)

    def get_path_only_from_parent(self, parent, child):
        ''' BFS from parent to child, avoids loops, returns list of edges '''
        from collections import deque
        # A queue to hold the node and the path so far
        queue = deque([(parent, [])])
        while queue:
            node, path = queue.popleft()  # Dequeue the front node-path pair
            if node == child:
                return path  # Return the path when the end node is reached
            for edge_key, edges in self.pairs.items():
                parent_node, child_node = edge_key
                if parent_node == node and edge_key not in path:
                    for edge in edges:
                        # Enqueue all neighboring nodes and the path so far
                        new_path = path + [(parent_node, child_node, edge)]
                        queue.append((child_node, new_path))
        return None  # Return None if no path is found

    def get_path_only_from_child(self, parent, child):
        ''' breadth-first-search from child to parent, avoids loops, returns list of edges '''
        from collections import deque
        # A queue to hold the node and the path so far
        queue = deque([(child, [])])
        while queue:
            node, path = queue.popleft()  # Dequeue the front node-path pair
            if node == parent:
                return path  # Return the path when the end node is reached
            for edge_key, edges in self.pairs.items():
                parent_node, child_node = edge_key
                if child_node == node and edge_key not in path:
                    for edge in edges:
                        # Enqueue all neighboring nodes and the path so far
                        new_path = path + [(parent_node, child_node, edge)]
                        queue.append((parent_node, new_path))
        return None  # Return None if no path is found

    def get_path_from_parent(self, parent):
        ''' Generator for BFS from parent to child, avoids loops, yields nodes and paths '''
        from collections import deque
        queue = deque([(parent, [])])
        while queue:
            node, path = queue.popleft()
            yield node, path  # Yield the current node and path
            for edge_key, edges in self.pairs.items():
                parent_node, child_node = edge_key
                if parent_node == node and edge_key not in path:
                    for edge in edges:
                        new_path = path + [(parent_node, child_node, edge)]
                        queue.append((child_node, new_path))

    def get_path_from_child(self, child):
        ''' Generator for BFS from child to parent, avoids loops, yields nodes and paths '''
        from collections import deque
        queue = deque([(child, [])])
        while queue:
            node, path = queue.popleft()
            yield node, path  # Yield the current node and path
            for edge_key, edges in self.pairs.items():
                parent_node, child_node = edge_key
                if child_node == node and edge_key not in path:
                    for edge in edges:
                        new_path = path + [(parent_node, child_node, edge)]
                        queue.append((parent_node, new_path))

    def get_path(self, parent, child):
        ''' Bidirectional BFS from both ends, avoids loops, returns list of edges '''
        search_from_parent = self.get_path_from_parent(parent)
        search_from_child = self.get_path_from_child(child)
        visited_from_parent = {}
        visited_from_child = {}
        while True:
            try:
                node_parent, path_parent = next(search_from_parent)
                visited_from_parent[node_parent] = path_parent
            except StopIteration:
                break  # Stop if the search from parent has exhausted all nodes
            try:
                node_child, path_child = next(search_from_child)
                visited_from_child[node_child] = path_child
            except StopIteration:
                break  # Stop if the search from child has exhausted all nodes
            # Check for intersection
            if node_parent in visited_from_child:
                # Combine the paths at the intersecting node
                return path_parent + list(reversed(visited_from_child[node_parent]))
            if node_child in visited_from_parent:
                # Combine the paths at the intersecting node
                return visited_from_parent[node_child] + list(reversed(path_child))
        return None  # Return None if no path is found

    def visualize(self, filename=None):
        from graphviz import Digraph
        dot = Digraph(format='png')
        names = set()
        for k, edge in self.pairs.items():
            parent = k[0]
            child = k[1]
            if parent not in names:
                dot.node(parent)
                names.add(parent)
            if child not in names:
                dot.node(child)
                names.add(child)
            dot.edge(parent, child, label=edge)
        if filename is not None:
            dot.render(filename, view=True)
        return dot
# graph = Graph()
#
# Add nodes
# graph.add_node('A')
# graph.add_node('B')
# graph.add_node('C')
#
# Add edges
# graph.add_edge('A', 'B', 'x')
# graph.add_edge('A', 'C', 'y')
# graph.add_edge('A', 'A', 'z')
# graph.add_edge('B', 'C', 'j')


class Vector:
    def __init__(self, edge, name):
        self.edge = edge
        self.name = name


class GraphTable:
    def __init__(self):
        # source of truth maps a node to it's children
        self.parents: dict[str:list[Vector]] = {}
        # reverse of children used to speed up path finding
        # self.children: dict[str:list[Vector]] = {}

    def node_exists(self, name: str):
        return name in self.parents.items()

    def get_node(self, name: str):
        self.parents.get(name, [])

    def find_path(self, start: str, end: str):
        from collections import deque
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            if node.name == end:
                return path
            for neighbor in self.parents.get(node.name, []):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
        return None

    # # optimization: by using a reverse of parents (children), and yeild...
    # def find_path_by_parents(self, start: str, end: str):
    #    from collections import deque
    #    queue = deque([(start, [start])])  # A queue to hold the node and the path so far
    #    while queue:
    #        node, path = queue.popleft()  # Dequeue the front node-path pair
    #        if node.name == end:
    #            return path  # Return the path when the end node is reached
    #        # Find all children of the current node
    #        children = [key for key, value in self.children.items() if node in value]
    #        for child in children:  # Enqueue all children and the path so far
    #            if child not in path:  # Avoid cycles by checking if child is already in path
    #                queue.append((child, path + [child]))
    #    return None  # Return None if no path is found

    def add_node(self, name: str):
        if name not in self.parents.items():
            self.parents[name] = []
        # optimization:
        # if name not in self.children.items():
        #    self.children[name] = []

    def add_edge(self, parent: str, edge: str, child: str):
        self.parents[parent].append(Vector(edge, child))
        # optimization:
        # self.children[child].append(Vector(edge, parent))

    def visualize(self, filename: str = None):
        dot = Digraph(format='png')
        for name, vectors in self.parents.items():
            dot.node(name)
            for vector in vectors:
                dot.edge(name, vector.name, label=vector.edge)
        if filename is not None:
            dot.render(filename, view=True)
        return dot


graph = GraphTable()
graph.add_node('A')
graph.add_node('B')
graph.add_node('C')
graph.add_node('D')

graph.add_edge('A', 'x', 'A')
graph.add_edge('A', 'z', 'C')
graph.add_edge('C', 'j', 'A')
graph.add_edge('C', 'k', 'D')
graph.add_edge('D', 'y', 'B')

graph.find_path('A', 'B')
