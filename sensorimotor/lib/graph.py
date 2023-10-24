from typing import Union, List, Tuple, Optional, Dict, Generator
from collections import deque
# I'm not using anytree correctly. nodes are the entire path, very inefficient.
# Perhaps I can use these graphs to map it... idk.
# otherwise just go back to table structure.
# I see now why I was using anytree like that, every node is a path from a node,
# maybe the root, to every node, making lookup times on getting a path to any
# node very fast.


class Graph:
    def __init__(self):
        self.pairs: dict[tuple[str, str], Union[str, int]] = {}
        self.adj_list = self.build_adj_list()

    def get(self, parent: str, child: str) -> Union[Union[str, int], None]:
        return self.pairs.get((parent, child), [None])[0]

    def get_children(self, parent: str) -> dict[tuple[str, str], Union[str, int]]:
        return {k: v for k, v in self.pairs.items() if k[0] == parent}

    def get_parents(self, child: str) -> dict[tuple[str, str], Union[str, int]]:
        return {k: v for k, v in self.pairs.items() if k[1] == child}

    def add(self, parent: str, child: str, edge: str):
        if (parent, child) not in self.pairs:
            self.pairs[(parent, child)] = []
        if edge not in self.pairs[(parent, child)]:
            self.pairs[(parent, child)].append(edge)

    def get_path_only_from_parent(self, parent: str, child: str):
        ''' BFS from parent to child, avoids loops, returns list of edges '''
        queue = deque([(parent, [])])
        visited = set()
        while queue:
            node, path = queue.popleft()  # Dequeue the front node-path pair
            if node == child:
                return path  # Return the path when the end node is reached
            if node not in visited:
                visited.add(node)  # Mark the node as visited
                for edge_key, edges in self.pairs.items():
                    parent_node, child_node = edge_key
                    if parent_node == node:
                        for edge in edges:
                            # Enqueue all neighboring nodes and the path so far
                            new_path = path + [(parent_node, child_node, edge)]
                            queue.append((child_node, new_path))
        return None  # Return None if no path is found

    def get_path_only_from_child(self, parent: str, child: str):
        ''' breadth-first-search from child to parent, avoids loops, returns list of edges '''
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

    def build_adj_list(self) -> Dict[str, List[Tuple[str, str]]]:
        adj_list = {}
        for edge_key, edges in self.pairs.items():
            parent_node, child_node = edge_key
            adj_list.setdefault(parent_node, []).extend(
                (child_node, edge) for edge in edges)
            adj_list.setdefault(child_node, []).extend(
                (parent_node, edge) for edge in edges)  # For bidirectional search
        return adj_list

    def bfs(self, start: str) -> Generator[Tuple[str, List[Tuple[str, str, str]]], None, None]:
        queue = deque([(start, [])])
        visited = set()

        while queue:
            node, path = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            yield node, path
            for child_node, edge in self.adj_list.get(node, []):
                new_path = path + [(node, child_node, edge)]
                queue.append((child_node, new_path))

    def path(self, parent: str, child: str) -> Optional[List[Tuple[str, str, str]]]:
        ''' bidirectional breadth-first-search from both ends, avoids loops, yields nodes and paths'''
        search_from_parent = self.bfs(parent)
        search_from_child = self.bfs(child)
        visited_from_parent = {}
        visited_from_child = {}

        while True:
            try:
                node_parent, path_parent = next(search_from_parent)
                visited_from_parent[node_parent] = path_parent
            except StopIteration:
                break

            try:
                node_child, path_child = next(search_from_child)
                visited_from_child[node_child] = path_child
            except StopIteration:
                break

            if node_parent in visited_from_child:
                return path_parent + list(reversed([(child_node, parent_node, edge) for parent_node, child_node, edge in visited_from_child[node_parent]]))

            if node_child in visited_from_parent:
                return visited_from_parent[node_child] + list(reversed([(child_node, parent_node, edge) for parent_node, child_node, edge in visited_from_child[node_child]]))

        return None  # Return None if no path is found

    def visualize(self, filename: str = None):
        from graphviz import Digraph
        dot = Digraph(format='png')
        names = set()
        for k, edge in self.pairs.items():
            parent = k[0]
            child = k[1]
            if parent is not None and parent not in names:
                dot.node(str(parent))
                names.add(parent)
            if child is not None and child not in names:
                dot.node(str(child))
                names.add(child)
            if parent is not None and child is not None:
                dot.edge(str(parent), str(child), label=str(edge))
        if filename is not None:
            dot.render(filename, view=True)
        return dot

# Usage
# graph = Graph()
# graph.add_edge('a', 'b', 'edge1')
# graph.add_edge('b', 'c', 'edge2')
# graph.add_edge('b', 'a', 'edge3')
# graph.add_edge('c', 'd', 'edge4')
# path = graph.get_path('a', 'c')
# print(path)  # Output: [('a', 'b', 'edge1'), ('b', 'c', 'edge2')]
