from typing import Union, Optional, Generator
import random
from collections import deque


class Graph:
    def __init__(self):
        self.pairs: dict[tuple[str, str], list[Union[str, int]]] = {}
        self.adj_list: dict[str, list[tuple[str, str]]] = self.build_adj_list()

    def get_dataset(self) -> Union[None, dict[Union[str, int], list[list[str], list[str]]]]:
        if len(self.pairs) == 0:
            return None
        ret = {}
        for (parent, child), actions in self.pairs.items():
            for action in actions:
                if action not in ret:
                    ret[action] = [[], []]
                ret[action][0].append(parent)
                ret[action][1].append(child)
        return ret

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
        self.build_adj_list()

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

    # def find_path(self, parent: str, child: str):
    #     ''' breadth-first-search from parent to child, avoids loops, returns list of edges '''

    #     def get_all_children(parent: str) -> list[tuple[str, str]]:
    #         return [k for k in self.pairs.keys() if k[0] == parent]

    #     def get_all_parents(child: str) -> list[tuple[str, str]]:
    #         return [k for k in self.pairs.keys() if k[1] == child]

    #     runningRights = []
    #     runningLefts = []
    #     rights = get_all_children(parent)
    #     lefts = get_all_parents(child)
    #     runningRights.extend(rights)
    #     runningLefts.extend(lefts)
    #     while True:
    #         for pc in rights:
    #             nextRights = get_all_children(pc[1])
    #             runningRights.extend(nextRights)
    #             for r in nextRights:
    #                 if r[1] in [x[0] for x in runningLefts]:
    #                     found = r[1]
    #                     break
    #         rights = nextRights
    #         for pc in lefts:
    #             nextLefts = get_all_parents(pc[0])
    #             runningLefts.extend(nextLefts)
    #             for l in nextLefts:
    #                 if l[0] in [x[1] for x in runningRights]:
    #                     found = l[0]
    #                     break
    #         lefts = nextLefts

    # not working
    def find_path(self, start: str, end: str) -> Optional[list[Union[str, int]]]:
        if start == end:
            return []

        def bfs(queue, visited, paths):
            current, path = queue.popleft()
            for (src, dest), actions in self.pairs.items():
                if src == current and dest not in visited:
                    visited.add(dest)
                    new_path = path + actions
                    paths[dest] = new_path
                    queue.append((dest, new_path))

        # Initialize BFS from both ends
        queue_start = deque([(start, [])])
        queue_end = deque([(end, [])])
        visited_start, visited_end = {start}, {end}
        paths_start, paths_end = {start: []}, {end: []}

        while queue_start and queue_end:
            if queue_start:
                bfs(queue_start, visited_start, paths_start)
            if queue_end:
                bfs(queue_end, visited_end, paths_end)

            # Check for intersection
            intersection = visited_start.intersection(visited_end)
            if intersection:
                meeting_point = intersection.pop()
                path_from_start = paths_start[meeting_point]
                path_from_end = paths_end[meeting_point][::-1]
                return path_from_start + path_from_end

        return None

    def build_adj_list(self) -> dict[str, list[tuple[str, str]]]:
        adj_list = {}
        for edge_key, edges in self.pairs.items():
            parent_node, child_node = edge_key
            adj_list.setdefault(parent_node, []).extend(
                (child_node, edge) for edge in edges)
            adj_list.setdefault(child_node, []).extend(
                (parent_node, edge) for edge in edges)  # For bidirectional search
        return adj_list

    def bfs(self, start: str) -> Generator[tuple[str, list[tuple[str, str, str]]], None, None]:
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

    # v0
    # def path(self, parent: str, child: str) -> Optional[list[tuple[str, str, str]]]:
    #    ''' bidirectional breadth-first-search from both ends, avoids loops, yields nodes and paths'''
    #    search_from_parent = self.bfs(parent)
    #    search_from_child = self.bfs(child)
    #    visited_from_parent = {}
    #    visited_from_child = {}
    #    while True:
    #        try:
    #            node_parent, path_parent = next(search_from_parent)
    #            visited_from_parent[node_parent] = path_parent
    #        except StopIteration:
    #            break
    #        try:
    #            node_child, path_child = next(search_from_child)
    #            visited_from_child[node_child] = path_child
    #        except StopIteration:
    #            break
    #        if node_parent in visited_from_child:
    #            return path_parent + list(reversed([(child_node, parent_node, edge) for parent_node, child_node, edge in visited_from_child[node_parent]]))
    #        if node_child in visited_from_parent:
    #            return visited_from_parent[node_child] + list(reversed([(child_node, parent_node, edge) for parent_node, child_node, edge in visited_from_child[node_child]]))
    #    return None  # Return None if no path is found

    # v1
    def path(self, parent: str, child: str) -> Optional[list[tuple[str, str, str]]]:
        search_from_parent = self.bfs(parent)
        search_from_child = self.bfs(child)
        visited_from_parent = {}
        visited_from_child = {}

        while True:
            node_parent, path_parent = next(search_from_parent, (None, None))
            node_child, path_child = next(search_from_child, (None, None))

            if node_parent is None and node_child is None:
                print("Both searches exhausted without intersection.")
                return None

            if node_parent is not None:
                visited_from_parent[node_parent] = path_parent
                if node_parent in visited_from_child:
                    print(f"Intersection found at {node_parent}.")
                    return path_parent + list(reversed([(child_node, parent_node, edge)
                                                        for parent_node, child_node, edge in visited_from_child[node_parent]]))

            if node_child is not None:
                visited_from_child[node_child] = path_child
                if node_child in visited_from_parent:
                    print(f"Intersection found at {node_child}.")
                    return visited_from_parent[node_child] + list(reversed([(child_node, parent_node, edge)
                                                                            for parent_node, child_node, edge in path_child]))

        return None

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


class CountingGraph(Graph):
    def __init__(self):
        super().__init__()
        self.counts: dict[str, int] = {}

    def count(self, child: str) -> None:
        self.counts[child] = self.counts.get(child, 0) + 1

    # override
    def add(self, parent: str, child: str, edge: str):
        if (parent, child) not in self.pairs:
            self.pairs[(parent, child)] = []
        if edge not in self.pairs[(parent, child)]:
            self.pairs[(parent, child)].append(edge)
        if self.counts is not None:
            self.count(child)

    def get_random_popular_state(self) -> Union[str, None]:
        ''' get a random state that has been visited more than once '''
        average = sum([v for v in self.counts.values()]) / len(self.counts)
        popular = [k for k, v in self.counts.items() if v >= average]
        if len(popular) == 0:
            return None
        else:
            return random.choice(popular)

# Usage
# graph = Graph()
# graph.add_edge('a', 'b', 'edge1')
# graph.add_edge('b', 'c', 'edge2')
# graph.add_edge('b', 'a', 'edge3')
# graph.add_edge('c', 'd', 'edge4')
# path = graph.get_path('a', 'c')
# print(path)  # Output: [('a', 'b', 'edge1'), ('b', 'c', 'edge2')]
