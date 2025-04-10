import heapq
from abc import ABC, abstractmethod
import math
class SPAlgorithm(ABC):
    @abstractmethod
    def calc_sp(self, graph, source: int, dest: int) -> float:
        pass

"""The Graph class (Directed Version)"""
class Graph:
    def __init__(self, nodes):
        # Adjacency list for outgoing edges only
        self.adj = [[] for _ in range(nodes)]
        # Weights dictionary: key is (source, destination) tuple
        self.weights = {}

    def adjacent_nodes(self, node):
        """Returns nodes reachable via outgoing edges from 'node'."""
        return self.adj[node]

    def add_edge(self, src, dst, weight):
        """Adds a directed edge from src to dst with the given weight."""
        if dst not in self.adj[src]:
            self.adj[src].append(dst)
        self.weights[(src, dst)] = weight  # Update or add weight

    def get_weight(self, src, dst):
        """Returns weight of the edge from src to dst, None if not exists."""
        return self.weights.get((src, dst), None)

    def has_edge(self, src, dst):
        """Checks if there's a directed edge from src to dst."""
        return dst in self.adj[src]

    def remove_edge(self, src, dst):
        """Removes the directed edge from src to dst."""
        if dst in self.adj[src]:
            self.adj[src].remove(dst)
        if (src, dst) in self.weights:
            del self.weights[(src, dst)]

    # Get the number of nodes in the graph
    def get_size(self):
        return len(self.adj)

    # String representation of the graph
    def __str__(self):
        output = ""
        for (i, j), weight in self.weights.items():
            output += f"\n\t{i} <-> {j}   weight: {weight}"
        return output
"""End of Graph class"""


"""Weighted Graph class"""
class WeightedGraph(Graph):
    def __init__(self, nodes):
        super().__init__(nodes)  # Inherit base Graph constructor

    def are_connected(self, node1, node2):
        """Check if two nodes are directly connected."""
        return node2 in self.adjacent_nodes(node1)

    def connected_nodes(self, node):
        """Return all nodes connected to the given node."""
        return self.adjacent_nodes(node)

    def add_node(self):
        """Add a new node to the graph."""
        self.adj.append([])

    def number_of_nodes(self):
        """Return total number of nodes."""
        return self.get_size()

    def get_total_weight(self):
        """Return the sum of all edge weights (for directed graph)."""
        return sum(self.weights.values())
    def get_weight(self, ):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
        return total / 2

"""Heuristics class"""
class Heuristics(WeightedGraph):
    def __init__(self,nodes=0):
        super().__init__(nodes)  # Setup graph structure
        self.heuristics = {}
    @staticmethod
    def euclidean_distance(lat1, lon1, lat2, lon2):
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    def get_heurisitics(self,stations_data, goal):
        goal_latitude = stations_data[goal][0]
        goal_longitude = stations_data[goal][1]
        h = {}
        for sid, values in stations_data.items():
            h[sid] = self.euclidean_distance(lon1=goal_longitude, lat1=goal_latitude, lon2=values[1], lat2=values[0])
        return h







"""Bellman_ford Algorithm"""
class Bellman_Ford(SPAlgorithm):
    def calc_sp(self, graph, source: int, dest: int) -> float:
        distances = {node: float('inf') for node in graph.graph}
        distances[source] = 0
        paths = {node: [] for node in graph.graph}
        paths[source] = [source]
        relax_count = {node: 0 for node in graph.graph}

        # Assuming max k = number of nodes - 1 (standard Bellman-Ford)
        k = len(graph.graph) - 1

        for _ in range(k):
            for node in graph.graph:
                if relax_count[node] >= k:
                    continue
                for neighbor in graph.graph[node]:
                    weight = graph.weight[(node, neighbor)]
                    if distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
                        paths[neighbor] = paths[node] + [neighbor]
                        relax_count[neighbor] += 1

        return distances[dest]
"""end of Bellman_ford Algorithm"""



"""A* Algorithm for Directed Graphs"""
def A_star(graph: Graph, source: int, destination: int, heuristic: dict):
    # Priority queue (min-heap) storing tuples of (f_score, node)
    open_set = []
    heapq.heappush(open_set, (heuristic[source], source))

    # Dictionary to track the optimal path (i.e., where each node came from)
    came_from = {}

    # Initialize g_score: cost from source to each node (infinite initially)
    g_score = {node: float('inf') for node in range(graph.get_size())}
    g_score[source] = 0  # Cost from source to itself is 0

    # Initialize f_score: estimated total cost from source to goal through each node
    f_score = {node: float('inf') for node in range(graph.get_size())}
    f_score[source] = heuristic[source]  # f_score = g_score + heuristic

    while open_set:
        # Pop the node with the lowest f_score
        current_f, current = heapq.heappop(open_set)

        # If the destination is reached, reconstruct and return the path
        if current == destination:
            path = [current]
            while current != source:
                current = came_from[current]
                path.append(current)
            return (came_from, path[::-1])  # Return reversed path from source to destination

        # If this entry is outdated (higher than recorded f_score), skip it
        if current_f > f_score[current]:
            continue

        # Explore neighbors of the current node
        for neighbor in graph.adjacent_nodes(current):
            edge_weight = graph.get_weight(current, neighbor)
            tentative_g = g_score[current] + edge_weight  # Compute tentative g_score

            # If a shorter path to neighbor is found
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current  # Update path
                g_score[neighbor] = tentative_g  # Update g_score
                f_score[neighbor] = tentative_g + heuristic[neighbor]  # Update f_score
                heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Add to open set

    return (came_from, None)  # Return None if no path is found
"""A* Algorithm for Directed Graphs"""



"""Class A_star to inherit from SPAlgorithm """
class A_Star(SPAlgorithm):
    def __init__(self, heuristic: dict):
        self.heuristic = heuristic  # Provide heuristic externally

    def calc_sp(self, graph, source: int, dest: int) -> float:
        _, path = A_star(graph, source, dest, self.heuristic)
        if path is None:
            return float('inf')

        # Sum weights along the path to calculate distance (since A_star doesn't return distance directly)
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += graph.get_weight(path[i], path[i + 1])
        return total_cost
"""End of A_star class"""



"""class Dijskstra algorithm to inherit from SPAlgorithm"""
class Dijkstra (SPAlgorithm):
    def calc_sp(self, graph, source: int, dest: int) -> float:
        distances = {node: float('inf') for node in range(graph.get_size())}
        distances[source] = 0
        paths = {node: [] for node in range(graph.get_size())}
        paths[source] = [source]

        priority_queue = []
        heapq.heappush(priority_queue, (0, source))

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor in graph.adjacent_nodes(current_node):
                weight = graph.get_weight(current_node, neighbor)
                if weight is None:
                    continue

                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current_node] + [neighbor]
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[dest]
"""end of class Dijskstra"""



class ShortPathFinder:
    def __init__(self):
        self._graph = None      # Composition: Graph object
        self._algorithm = None  # Composition: SPAlgorithm object

    def calc_short_path(self, source: int, dest: int) -> float:
        return self._algorithm.calc_sp(self._graph, source, dest)

    def set_graph(self, graph: Graph) -> None:
        """Inject a graph object."""
        self._graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm) -> None:
        """Inject an algorithm object."""
        self._algorithm = algorithm
