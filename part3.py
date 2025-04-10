import numpy as np
import heapq

## Weighted Graph Class
class WeightedGraph:

    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []

    def are_connected(self, node1, node2):
        for node in self.adj[node1]:
            if node == node2:
                return True
        return False

    def connected_nodes(self, node):
        return self.graph[node]

    def add_node(self,):
        self.graph[len(self.graph)] = []

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight

            #since it is undirected
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight

    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
                
        return total/2

#Classical implementations 
def bellman_ford_classical(graph_obj, source):
    distances = {node: float('inf') for node in graph_obj.graph}
    distances[source] = 0
    paths = {node: [] for node in graph_obj.graph}
    paths[source] = [source]

    for _ in range(len(graph_obj.graph) - 1):
        for node in graph_obj.graph:
            for neighbor in graph_obj.graph[node]:
                weight = graph_obj.weight[(node, neighbor)]
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    paths[neighbor] = paths[node] + [neighbor]

    return distances, paths

def djikstra_classical(graph_obj, source):
    distances = {node: float('inf') for node in graph_obj.graph}
    distances[source] = 0
    paths = {node: [] for node in graph_obj.graph}
    paths[source] = [source]

    # Use heapq for the priority queue
    priority_queue = [(0, source)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)  # Pop the smallest distance node

        # Iterate over neighbors
        for neighbor in graph_obj.graph[current_node]:
            weight = graph_obj.weight[(current_node, neighbor)]
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))  # Push the updated distance and node

    return distances, paths

def all_pairs_shortest_path_dijkstra(graph):
    distances = {}
    previous = {}

    for source in graph.graph:
        # Run Dijkstra's algorithm for each source node
        distances[source], paths = djikstra_classical(graph, source)
        previous[source] = {v: paths[v][-2] if len(paths[v]) > 1 else None for v in paths}
    return distances, previous

def all_pairs_shortest_path_bellman_ford(graph):
    distances = {}
    previous = {}

    for source in graph.graph:
        # Run Bellman-Ford's algorithm for each source node
        distances[source], paths = bellman_ford_classical(graph, source)
        previous[source] = {v: paths[v][-2] if len(paths[v]) > 1 else None for v in paths}
    return distances, previous

def create_random_graph(num_nodes, num_edges, max_weight, min_weight):
    graph = WeightedGraph(num_nodes)
    edges = set()

    while len(edges) < num_edges:
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)
        if node1 != node2 and (node1, node2) not in edges:
            weight = np.random.randint(min_weight, max_weight + 1)
            graph.add_edge(node1, node2, weight)
            edges.add((node1, node2))
    return graph

#test both algorithms with a random graph
graph = create_random_graph(10, 20, 10, -10)
distances_bf, previous_bf = all_pairs_shortest_path_bellman_ford(graph)
print("Bellman-Ford distances: ", distances_bf)
print("Bellman-Ford previous nodes: ", previous_bf)

graph = create_random_graph(10, 20, 10, 0)
disrances_dj, previous_dj = all_pairs_shortest_path_dijkstra(graph)
print("Dijkstra distances: ", disrances_dj)
print("Dijkstra previous nodes: ", previous_dj)
