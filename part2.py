import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import tracemalloc
from copy import deepcopy
import matplotlib.pyplot as plt
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

## Part 2.1
def dijkstra(graph, source, k):
    distances = {node: float('inf') for node in graph.graph}
    distances[source] = 0
    paths = {node: [] for node in graph.graph}
    paths[source] = [source]

    relax_count = {node: 0 for node in graph.graph}

    # Use heapq for the priority queue
    priority_queue = [(0, source)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)  # Pop the smallest distance node

        # Skip if the node has already been relaxed k times
        if relax_count[current_node] >= k:
            continue

        relax_count[current_node] += 1

        for neighbor in graph.graph[current_node]:
            weight = graph.weight[(current_node, neighbor)]
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))  # Push the updated distance and node

    return distances, paths

#Part 2.2
def bellman_ford(graph_obj, source, k):
    distances = {node: float('inf') for node in graph_obj.graph}
    distances[source] = 0
    paths = {node: [] for node in graph_obj.graph}
    paths[source] = [source]
    relax_count = {node: 0 for node in graph_obj.graph}

    for _ in range(k):
        for node in graph_obj.graph:
            #skip if node has already relaxed k times
            if relax_count[node] >= k:
                continue
            for neighbor in graph_obj.graph[node]:
                weight = graph_obj.weight[(node, neighbor)]
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    paths[neighbor] = paths[node] + [neighbor]
                    relax_count[neighbor] += 1

    return distances, paths

#Part 2.3
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

def create_random_graph(num_nodes, num_edges, max_weight):
    graph = WeightedGraph(num_nodes)
    edges = set()

    while len(edges) < num_edges:
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)
        if node1 != node2 and (node1, node2) not in edges:
            weight = np.random.randint(1, max_weight + 1)
            graph.add_edge(node1, node2, weight)
            edges.add((node1, node2))
    return graph

def experiment_accuracy(trials, k, nodes=100, edges=200, max_weight=10):
    bellman_ford_accuracies = []
    djikstra_accuracies = []
    for _ in range(trials):
        
        graph = create_random_graph(100, 200, 10)
        source = np.random.randint(0, 100)
        
        test_graph = deepcopy(graph)
        
        djikstra_distances, djikstra_paths = dijkstra(test_graph, source, k)
        bellman_ford_distances, bellman_ford_paths = bellman_ford(test_graph, source, k)
        classical_bellman_ford_distances, classical_bellman_ford_paths = bellman_ford_classical(test_graph, source)
        classical_djikstra_distances, classical_djikstra_paths = djikstra_classical(test_graph, source)
        
        # Check if the distances are equal
        if classical_djikstra_distances == djikstra_distances:
            djikstra_accuracies.append(1)
        else:
            djikstra_accuracies.append(0)
        
        if classical_bellman_ford_distances == bellman_ford_distances:
            bellman_ford_accuracies.append(1)
        else:
            bellman_ford_accuracies.append(0)
        
    return bellman_ford_accuracies, djikstra_accuracies

def experiment_performance_time(trials, k, nodes=100, edges=200, max_weight=10):
    bellman_ford_times = []
    djikstra_times = []
    bellman_ford_classical_times = []
    djikstra_classical_times = []
    
    for _ in range(trials):
        graph = create_random_graph(nodes, edges, max_weight)
        test_graph = deepcopy(graph)
        source = np.random.randint(0, nodes)
        
        start = timeit.default_timer()
        bellman_ford(test_graph, source, k)
        end = timeit.default_timer()
        bellman_ford_times.append(end - start)
        
        start = timeit.default_timer()
        dijkstra(test_graph, source, k)
        end = timeit.default_timer()
        djikstra_times.append(end - start)
        
        start = timeit.default_timer()
        bellman_ford_classical(test_graph, source)
        end = timeit.default_timer()
        bellman_ford_classical_times.append(end - start)
        
        start = timeit.default_timer()
        djikstra_classical(test_graph, source)
        end = timeit.default_timer()
        djikstra_classical_times.append(end - start)
        
    return bellman_ford_times, djikstra_times, bellman_ford_classical_times, djikstra_classical_times

def experiment_performance_memory(trials, k, nodes=100, edges=200, max_weight=10):
    bellman_ford_memory = []
    djikstra_memory = []
    bellman_ford_classical_memory = []
    djikstra_classical_memory = []

    for _ in range(trials):
        graph = create_random_graph(nodes, edges, max_weight)
        test_graph = deepcopy(graph)
        source = np.random.randint(0, nodes)

        tracemalloc.start()
        bellman_ford(test_graph, source, k)
        current, peak = tracemalloc.get_traced_memory()
        bellman_ford_memory.append(peak)
        tracemalloc.stop()
        
        tracemalloc.start()
        dijkstra(test_graph, source, k)
        current, peak = tracemalloc.get_traced_memory()
        djikstra_memory.append(peak)
        tracemalloc.stop()

        tracemalloc.start()
        bellman_ford_classical(test_graph, source)
        current, peak = tracemalloc.get_traced_memory()
        bellman_ford_classical_memory.append(peak)
        tracemalloc.stop()

        tracemalloc.start()
        djikstra_classical(test_graph, source)
        current, peak = tracemalloc.get_traced_memory()
        djikstra_classical_memory.append(peak)
        tracemalloc.stop()

    return bellman_ford_memory, djikstra_memory, bellman_ford_classical_memory, djikstra_classical_memory


# Experiment of Accuracy
bellman_accuracies_k1, djikstra_accuracies_k1 = experiment_accuracy(50, 1)
bellman_accuracies_k2, djikstra_accuracies_k2 = experiment_accuracy(50, 2)
bellman_accuracies_k3, djikstra_accuracies_k3 = experiment_accuracy(50, 3)
bellman_accuracies_k4, djikstra_accuracies_k4 = experiment_accuracy(50, 4)
bellman_accuracies_k5, djikstra_accuracies_k5 = experiment_accuracy(50, 5)
bellman_accuracies_k6, djikstra_accuracies_k6 = experiment_accuracy(10, 6)
bellman_accuracies_k7, djikstra_accuracies_k7 = experiment_accuracy(50, 7)
bellman_accuracies_k8, djikstra_accuracies_k8 = experiment_accuracy(50, 8)
bellman_accuracies_k9, djikstra_accuracies_k9 = experiment_accuracy(50, 9)
bellman_accuracies_k10, djikstra_accuracies_k10 = experiment_accuracy(50, 10)

# Combined bar chart for accuracy (k = 1 to 10)
x = np.arange(1, 11)  # k values from 1 to 10
bar_width = 0.35

dijkstra_accuracies = [
    sum(djikstra_accuracies_k1) / len(djikstra_accuracies_k1),
    sum(djikstra_accuracies_k2) / len(djikstra_accuracies_k2),
    sum(djikstra_accuracies_k3) / len(djikstra_accuracies_k3),
    sum(djikstra_accuracies_k4) / len(djikstra_accuracies_k4),
    sum(djikstra_accuracies_k5) / len(djikstra_accuracies_k5),
    sum(djikstra_accuracies_k6) / len(djikstra_accuracies_k6),
    sum(djikstra_accuracies_k7) / len(djikstra_accuracies_k7),
    sum(djikstra_accuracies_k8) / len(djikstra_accuracies_k8),
    sum(djikstra_accuracies_k9) / len(djikstra_accuracies_k9),
    sum(djikstra_accuracies_k10) / len(djikstra_accuracies_k10),
]

bellman_ford_accuracies = [
    sum(bellman_accuracies_k1) / len(bellman_accuracies_k1),
    sum(bellman_accuracies_k2) / len(bellman_accuracies_k2),
    sum(bellman_accuracies_k3) / len(bellman_accuracies_k3),
    sum(bellman_accuracies_k4) / len(bellman_accuracies_k4),
    sum(bellman_accuracies_k5) / len(bellman_accuracies_k5),
    sum(bellman_accuracies_k6) / len(bellman_accuracies_k6),
    sum(bellman_accuracies_k7) / len(bellman_accuracies_k7),
    sum(bellman_accuracies_k8) / len(bellman_accuracies_k8),
    sum(bellman_accuracies_k9) / len(bellman_accuracies_k9),
    sum(bellman_accuracies_k10) / len(bellman_accuracies_k10),
]

plt.bar(x - bar_width / 2, dijkstra_accuracies, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + bar_width / 2, bellman_ford_accuracies, width=bar_width, label='Bellman-Ford', color='green')

plt.xticks(x)  # Set x-axis ticks to k values
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for k = 1 to 10')
plt.legend()
plt.show()

# Combined bar chart for accuracy (Graph Sizes)
graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
edges = [int((0.5 * size * (size - 1))/2) for size in graph_sizes]

bellman_accuracies_10, djikstra_accuracies_10 = experiment_accuracy(50, 10, nodes=10, edges=edges[0])
bellman_accuracies_20, djikstra_accuracies_20 = experiment_accuracy(50, 10, nodes=20, edges=edges[1])
bellman_accuracies_30, djikstra_accuracies_30 = experiment_accuracy(50, 10, nodes=30, edges=edges[2])
bellman_accuracies_40, djikstra_accuracies_40 = experiment_accuracy(50, 10, nodes=40, edges=edges[3])
bellman_accuracies_50, djikstra_accuracies_50 = experiment_accuracy(50, 10, nodes=50, edges=edges[4])
bellman_accuracies_60, djikstra_accuracies_60 = experiment_accuracy(50, 10, nodes=60, edges=edges[5])
bellman_accuracies_70, djikstra_accuracies_70 = experiment_accuracy(50, 10, nodes=70, edges=edges[6])
bellman_accuracies_80, djikstra_accuracies_80 = experiment_accuracy(50, 10, nodes=80, edges=edges[7])
bellman_accuracies_90, djikstra_accuracies_90 = experiment_accuracy(50, 10, nodes=90, edges=edges[8])
bellman_accuracies_100, djikstra_accuracies_100 = experiment_accuracy(50, 10, nodes=100, edges=edges[9])

dijkstra_accuracies = [
    sum(djikstra_accuracies_10) / len(djikstra_accuracies_10),
    sum(djikstra_accuracies_20) / len(djikstra_accuracies_20),
    sum(djikstra_accuracies_30) / len(djikstra_accuracies_30),
    sum(djikstra_accuracies_40) / len(djikstra_accuracies_40),
    sum(djikstra_accuracies_50) / len(djikstra_accuracies_50),
    sum(djikstra_accuracies_60) / len(djikstra_accuracies_60),
    sum(djikstra_accuracies_70) / len(djikstra_accuracies_70),
    sum(djikstra_accuracies_80) / len(djikstra_accuracies_80),
    sum(djikstra_accuracies_90) / len(djikstra_accuracies_90),
    sum(djikstra_accuracies_100) / len(djikstra_accuracies_100),
]

bellman_ford_accuracies = [
    sum(bellman_accuracies_10) / len(bellman_accuracies_10),
    sum(bellman_accuracies_20) / len(bellman_accuracies_20),
    sum(bellman_accuracies_30) / len(bellman_accuracies_30),
    sum(bellman_accuracies_40) / len(bellman_accuracies_40),
    sum(bellman_accuracies_50) / len(bellman_accuracies_50),
    sum(bellman_accuracies_60) / len(bellman_accuracies_60),
    sum(bellman_accuracies_70) / len(bellman_accuracies_70),
    sum(bellman_accuracies_80) / len(bellman_accuracies_80),
    sum(bellman_accuracies_90) / len(bellman_accuracies_90),
    sum(bellman_accuracies_100) / len(bellman_accuracies_100),
]

x = np.arange(len(graph_sizes))  # Graph sizes (10, 20, ..., 100)

plt.bar(x - bar_width / 2, dijkstra_accuracies, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + bar_width / 2, bellman_ford_accuracies, width=bar_width, label='Bellman-Ford', color='green')

plt.xticks(x, labels=graph_sizes)  # Set x-axis ticks to graph sizes
plt.xlabel('Graph Size (V)')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Graph Sizes (10 to 100)')
plt.legend()
plt.show()

# Combined bar chart for accuracy (Graph Densities)
graph_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
edges = [int((density * 20 * (20 - 1))/2) for density in graph_densities]

bellman_accuracies_density_0_1, djikstra_accuracies_density_0_1 = experiment_accuracy(50, 10, nodes=20, edges=edges[0])
bellman_accuracies_density_0_2, djikstra_accuracies_density_0_2 = experiment_accuracy(50, 10, nodes=20, edges=edges[1])
bellman_accuracies_density_0_3, djikstra_accuracies_density_0_3 = experiment_accuracy(50, 10, nodes=20, edges=edges[2])
bellman_accuracies_density_0_4, djikstra_accuracies_density_0_4 = experiment_accuracy(50, 10, nodes=20, edges=edges[3])
bellman_accuracies_density_0_5, djikstra_accuracies_density_0_5 = experiment_accuracy(50, 10, nodes=20, edges=edges[4])
bellman_accuracies_density_0_6, djikstra_accuracies_density_0_6 = experiment_accuracy(50, 10, nodes=20, edges=edges[5])
bellman_accuracies_density_0_7, djikstra_accuracies_density_0_7 = experiment_accuracy(50, 10, nodes=20, edges=edges[6])
bellman_accuracies_density_0_8, djikstra_accuracies_density_0_8 = experiment_accuracy(50, 10, nodes=20, edges=edges[7])
bellman_accuracies_density_0_9, djikstra_accuracies_density_0_9 = experiment_accuracy(50, 10, nodes=20, edges=edges[8])
bellman_accuracies_density_1_0, djikstra_accuracies_density_1_0 = experiment_accuracy(50, 10, nodes=20, edges=edges[9])

bellman_accuracies = [
    sum(bellman_accuracies_density_0_1) / len(bellman_accuracies_density_0_1),
    sum(bellman_accuracies_density_0_2) / len(bellman_accuracies_density_0_2),
    sum(bellman_accuracies_density_0_3) / len(bellman_accuracies_density_0_3),
    sum(bellman_accuracies_density_0_4) / len(bellman_accuracies_density_0_4),
    sum(bellman_accuracies_density_0_5) / len(bellman_accuracies_density_0_5),
    sum(bellman_accuracies_density_0_6) / len(bellman_accuracies_density_0_6),
    sum(bellman_accuracies_density_0_7) / len(bellman_accuracies_density_0_7),
    sum(bellman_accuracies_density_0_8) / len(bellman_accuracies_density_0_8),
    sum(bellman_accuracies_density_0_9) / len(bellman_accuracies_density_0_9),
    sum(bellman_accuracies_density_1_0) / len(bellman_accuracies_density_1_0),
]

dijkstra_accuracies = [
    sum(djikstra_accuracies_density_0_1) / len(djikstra_accuracies_density_0_1),
    sum(djikstra_accuracies_density_0_2) / len(djikstra_accuracies_density_0_2),
    sum(djikstra_accuracies_density_0_3) / len(djikstra_accuracies_density_0_3),
    sum(djikstra_accuracies_density_0_4) / len(djikstra_accuracies_density_0_4),
    sum(djikstra_accuracies_density_0_5) / len(djikstra_accuracies_density_0_5),
    sum(djikstra_accuracies_density_0_6) / len(djikstra_accuracies_density_0_6),
    sum(djikstra_accuracies_density_0_7) / len(djikstra_accuracies_density_0_7),
    sum(djikstra_accuracies_density_0_8) / len(djikstra_accuracies_density_0_8),
    sum(djikstra_accuracies_density_0_9) / len(djikstra_accuracies_density_0_9),
    sum(djikstra_accuracies_density_1_0) / len(djikstra_accuracies_density_1_0),
]

x = np.arange(len(graph_densities))  # Graph densities (0.1, 0.2, ..., 1.0)

plt.bar(x - bar_width / 2, dijkstra_accuracies, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + bar_width / 2, bellman_accuracies, width=bar_width, label='Bellman-Ford', color='green')

plt.xticks(x, labels=[str(d) for d in graph_densities])  # Set x-axis ticks to densities
plt.xlabel('Graph Density')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Graph Densities')
plt.legend()
plt.show()

#Test perofrmance for different k values
bellman_ford_times_k1, dijkstra_times_k1, bellman_ford_classical_times_k1, dijkstra_classical_times_k1 = experiment_performance_time(50, 1)
bellman_ford_times_k2, dijkstra_times_k2, bellman_ford_classical_times_k2, dijkstra_classical_times_k2 = experiment_performance_time(50, 2)
bellman_ford_times_k3, dijkstra_times_k3, bellman_ford_classical_times_k3, dijkstra_classical_times_k3 = experiment_performance_time(50, 3)
bellman_ford_times_k4, dijkstra_times_k4, bellman_ford_classical_times_k4, dijkstra_classical_times_k4 = experiment_performance_time(50, 4)
bellman_ford_times_k5, dijkstra_times_k5, bellman_ford_classical_times_k5, dijkstra_classical_times_k5 = experiment_performance_time(50, 5)
bellman_ford_times_k6, dijkstra_times_k6, bellman_ford_classical_times_k6, dijkstra_classical_times_k6 = experiment_performance_time(50, 6)
bellman_ford_times_k7, dijkstra_times_k7, bellman_ford_classical_times_k7, dijkstra_classical_times_k7 = experiment_performance_time(50, 7)
bellman_ford_times_k8, dijkstra_times_k8, bellman_ford_classical_times_k8, dijkstra_classical_times_k8 = experiment_performance_time(50, 8)
bellman_ford_times_k9, dijkstra_times_k9, bellman_ford_classical_times_k9, dijkstra_classical_times_k9 = experiment_performance_time(50, 9)
bellman_ford_times_k10, dijkstra_times_k10, bellman_ford_classical_times_k10, dijkstra_classical_times_k10 = experiment_performance_time(50, 10)

average_times_bellman_ford = [
    sum(bellman_ford_times_k1) / len(bellman_ford_times_k1),
    sum(bellman_ford_times_k2) / len(bellman_ford_times_k2),
    sum(bellman_ford_times_k3) / len(bellman_ford_times_k3),
    sum(bellman_ford_times_k4) / len(bellman_ford_times_k4),
    sum(bellman_ford_times_k5) / len(bellman_ford_times_k5),
    sum(bellman_ford_times_k6) / len(bellman_ford_times_k6),
    sum(bellman_ford_times_k7) / len(bellman_ford_times_k7),
    sum(bellman_ford_times_k8) / len(bellman_ford_times_k8),
    sum(bellman_ford_times_k9) / len(bellman_ford_times_k9),
    sum(bellman_ford_times_k10) / len(bellman_ford_times_k10),
]

average_times_dijkstra = [
    sum(dijkstra_times_k1) / len(dijkstra_times_k1),
    sum(dijkstra_times_k2) / len(dijkstra_times_k2),
    sum(dijkstra_times_k3) / len(dijkstra_times_k3),
    sum(dijkstra_times_k4) / len(dijkstra_times_k4),
    sum(dijkstra_times_k5) / len(dijkstra_times_k5),
    sum(dijkstra_times_k6) / len(dijkstra_times_k6),
    sum(dijkstra_times_k7) / len(dijkstra_times_k7),
    sum(dijkstra_times_k8) / len(dijkstra_times_k8),
    sum(dijkstra_times_k9) / len(dijkstra_times_k9),
    sum(dijkstra_times_k10) / len(dijkstra_times_k10),
]

average_times_bellman_ford_classical = [
    sum(bellman_ford_classical_times_k1) / len(bellman_ford_classical_times_k1),
    sum(bellman_ford_classical_times_k2) / len(bellman_ford_classical_times_k2),
    sum(bellman_ford_classical_times_k3) / len(bellman_ford_classical_times_k3),
    sum(bellman_ford_classical_times_k4) / len(bellman_ford_classical_times_k4),
    sum(bellman_ford_classical_times_k5) / len(bellman_ford_classical_times_k5),
    sum(bellman_ford_classical_times_k6) / len(bellman_ford_classical_times_k6),
    sum(bellman_ford_classical_times_k7) / len(bellman_ford_classical_times_k7),
    sum(bellman_ford_classical_times_k8) / len(bellman_ford_classical_times_k8),
    sum(bellman_ford_classical_times_k9) / len(bellman_ford_classical_times_k9),
    sum(bellman_ford_classical_times_k10) / len(bellman_ford_classical_times_k10),
]

average_times_dijkstra_classical = [
    sum(dijkstra_classical_times_k1) / len(dijkstra_classical_times_k1),
    sum(dijkstra_classical_times_k2) / len(dijkstra_classical_times_k2),
    sum(dijkstra_classical_times_k3) / len(dijkstra_classical_times_k3),
    sum(dijkstra_classical_times_k4) / len(dijkstra_classical_times_k4),
    sum(dijkstra_classical_times_k5) / len(dijkstra_classical_times_k5),
    sum(dijkstra_classical_times_k6) / len(dijkstra_classical_times_k6),
    sum(dijkstra_classical_times_k7) / len(dijkstra_classical_times_k7),
    sum(dijkstra_classical_times_k8) / len(dijkstra_classical_times_k8),
    sum(dijkstra_classical_times_k9) / len(dijkstra_classical_times_k9),
    sum(dijkstra_classical_times_k10) / len(dijkstra_classical_times_k10),
]

# Plot the results in a multi-bar bar chart
x = np.arange(1, 11)  
bar_width = 0.2
plt.bar(x - 1.5 * bar_width, average_times_bellman_ford, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, average_times_dijkstra, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, average_times_bellman_ford_classical, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, average_times_dijkstra_classical, width=bar_width, label='Dijkstra Classical', color='red')
plt.xticks(x, labels=[str(i) for i in x]) 
plt.xlabel('k')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison of Algorithms for Different k Values')
plt.legend()
plt.show()

#Test Performance for different graph sizes with density 0.5
graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
edges = [int((0.5 * size * (size - 1))/2) for size in graph_sizes]

bellman_ford_times_10, dijkstra_times_10, bellman_ford_classical_times_10, dijkstra_classical_times_10 = experiment_performance_time(50, 10, nodes=10, edges=edges[0])
bellman_ford_times_20, dijkstra_times_20, bellman_ford_classical_times_20, dijkstra_classical_times_20 = experiment_performance_time(50, 10, nodes=20, edges=edges[1])
bellman_ford_times_30, dijkstra_times_30, bellman_ford_classical_times_30, dijkstra_classical_times_30 = experiment_performance_time(50, 10, nodes=30, edges=edges[2])
bellman_ford_times_40, dijkstra_times_40, bellman_ford_classical_times_40, dijkstra_classical_times_40 = experiment_performance_time(50, 10, nodes=40, edges=edges[3])
bellman_ford_times_50, dijkstra_times_50, bellman_ford_classical_times_50, dijkstra_classical_times_50 = experiment_performance_time(50, 10, nodes=50, edges=edges[4])
bellman_ford_times_60, dijkstra_times_60, bellman_ford_classical_times_60, dijkstra_classical_times_60 = experiment_performance_time(50, 10, nodes=60, edges=edges[5])
bellman_ford_times_70, dijkstra_times_70, bellman_ford_classical_times_70, dijkstra_classical_times_70 = experiment_performance_time(50, 10, nodes=70, edges=edges[6])
bellman_ford_times_80, dijkstra_times_80, bellman_ford_classical_times_80, dijkstra_classical_times_80 = experiment_performance_time(50, 10, nodes=80, edges=edges[7])
bellman_ford_times_90, dijkstra_times_90, bellman_ford_classical_times_90, dijkstra_classical_times_90 = experiment_performance_time(50, 10, nodes=90, edges=edges[8])
bellman_ford_times_100, dijkstra_times_100, bellman_ford_classical_times_100, dijkstra_classical_times_100 = experiment_performance_time(50, 5, nodes=100, edges=edges[9])

average_times_bellman_ford = [
    sum(bellman_ford_times_10) / len(bellman_ford_times_10),
    sum(bellman_ford_times_20) / len(bellman_ford_times_20),
    sum(bellman_ford_times_30) / len(bellman_ford_times_30),
    sum(bellman_ford_times_40) / len(bellman_ford_times_40),
    sum(bellman_ford_times_50) / len(bellman_ford_times_50),
    sum(bellman_ford_times_60) / len(bellman_ford_times_60),
    sum(bellman_ford_times_70) / len(bellman_ford_times_70),
    sum(bellman_ford_times_80) / len(bellman_ford_times_80),
    sum(bellman_ford_times_90) / len(bellman_ford_times_90),
    sum(bellman_ford_times_100) / len(bellman_ford_times_100),
]

average_times_dijkstra = [
    sum(dijkstra_times_10) / len(dijkstra_times_10),
    sum(dijkstra_times_20) / len(dijkstra_times_20),
    sum(dijkstra_times_30) / len(dijkstra_times_30),
    sum(dijkstra_times_40) / len(dijkstra_times_40),
    sum(dijkstra_times_50) / len(dijkstra_times_50),
    sum(dijkstra_times_60) / len(dijkstra_times_60),
    sum(dijkstra_times_70) / len(dijkstra_times_70),
    sum(dijkstra_times_80) / len(dijkstra_times_80),
    sum(dijkstra_times_90) / len(dijkstra_times_90),
    sum(dijkstra_times_100) / len(dijkstra_times_100),
]

average_times_bellman_ford_classical = [
    sum(bellman_ford_classical_times_10) / len(bellman_ford_classical_times_10),
    sum(bellman_ford_classical_times_20) / len(bellman_ford_classical_times_20),
    sum(bellman_ford_classical_times_30) / len(bellman_ford_classical_times_30),
    sum(bellman_ford_classical_times_40) / len(bellman_ford_classical_times_40),
    sum(bellman_ford_classical_times_50) / len(bellman_ford_classical_times_50),
    sum(bellman_ford_classical_times_60) / len(bellman_ford_classical_times_60),
    sum(bellman_ford_classical_times_70) / len(bellman_ford_classical_times_70),
    sum(bellman_ford_classical_times_80) / len(bellman_ford_classical_times_80),
    sum(bellman_ford_classical_times_90) / len(bellman_ford_classical_times_90),
    sum(bellman_ford_classical_times_100) / len(bellman_ford_classical_times_100),
]

average_times_dijkstra_classical = [
    sum(dijkstra_classical_times_10) / len(dijkstra_classical_times_10),
    sum(dijkstra_classical_times_20) / len(dijkstra_classical_times_20),
    sum(dijkstra_classical_times_30) / len(dijkstra_classical_times_30),
    sum(dijkstra_classical_times_40) / len(dijkstra_classical_times_40),
    sum(dijkstra_classical_times_50) / len(dijkstra_classical_times_50),
    sum(dijkstra_classical_times_60) / len(dijkstra_classical_times_60),
    sum(dijkstra_classical_times_70) / len(dijkstra_classical_times_70),
    sum(dijkstra_classical_times_80) / len(dijkstra_classical_times_80),
    sum(dijkstra_classical_times_90) / len(dijkstra_classical_times_90),
    sum(dijkstra_classical_times_100) / len(dijkstra_classical_times_100),
]

x = np.arange(len(range(10, 110, 10)))  
bar_width = 0.2
plt.bar(x - 1.5 * bar_width, average_times_bellman_ford, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, average_times_dijkstra, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, average_times_bellman_ford_classical, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, average_times_dijkstra_classical, width=bar_width, label='Dijkstra Classical', color='red')
plt.xticks(x, labels=["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])  # Set x-axis ticks to graph sizes
plt.xlabel('Graph Size (V)')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison of Algorithms for Different Graph Sizes')
plt.legend()
plt.show()

#Test Performance for different graph densities with size 20
graph_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
edges = [int((density * 20 * (20 - 1))/2) for density in graph_densities]

bellman_ford_time_0_1, dijkstra_time_0_1, bellman_ford_classical_time_0_1, dijkstra_classical_time_0_1 = experiment_performance_time(50, 10, nodes=20, edges=edges[0])
bellman_ford_time_0_2, dijkstra_time_0_2, bellman_ford_classical_time_0_2, dijkstra_classical_time_0_2 = experiment_performance_time(50, 10, nodes=20, edges=edges[1])
bellman_ford_time_0_3, dijkstra_time_0_3, bellman_ford_classical_time_0_3, dijkstra_classical_time_0_3 = experiment_performance_time(50, 10, nodes=20, edges=edges[2])
bellman_ford_time_0_4, dijkstra_time_0_4, bellman_ford_classical_time_0_4, dijkstra_classical_time_0_4 = experiment_performance_time(50, 10, nodes=20, edges=edges[3])
bellman_ford_time_0_5, dijkstra_time_0_5, bellman_ford_classical_time_0_5, dijkstra_classical_time_0_5 = experiment_performance_time(50, 10, nodes=20, edges=edges[4])
bellman_ford_time_0_6, dijkstra_time_0_6, bellman_ford_classical_time_0_6, dijkstra_classical_time_0_6 = experiment_performance_time(50, 10, nodes=20, edges=edges[5])
bellman_ford_time_0_7, dijkstra_time_0_7, bellman_ford_classical_time_0_7, dijkstra_classical_time_0_7 = experiment_performance_time(50, 10, nodes=20, edges=edges[6])
bellman_ford_time_0_8, dijkstra_time_0_8, bellman_ford_classical_time_0_8, dijkstra_classical_time_0_8 = experiment_performance_time(50, 10, nodes=20, edges=edges[7])
bellman_ford_time_0_9, dijkstra_time_0_9, bellman_ford_classical_time_0_9, dijkstra_classical_time_0_9 = experiment_performance_time(50, 10, nodes=20, edges=edges[8])
bellman_ford_time_1_0, dijkstra_time_1_0, bellman_ford_classical_time_1_0, dijkstra_classical_time_1_0 = experiment_performance_time(50, 10, nodes=20, edges=edges[9])

average_times_bellman_ford = [
    sum(bellman_ford_time_0_1) / len(bellman_ford_time_0_1),
    sum(bellman_ford_time_0_2) / len(bellman_ford_time_0_2),
    sum(bellman_ford_time_0_3) / len(bellman_ford_time_0_3),
    sum(bellman_ford_time_0_4) / len(bellman_ford_time_0_4),
    sum(bellman_ford_time_0_5) / len(bellman_ford_time_0_5),
    sum(bellman_ford_time_0_6) / len(bellman_ford_time_0_6),
    sum(bellman_ford_time_0_7) / len(bellman_ford_time_0_7),
    sum(bellman_ford_time_0_8) / len(bellman_ford_time_0_8),
    sum(bellman_ford_time_0_9) / len(bellman_ford_time_0_9),
    sum(bellman_ford_time_1_0) / len(bellman_ford_time_1_0),
]

average_times_dijkstra = [
    sum(dijkstra_time_0_1) / len(dijkstra_time_0_1),
    sum(dijkstra_time_0_2) / len(dijkstra_time_0_2),
    sum(dijkstra_time_0_3) / len(dijkstra_time_0_3),
    sum(dijkstra_time_0_4) / len(dijkstra_time_0_4),
    sum(dijkstra_time_0_5) / len(dijkstra_time_0_5),
    sum(dijkstra_time_0_6) / len(dijkstra_time_0_6),
    sum(dijkstra_time_0_7) / len(dijkstra_time_0_7),
    sum(dijkstra_time_0_8) / len(dijkstra_time_0_8),
    sum(dijkstra_time_0_9) / len(dijkstra_time_0_9),
    sum(dijkstra_time_1_0) / len(dijkstra_time_1_0),
]

average_times_bellman_ford_classical = [
    sum(bellman_ford_classical_time_0_1) / len(bellman_ford_classical_time_0_1),
    sum(bellman_ford_classical_time_0_2) / len(bellman_ford_classical_time_0_2),
    sum(bellman_ford_classical_time_0_3) / len(bellman_ford_classical_time_0_3),
    sum(bellman_ford_classical_time_0_4) / len(bellman_ford_classical_time_0_4),
    sum(bellman_ford_classical_time_0_5) / len(bellman_ford_classical_time_0_5),
    sum(bellman_ford_classical_time_0_6) / len(bellman_ford_classical_time_0_6),
    sum(bellman_ford_classical_time_0_7) / len(bellman_ford_classical_time_0_7),
    sum(bellman_ford_classical_time_0_8) / len(bellman_ford_classical_time_0_8),
    sum(bellman_ford_classical_time_0_9) / len(bellman_ford_classical_time_0_9),
    sum(bellman_ford_classical_time_1_0) / len(bellman_ford_classical_time_1_0),
]

average_times_dijkstra_classical = [
    sum(dijkstra_classical_time_0_1) / len(dijkstra_classical_time_0_1),
    sum(dijkstra_classical_time_0_2) / len(dijkstra_classical_time_0_2),
    sum(dijkstra_classical_time_0_3) / len(dijkstra_classical_time_0_3),
    sum(dijkstra_classical_time_0_4) / len(dijkstra_classical_time_0_4),
    sum(dijkstra_classical_time_0_5) / len(dijkstra_classical_time_0_5),
    sum(dijkstra_classical_time_0_6) / len(dijkstra_classical_time_0_6),
    sum(dijkstra_classical_time_0_7) / len(dijkstra_classical_time_0_7),
    sum(dijkstra_classical_time_0_8) / len(dijkstra_classical_time_0_8),
    sum(dijkstra_classical_time_0_9) / len(dijkstra_classical_time_0_9),
    sum(dijkstra_classical_time_1_0) / len(dijkstra_classical_time_1_0),
]

# Plot the results in a multi-bar bar chart
x = np.arange(len(graph_densities)) 
bar_width = 0.2
plt.bar(x - 1.5 * bar_width, average_times_bellman_ford, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, average_times_dijkstra, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, average_times_bellman_ford_classical, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, average_times_dijkstra_classical, width=bar_width, label='Dijkstra Classical', color='red')
plt.xticks(x, labels=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])  # Set x-axis ticks to densities
plt.xlabel('Graph Density')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison of Algorithms for Different Graph Densities')
plt.legend()
plt.show()

#Repeat the above experiments for memory usage

#Experiment for different k values
bellman_ford_memory_1, dijkstra_memory_1, bellman_ford_classical_memory_1, dijkstra_classical_memory_1 = experiment_performance_memory(10, 1)
bellman_ford_memory_2, dijkstra_memory_2, bellman_ford_classical_memory_2, dijkstra_classical_memory_2 = experiment_performance_memory(10, 2)
bellman_ford_memory_3, dijkstra_memory_3, bellman_ford_classical_memory_3, dijkstra_classical_memory_3 = experiment_performance_memory(10, 3)
bellman_ford_memory_4, dijkstra_memory_4, bellman_ford_classical_memory_4, dijkstra_classical_memory_4 = experiment_performance_memory(10, 4)
bellman_ford_memory_5, dijkstra_memory_5, bellman_ford_classical_memory_5, dijkstra_classical_memory_5 = experiment_performance_memory(10, 5)
bellman_ford_memory_6, dijkstra_memory_6, bellman_ford_classical_memory_6, dijkstra_classical_memory_6 = experiment_performance_memory(10, 6)
bellman_ford_memory_7, dijkstra_memory_7, bellman_ford_classical_memory_7, dijkstra_classical_memory_7 = experiment_performance_memory(10, 7)
bellman_ford_memory_8, dijkstra_memory_8, bellman_ford_classical_memory_8, dijkstra_classical_memory_8 = experiment_performance_memory(10, 8)
bellman_ford_memory_9, dijkstra_memory_9, bellman_ford_classical_memory_9, dijkstra_classical_memory_9 = experiment_performance_memory(10, 9)
bellman_ford_memory_10, dijkstra_memory_10, bellman_ford_classical_memory_10, dijkstra_classical_memory_10 = experiment_performance_memory(10, 10)

bellman_ford_memory_average = [
    sum(bellman_ford_memory_1) / len(bellman_ford_memory_1),
    sum(bellman_ford_memory_2) / len(bellman_ford_memory_2),
    sum(bellman_ford_memory_3) / len(bellman_ford_memory_3),
    sum(bellman_ford_memory_4) / len(bellman_ford_memory_4),
    sum(bellman_ford_memory_5) / len(bellman_ford_memory_5),
    sum(bellman_ford_memory_6) / len(bellman_ford_memory_6),
    sum(bellman_ford_memory_7) / len(bellman_ford_memory_7),
    sum(bellman_ford_memory_8) / len(bellman_ford_memory_8),
    sum(bellman_ford_memory_9) / len(bellman_ford_memory_9),
    sum(bellman_ford_memory_10) / len(bellman_ford_memory_10),
]

dijkstra_memory_average = [
    sum(dijkstra_memory_1) / len(dijkstra_memory_1),
    sum(dijkstra_memory_2) / len(dijkstra_memory_2),
    sum(dijkstra_memory_3) / len(dijkstra_memory_3),
    sum(dijkstra_memory_4) / len(dijkstra_memory_4),
    sum(dijkstra_memory_5) / len(dijkstra_memory_5),
    sum(dijkstra_memory_6) / len(dijkstra_memory_6),
    sum(dijkstra_memory_7) / len(dijkstra_memory_7),
    sum(dijkstra_memory_8) / len(dijkstra_memory_8),
    sum(dijkstra_memory_9) / len(dijkstra_memory_9),
    sum(dijkstra_memory_10) / len(dijkstra_memory_10),
]

bellman_ford_classical_memory_average = [
    sum(bellman_ford_classical_memory_1) / len(bellman_ford_classical_memory_1),
    sum(bellman_ford_classical_memory_2) / len(bellman_ford_classical_memory_2),
    sum(bellman_ford_classical_memory_3) / len(bellman_ford_classical_memory_3),
    sum(bellman_ford_classical_memory_4) / len(bellman_ford_classical_memory_4),
    sum(bellman_ford_classical_memory_5) / len(bellman_ford_classical_memory_5),
    sum(bellman_ford_classical_memory_6) / len(bellman_ford_classical_memory_6),
    sum(bellman_ford_classical_memory_7) / len(bellman_ford_classical_memory_7),
    sum(bellman_ford_classical_memory_8) / len(bellman_ford_classical_memory_8),
    sum(bellman_ford_classical_memory_9) / len(bellman_ford_classical_memory_9),
    sum(bellman_ford_classical_memory_10) / len(bellman_ford_classical_memory_10),
]

dijkstra_classical_memory_average = [
    sum(dijkstra_classical_memory_1) / len(dijkstra_classical_memory_1),
    sum(dijkstra_classical_memory_2) / len(dijkstra_classical_memory_2),
    sum(dijkstra_classical_memory_3) / len(dijkstra_classical_memory_3),
    sum(dijkstra_classical_memory_4) / len(dijkstra_classical_memory_4),
    sum(dijkstra_classical_memory_5) / len(dijkstra_classical_memory_5),
    sum(dijkstra_classical_memory_6) / len(dijkstra_classical_memory_6),
    sum(dijkstra_classical_memory_7) / len(dijkstra_classical_memory_7),
    sum(dijkstra_classical_memory_8) / len(dijkstra_classical_memory_8),
    sum(dijkstra_classical_memory_9) / len(dijkstra_classical_memory_9),
    sum(dijkstra_classical_memory_10) / len(dijkstra_classical_memory_10),
]

#Plot the results in a bar chart
x = np.arange(1, 11)  
bar_width = 0.2
plt.bar(x - 1.5 * bar_width, bellman_ford_memory_average, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, dijkstra_memory_average, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, bellman_ford_classical_memory_average, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, dijkstra_classical_memory_average, width=bar_width, label='Dijkstra Classical', color='red')
plt.xticks(x, labels=[str(i) for i in x])
plt.xlabel('k')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage Comparison of Algorithms for Different k Values')
plt.legend()
plt.show()

#experiment for different graph sizes with density 0.5
graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
edges = [int((0.5 * size * (size - 1))/2) for size in graph_sizes]

bellman_ford_memory_10, dijkstra_memory_10, bellman_ford_classical_memory_10, dijkstra_classical_memory_10 = experiment_performance_memory(50, 10, nodes=10, edges=edges[0])
bellman_ford_memory_20, dijkstra_memory_20, bellman_ford_classical_memory_20, dijkstra_classical_memory_20 = experiment_performance_memory(50, 10, nodes=20, edges=edges[0])
bellman_ford_memory_30, dijkstra_memory_30, bellman_ford_classical_memory_30, dijkstra_classical_memory_30 = experiment_performance_memory(50, 10, nodes=30, edges=edges[0])
bellman_ford_memory_40, dijkstra_memory_40, bellman_ford_classical_memory_40, dijkstra_classical_memory_40 = experiment_performance_memory(50, 10, nodes=40, edges=edges[0])
bellman_ford_memory_50, dijkstra_memory_50, bellman_ford_classical_memory_50, dijkstra_classical_memory_50 = experiment_performance_memory(50, 10, nodes=50, edges=edges[4])
bellman_ford_memory_60, dijkstra_memory_60, bellman_ford_classical_memory_60, dijkstra_classical_memory_60 = experiment_performance_memory(50, 10, nodes=60, edges=edges[5])
bellman_ford_memory_70, dijkstra_memory_70, bellman_ford_classical_memory_70, dijkstra_classical_memory_70 = experiment_performance_memory(50, 10, nodes=70, edges=edges[6])
bellman_ford_memory_80, dijkstra_memory_80, bellman_ford_classical_memory_80, dijkstra_classical_memory_80 = experiment_performance_memory(50, 10, nodes=80, edges=edges[7])
bellman_ford_memory_90, dijkstra_memory_90, bellman_ford_classical_memory_90, dijkstra_classical_memory_90 = experiment_performance_memory(50, 10, nodes=90, edges=edges[8])
bellman_ford_memory_100, dijkstra_memory_100, bellman_ford_classical_memory_100, dijkstra_classical_memory_100 = experiment_performance_memory(50, 10, nodes=100, edges=edges[9])

bellman_ford_memory_average = [
    sum(bellman_ford_memory_10) / len(bellman_ford_memory_10),
    sum(bellman_ford_memory_20) / len(bellman_ford_memory_20),
    sum(bellman_ford_memory_30) / len(bellman_ford_memory_30),
    sum(bellman_ford_memory_40) / len(bellman_ford_memory_40),
    sum(bellman_ford_memory_50) / len(bellman_ford_memory_50),
    sum(bellman_ford_memory_60) / len(bellman_ford_memory_60),
    sum(bellman_ford_memory_70) / len(bellman_ford_memory_70),
    sum(bellman_ford_memory_80) / len(bellman_ford_memory_80),
    sum(bellman_ford_memory_90) / len(bellman_ford_memory_90),
    sum(bellman_ford_memory_100) / len(bellman_ford_memory_100),
]

dijkstra_memory_average = [
    sum(dijkstra_memory_10) / len(dijkstra_memory_10),
    sum(dijkstra_memory_20) / len(dijkstra_memory_20),
    sum(dijkstra_memory_30) / len(dijkstra_memory_30),
    sum(dijkstra_memory_40) / len(dijkstra_memory_40),
    sum(dijkstra_memory_50) / len(dijkstra_memory_50),
    sum(dijkstra_memory_60) / len(dijkstra_memory_60),
    sum(dijkstra_memory_70) / len(dijkstra_memory_70),
    sum(dijkstra_memory_80) / len(dijkstra_memory_80),
    sum(dijkstra_memory_90) / len(dijkstra_memory_90),
    sum(dijkstra_memory_100) / len(dijkstra_memory_100),
]

bellman_ford_classical_memory_average = [
    sum(bellman_ford_classical_memory_10) / len(bellman_ford_classical_memory_10),
    sum(bellman_ford_classical_memory_20) / len(bellman_ford_classical_memory_20),
    sum(bellman_ford_classical_memory_30) / len(bellman_ford_classical_memory_30),
    sum(bellman_ford_classical_memory_40) / len(bellman_ford_classical_memory_40),
    sum(bellman_ford_classical_memory_50) / len(bellman_ford_classical_memory_50),
    sum(bellman_ford_classical_memory_60) / len(bellman_ford_classical_memory_60),
    sum(bellman_ford_classical_memory_70) / len(bellman_ford_classical_memory_70),
    sum(bellman_ford_classical_memory_80) / len(bellman_ford_classical_memory_80),
    sum(bellman_ford_classical_memory_90) / len(bellman_ford_classical_memory_90),
    sum(bellman_ford_classical_memory_100) / len(bellman_ford_classical_memory_100),
]

dijkstra_classical_memory_average = [
    sum(dijkstra_classical_memory_10) / len(dijkstra_classical_memory_10),
    sum(dijkstra_classical_memory_20) / len(dijkstra_classical_memory_20),
    sum(dijkstra_classical_memory_30) / len(dijkstra_classical_memory_30),
    sum(dijkstra_classical_memory_40) / len(dijkstra_classical_memory_40),
    sum(dijkstra_classical_memory_50) / len(dijkstra_classical_memory_50),
    sum(dijkstra_classical_memory_60) / len(dijkstra_classical_memory_60),
    sum(dijkstra_classical_memory_70) / len(dijkstra_classical_memory_70),
    sum(dijkstra_classical_memory_80) / len(dijkstra_classical_memory_80),
    sum(dijkstra_classical_memory_90) / len(dijkstra_classical_memory_90),
    sum(dijkstra_classical_memory_100) / len(dijkstra_classical_memory_100),
]

#Plot the results in a bar chart
x = np.arange(1, 11)  
bar_width = 0.2
plt.bar(x - 1.5 * bar_width, bellman_ford_memory_average, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, dijkstra_memory_average, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, bellman_ford_classical_memory_average, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, dijkstra_classical_memory_average, width=bar_width, label='Dijkstra Classical', color='red')
plt.xticks(x, labels=[str(i) for i in x])
plt.xlabel('k')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage Comparison of Algorithms for Different Graph Sizes')
plt.legend()
plt.show()

#Experiment for different graph densities with size 20
graph_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
edges = [int((density * 20 * (20 - 1))/2) for density in graph_densities]

bellman_ford_memory_0_1, dijkstra_memory_0_1, bellman_ford_classical_memory_0_1, dijkstra_classical_memory_0_1 = experiment_performance_memory(50, 20, nodes=20, edges=edges[0])
bellman_ford_memory_0_2, dijkstra_memory_0_2, bellman_ford_classical_memory_0_2, dijkstra_classical_memory_0_2 = experiment_performance_memory(50, 20, nodes=20, edges=edges[1])
bellman_ford_memory_0_3, dijkstra_memory_0_3, bellman_ford_classical_memory_0_3, dijkstra_classical_memory_0_3 = experiment_performance_memory(50, 20, nodes=20, edges=edges[2])
bellman_ford_memory_0_4, dijkstra_memory_0_4, bellman_ford_classical_memory_0_4, dijkstra_classical_memory_0_4 = experiment_performance_memory(50, 20, nodes=20, edges=edges[3])
bellman_ford_memory_0_5, dijkstra_memory_0_5, bellman_ford_classical_memory_0_5, dijkstra_classical_memory_0_5 = experiment_performance_memory(50, 20, nodes=20, edges=edges[4])
bellman_ford_memory_0_6, dijkstra_memory_0_6, bellman_ford_classical_memory_0_6, dijkstra_classical_memory_0_6 = experiment_performance_memory(50, 20, nodes=20, edges=edges[5])
bellman_ford_memory_0_7, dijkstra_memory_0_7, bellman_ford_classical_memory_0_7, dijkstra_classical_memory_0_7 = experiment_performance_memory(50, 20, nodes=20, edges=edges[6])
bellman_ford_memory_0_8, dijkstra_memory_0_8, bellman_ford_classical_memory_0_8, dijkstra_classical_memory_0_8 = experiment_performance_memory(50, 20, nodes=20, edges=edges[7])
bellman_ford_memory_0_9, dijkstra_memory_0_9, bellman_ford_classical_memory_0_9, dijkstra_classical_memory_0_9 = experiment_performance_memory(50, 20, nodes=20, edges=edges[8])
bellman_ford_memory_1_0, dijkstra_memory_1_0, bellman_ford_classical_memory_1_0, dijkstra_classical_memory_1_0 = experiment_performance_memory(50, 20, nodes=20, edges=edges[9])

bellman_ford_memory_average = [
    sum(bellman_ford_memory_0_1) / len(bellman_ford_memory_0_1),
    sum(bellman_ford_memory_0_2) / len(bellman_ford_memory_0_2),
    sum(bellman_ford_memory_0_3) / len(bellman_ford_memory_0_3),
    sum(bellman_ford_memory_0_4) / len(bellman_ford_memory_0_4),
    sum(bellman_ford_memory_0_5) / len(bellman_ford_memory_0_5),
    sum(bellman_ford_memory_0_6) / len(bellman_ford_memory_0_6),
    sum(bellman_ford_memory_0_7) / len(bellman_ford_memory_0_7),
    sum(bellman_ford_memory_0_8) / len(bellman_ford_memory_0_8),
    sum(bellman_ford_memory_0_9) / len(bellman_ford_memory_0_9),
    sum(bellman_ford_memory_1_0) / len(bellman_ford_memory_1_0),
]

dijkstra_memory_average = [
    sum(dijkstra_memory_0_1) / len(dijkstra_memory_0_1),
    sum(dijkstra_memory_0_2) / len(dijkstra_memory_0_2),
    sum(dijkstra_memory_0_3) / len(dijkstra_memory_0_3),
    sum(dijkstra_memory_0_4) / len(dijkstra_memory_0_4),
    sum(dijkstra_memory_0_5) / len(dijkstra_memory_0_5),
    sum(dijkstra_memory_0_6) / len(dijkstra_memory_0_6),
    sum(dijkstra_memory_0_7) / len(dijkstra_memory_0_7),
    sum(dijkstra_memory_0_8) / len(dijkstra_memory_0_8),
    sum(dijkstra_memory_0_9) / len(dijkstra_memory_0_9),
    sum(dijkstra_memory_1_0) / len(dijkstra_memory_1_0),
]

bellman_ford_classical_memory_average = [
    sum(bellman_ford_classical_memory_0_1) / len(bellman_ford_classical_memory_0_1),
    sum(bellman_ford_classical_memory_0_2) / len(bellman_ford_classical_memory_0_2),
    sum(bellman_ford_classical_memory_0_3) / len(bellman_ford_classical_memory_0_3),
    sum(bellman_ford_classical_memory_0_4) / len(bellman_ford_classical_memory_0_4),
    sum(bellman_ford_classical_memory_0_5) / len(bellman_ford_classical_memory_0_5),
    sum(bellman_ford_classical_memory_0_6) / len(bellman_ford_classical_memory_0_6),
    sum(bellman_ford_classical_memory_0_7) / len(bellman_ford_classical_memory_0_7),
    sum(bellman_ford_classical_memory_0_8) / len(bellman_ford_classical_memory_0_8),
    sum(bellman_ford_classical_memory_0_9) / len(bellman_ford_classical_memory_0_9),
    sum(bellman_ford_classical_memory_1_0) / len(bellman_ford_classical_memory_1_0),
]

dijkstra_classical_memory_average = [
    sum(dijkstra_classical_memory_0_1) / len(dijkstra_classical_memory_0_1),
    sum(dijkstra_classical_memory_0_2) / len(dijkstra_classical_memory_0_2),
    sum(dijkstra_classical_memory_0_3) / len(dijkstra_classical_memory_0_3),
    sum(dijkstra_classical_memory_0_4) / len(dijkstra_classical_memory_0_4),
    sum(dijkstra_classical_memory_0_5) / len(dijkstra_classical_memory_0_5),
    sum(dijkstra_classical_memory_0_6) / len(dijkstra_classical_memory_0_6),
    sum(dijkstra_classical_memory_0_7) / len(dijkstra_classical_memory_0_7),
    sum(dijkstra_classical_memory_0_8) / len(dijkstra_classical_memory_0_8),
    sum(dijkstra_classical_memory_0_9) / len(dijkstra_classical_memory_0_9),
    sum(dijkstra_classical_memory_1_0) / len(dijkstra_classical_memory_1_0),
]

# Plot the results in a bar chart
x = np.arange(len(graph_densities))
bar_width = 0.2

plt.bar(x - 1.5 * bar_width, bellman_ford_memory_average, width=bar_width, label='Bellman-Ford', color='green')
plt.bar(x - 0.5 * bar_width, dijkstra_memory_average, width=bar_width, label='Dijkstra', color='blue')
plt.bar(x + 0.5 * bar_width, bellman_ford_classical_memory_average, width=bar_width, label='Bellman-Ford Classical', color='orange')
plt.bar(x + 1.5 * bar_width, dijkstra_classical_memory_average, width=bar_width, label='Dijkstra Classical', color='red')

plt.xticks(x, labels=[str(d) for d in graph_densities])  
plt.xlabel('Graph Density')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage Comparison of Algorithms for Different Graph Densities')
plt.legend()
plt.show()