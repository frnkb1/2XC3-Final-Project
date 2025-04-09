import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import tracemalloc
from copy import deepcopy

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

    priority_queue = [(0, source)]

    while priority_queue:
        priority_queue.sort(key=lambda x: x[0])
        current_distance, current_node = priority_queue.pop(0)  

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
                priority_queue.append((distance, neighbor))

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

    priority_queue = [(0, source)]

    while priority_queue:
        priority_queue.sort(key=lambda x: x[0])
        current_distance, current_node = priority_queue.pop(0)

        for neighbor in graph_obj.graph[current_node]:
            weight = graph_obj.weight[(current_node, neighbor)]
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                priority_queue.append((distance, neighbor))

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
        source = np.random.randint(0, 100)
        
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
        source = np.random.randint(0, 100)

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

bellman_accuracies, djikstra_accuracies = experiment_accuracy(10, 9)
print(bellman_accuracies)
print(djikstra_accuracies)

bellman_ford_times, dijkstra_times, bellman_ford_classical_times, dijkstra_classical_times = experiment_performance_time(10, 5)
print(np.mean(bellman_ford_times))
print(np.mean(dijkstra_times))
print(np.mean(bellman_ford_classical_times))
print(np.mean(dijkstra_classical_times))

bellman_ford_memory, dijkstra_memory, bellman_ford_classical_memory, dijkstra_classical_memory = experiment_performance_memory(10, 5)
print(np.mean(bellman_ford_memory))
print(np.mean(dijkstra_memory))
print(np.mean(bellman_ford_classical_memory))
print(np.mean(dijkstra_classical_memory))