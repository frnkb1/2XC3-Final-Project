import heapq
import math
import csv
import pandas as pd
import timeit 
import random
import matplotlib.pyplot as plt

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


"""dijkstra Algorithm for Directed Graphs"""
def dijkstra(graph: Graph, source: int):
    distances = {node: float('inf') for node in range(graph.get_size())}
    distances[source] = 0
    paths = {node: [] for node in range(graph.get_size())}
    paths[source] = [source]

    priority_queue = []
    heapq.heappush(priority_queue, (0, source))

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue  # Skip outdated entries

        for neighbor in graph.adjacent_nodes(current_node):
            weight = graph.get_weight(current_node, neighbor)
            if weight is None:
                continue  # No edge exists

            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, paths

def euclidean_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def get_stations_data(stations_csv_filename: str):
    data = {}
    with open(stations_csv_filename, mode='r') as file:
        csv_reader = csv.DictReader(file)  # Use DictReader to access columns by name
        for row in csv_reader:
            sid = int(row['id'])  # Assuming 'id' is the column name for station ID
            latitude = float(row['latitude'])  # Assuming 'latitude' is the column name
            longitude = float(row['longitude'])  # Assuming 'longitude' is the column name
            data[sid] = [latitude, longitude]
    return data

def add_edges_from_csv(graph: Graph, connections_csv_filename: str, stations_data):
    with open(connections_csv_filename, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            station1, station2, line, time = map(int, row)  
            edge_weight = euclidean_distance(lat1=stations_data[station1][0], lon1=stations_data[station1][1], 
                                             lat2=stations_data[station2][0], lon2=stations_data[station2][1] )
            
            graph.add_edge(src=station1, dst=station2, weight=edge_weight)
            # add twice because the graph type is directed
            graph.add_edge(src=station2, dst=station1, weight=edge_weight)

def heuristic(stations_data, goal):
    goal_latitude = stations_data[goal][0]
    goal_longitude = stations_data[goal][1]
    h = {}
    for sid, values in stations_data.items():
        h[sid] = euclidean_distance(lon1=goal_longitude, lat1=goal_latitude, lon2=values[1], lat2=values[0])
    return h

from collections import defaultdict
import csv

def count_lines_used_from_csv(csv_file, path):
    # Build the line_to_stations mapping from the CSV file
    line_to_stations = defaultdict(set)
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            station1 = int(row['station1'])
            station2 = int(row['station2'])
            line = int(row['line'])
            if station1 not in line_to_stations[line]:
                line_to_stations[line].add(station1)
            if station2 not in line_to_stations[line]:
                line_to_stations[line].add(station2)

    # Count the number of lines used by the path
    if not path:
        return 0
    used_lines = 0
    current_line = None
    for i in range(len(path)):
        station = path[i]
        if current_line is None:
            # First station, pick any line it belongs to
            for line, stations in line_to_stations.items():
                if station in stations:
                    current_line = line
                    used_lines += 1
                    break
        elif station not in line_to_stations[current_line]:
            # Find a new line that includes this station
            for line, stations in line_to_stations.items():
                if station in stations:
                    current_line = line
                    used_lines += 1
                    break
    return used_lines

def experiment():
    # Data initialization
    connections_csv = 'london_connections.csv'
    stations_csv = 'london_stations.csv'

    stations_data = get_stations_data(stations_csv)
    london_subway_graph = Graph(1000)
    # Graph construction
    add_edges_from_csv(graph=london_subway_graph, connections_csv_filename= connections_csv, stations_data=stations_data)

    heuristic_for_all_goal = {}
    # Precompute heuristics for all possible destinations
    for key in stations_data:
        heuristic_for_all_goal[key]= heuristic(stations_data, key)
    # Timing arrays
    a_star_run_array = []
    a_star_each_run_arry = []
    dijkstra_run_array = []
    dijkstra_each_run_arry  = []

    # A* timing block
    A_star_start = timeit.default_timer()
    for src in stations_data:
        for dst in stations_data:
            if src != dst:
                each_start = timeit.default_timer()
                run = A_star(london_subway_graph, src, dst, heuristic_for_all_goal[dst])
                each_end = timeit.default_timer()
                a_star_each_run_arry.append(each_end-each_start)
                a_star_run_array.append(each_end - A_star_start)
    A_star_end = timeit.default_timer()
    a_star_run_array.append(timeit.default_timer() -  A_star_start)

    # Dijkstra timing block
    dijkstra_start = timeit.default_timer()
    for src in stations_data:
        for dst in stations_data:
            if src != dst:
                each_start = timeit.default_timer()
                run = dijkstra(london_subway_graph, src)
                each_end = timeit.default_timer()
                dijkstra_each_run_arry.append(each_end-each_start)
                dijkstra_run_array.append(each_end - dijkstra_start)
    dijkstra_end = timeit.default_timer()
    dijkstra_run_array.append(timeit.default_timer() - dijkstra_start)

    print("A* sum time:", A_star_end - A_star_start)
    print("Dijkstra's sum time:", dijkstra_end - dijkstra_start)

    # Plotting cumulative performance
    plt.plot(a_star_run_array, label='A_star')
    plt.plot(dijkstra_run_array, label="Dijkstra's")
    plt.xlabel('Number of Paths Calculated')
    plt.ylabel('Time (seconds)')
    plt.title("Aggregate Performance Comparison: A* vs Dijkstra's")
    plt.legend()
    plt.show()

    
    plt.figure(figsize=(30, 6))  # Set a wider figure: 16 inches wide, 6 inches tall

    # Plot both arrays as lines
    plt.plot(a_star_each_run_arry, label='A*', linestyle='-', marker='')  
    plt.plot(dijkstra_each_run_arry, label="Dijkstra's", linestyle='-', marker='')

    # Add axis labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Time')
    plt.title("Performance Comparison Per Iteration: A* vs Dijkstra's")
    plt.legend()

    # Display the plot
    plt.show()
    
    a_star_random_run = {}
    dijkstra_random_run = {}

    for _ in range(100):
        u = random.randint(0, 303)
        v = random.randint(0, 303)

        if u in stations_data and v in stations_data:
            if (u, v) not in a_star_random_run and (v, u) not in dijkstra_random_run:

                # A* timing
                A_star_start = timeit.default_timer()
                a = A_star(graph=london_subway_graph, source=u, destination=v, heuristic=heuristic_for_all_goal[v])
                A_star_end = timeit.default_timer()
                a_star_random_run[(u, v)] = A_star_end - A_star_start

                # Dijkstra timing
                dijkstra_start = timeit.default_timer()
                b = dijkstra(graph=london_subway_graph, source=u)
                dijkstra_end = timeit.default_timer()
                dijkstra_random_run[(u, v)] = dijkstra_end - dijkstra_start
    # A* Data
    x_labels_astar = list(a_star_random_run.keys())
    y_values_astar = list(a_star_random_run.values())

    # Dijkstra Data
    x_labels_dijkstra = list(dijkstra_random_run.keys())
    y_values_dijkstra = list(dijkstra_random_run.values())

    # Plotting
    plt.figure(figsize=(25, 6))  
    # Plot A*
    plt.plot(range(len(x_labels_astar)), y_values_astar, label='A*', linestyle='-', marker='o')
    # Plot Dijkstra
    plt.plot(range(len(x_labels_dijkstra)), y_values_dijkstra, label="Dijkstra's", linestyle='-', marker='x')
    # X-axis labels
    plt.xticks(ticks=range(len(x_labels_astar)), labels=[f"{u}->{v}" for u, v in x_labels_astar], rotation=90)
    # Labels and title
    plt.xlabel('Source → Destination')
    plt.ylabel('Time (seconds)')
    plt.title("A* vs Dijkstra's Runtime on Random Pairs")
    plt.legend()
    plt.tight_layout()  
    plt.show()


    # A_star_start = timeit.default_timer()
    # a = A_star(graph=london_subway_graph, source=152, destination=243, heuristic=heuristic_for_all_goal[243])
    # A_star_end = timeit.default_timer()
    # print(A_star_end - A_star_start)

    # # Dijkstra timing
    # dijkstra_start = timeit.default_timer()
    # b = dijkstra(graph=london_subway_graph, source=152)
    # dijkstra_end = timeit.default_timer()
    # print(dijkstra_end - dijkstra_start)



def extra_experiment():
    connections_csv = 'london_connections.csv'
    stations_csv = 'london_stations.csv'

    stations_data = get_stations_data(stations_csv)
    london_subway_graph = Graph(1000)

    add_edges_from_csv(graph=london_subway_graph, connections_csv_filename= connections_csv, stations_data=stations_data)

    heuristic_for_all_goal = {}
    
    for key in stations_data:
        heuristic_for_all_goal[key]= heuristic(stations_data, key)
    
    one_line_a = []
    two_line_a = []
    more_than_two_a = []
    one_line_d = []
    two_line_d = []
    more_than_two_d = []

    while len(one_line_a)!= 50 or len(two_line_a) != 50 or len(more_than_two_a) != 50:
        u = random.randint(0, 303)
        v = random.randint(0, 303)

        if u in stations_data and v in stations_data:
            A_star_start = timeit.default_timer()
            a = A_star(graph=london_subway_graph, source=u, destination=v, heuristic=heuristic_for_all_goal[v])
            A_star_end = timeit.default_timer()

            # Dijkstra timing
            dijkstra_start = timeit.default_timer()
            b = dijkstra(graph=london_subway_graph, source=u)
            dijkstra_end = timeit.default_timer()

            num_of_line = count_lines_used_from_csv(connections_csv, a[1])
            if num_of_line == 1 and len(one_line_a) < 50:
                one_line_a.append(A_star_end-A_star_start)
                one_line_d.append(dijkstra_end-dijkstra_start)
            elif num_of_line == 2 and len(two_line_a) < 50:
                two_line_a.append(A_star_end-A_star_start)
                two_line_d.append(dijkstra_end-dijkstra_start)
            elif num_of_line >=3 and len(more_than_two_a) < 50:
                more_than_two_a.append(A_star_end-A_star_start)
                more_than_two_d.append(dijkstra_end-dijkstra_start)
    
    plt.figure(figsize=(10, 6))

    # Plot for A* and Dijkstra with distinct colors/linestyles
    plt.plot(one_line_a, label='A* (Same Line)', color='blue', linestyle='-')
    plt.plot(two_line_a, label='A* (Adjacent Line)', color='green', linestyle='-')
    plt.plot(more_than_two_a, label='A* (More Than One Transfer)', color='red', linestyle='-')

    plt.plot(one_line_d, label='Dijkstra (Same Line)', color='cyan', linestyle='--')
    plt.plot(two_line_d, label='Dijkstra (Adjacent Line)', color='orange', linestyle='--')
    plt.plot(more_than_two_d, label='Dijkstra (More Than One Transfer)', color='magenta', linestyle='--')
    # Labels and legend
    plt.title("A* and Dijkstra Runtime Comparison for Different Path Types")
    plt.xlabel("Number of Tests")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    # Show the plot
    plt.show()


experiment()
#extra_experiment()


