import heapq

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
