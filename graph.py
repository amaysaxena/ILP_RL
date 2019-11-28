from collections import defaultdict
from itertools import product
import random

class Graph(object):
    """Undirected graph in adjacency list format.
    """

    def __init__(self, n_vertices):
        self.n_vertices = n_vertices
        self.n_edges = 0
        self.neighbours = defaultdict(set)
        self.weights = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        self.edges = []

    def add_edge(self, u, v, weight=1):
        self.neighbours[u].add(v)
        self.neighbours[v].add(u)
        self.edges.append((min(u, v), max(u, v)))
        self.weights[min(u, v)][max(u, v)] = weight
        self.n_edges += 1

    def weight(self, u, v):
        return self.weights[min(u, v)][max(u, v)]

    def edge_exists(self, u, v):
        return (min(u, v), max(u, v)) in self.edges

def add_random_edges(graph, n_edges_to_add, weight_population):
    """
    Add n_edges_to_add new edges to graph sampled uniformly from all edges
    not present in the graph.
    """
    unseen_edges = [(i, j) for (i, j) in product(range(graph.n_vertices), repeat=2)
        if j not in graph.neighbours[i]]
    assert n_edges_to_add <= len(unseen_edges)
    new_edges = random.sample(unseen_edges, n_edges_to_add)
    for u, v in new_edges:
        graph.add_edge(u, v, random.choice(weight_population))
    return graph

def random_tree(n_vertices, weight_population):
    """
    Samples a tree uniformly at random from the set of all trees
    with n_vertices vertices.

    Returns graph in the form of an adjacency list.
    """
    tree = Graph(n_vertices)

    vertices = list(range(n_vertices))
    unvisited, random_tree = set(vertices), set()

    current_node = random.sample(unvisited, 1).pop()
    unvisited.remove(current_node)
    random_tree.add(current_node)
    while unvisited:
        neighbor_node = random.sample(vertices, 1).pop()
        if neighbor_node not in random_tree:
            tree.add_edge(current_node, neighbor_node, random.choice(weight_population))
            unvisited.remove(neighbor_node)
            random_tree.add(neighbor_node)
        current_node = neighbor_node
    return tree

def random_connected_graph(n_vertices, n_edges, weight_population):
    """
    Samples a graph uniformly at random from the set of all connected graphs
    with n_vertices vertices and n_edges edges.

    Returns graph in the form of an adjacency list.
    """
    graph = random_tree(n_vertices, weight_population)
    return add_random_edges(graph, n_edges - (n_vertices - 1), weight_population)
