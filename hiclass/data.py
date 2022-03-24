"""Helper functions for data manipulation and graph creation."""
from typing import Union, Tuple
import itertools

import networkx as nx
import numpy as np

PLACEHOLDER_LABEL_NUMERIC = -1
PLACEHOLDER_LABEL_CAT = ""
NODE_TYPE = Union[int, str, np.str_]
CATEGORICAL_ROOT_NODE = "HICLASS::__ROOT__"
SEPARATOR = "HICLAS::__SEPARATOR__"

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set("buif")


def graph_from_edge_pairs(
    file: str, delimiter: str = ",", skip_header: int = 1
) -> nx.DiGraph:
    """
    Create a DAG from a file containing (parent, child) pairs.

    Parameters
    ----------
    file : str
        File containing the edge pairs.
    delimiter : str, default=','
        The delimiter of the file.
    skip_header : int, default=1
        The number of rows header rows that should be skipped.

    Returns
    -------
    graph : nx.DiGraph
        The graph with all corresponding edges and their nodes.
    """
    edges = np.genfromtxt(file, delimiter=delimiter, skip_header=skip_header)
    return nx.from_edgelist(edges, create_using=nx.DiGraph)


def graph_from_hierarchical_labels(
    data: np.ndarray, placeholder: Union[str, int, None] = None
) -> nx.DiGraph:
    """
    Construct a DAG from hierarchical labels.

    In the case that multiple root nodes are found, a new root node is inserted and all previous root nodes are
    connected to it.

    Parameters
    ----------
    data : np.array
        Hierarchical labels, formatted to be (row, col). The columns should be ordered
        from least specific to most specific class. If some columns are invalid (i.e. there are columns with a
        number of labels lower than the number of columns), then they should be marked by a placeholder.
    placeholder : str or int, default=None
        Value for non-existent nodes in the data. Has to match data type of :code:`data`

    Returns
    -------
    graph : DiGraph
        The graph with all corresponding edges and their nodes.
    """
    g = nx.DiGraph()
    nodes = np.unique(data)
    if placeholder in nodes:
        nodes = list(nodes)
        nodes.remove(placeholder)
    g.add_nodes_from(nodes)

    if data.ndim > 1:
        roots = np.unique(data[:, 0])
    else:
        roots = np.unique(data)
    if len(roots) > 1:
        new_root = max(nodes)
        new_root = new_root + 1 if is_numeric_label(new_root) else CATEGORICAL_ROOT_NODE

        g.add_edges_from(zip(itertools.repeat(new_root), roots))

    col = 0
    if data.ndim > 1:
        while col + 2 <= data.shape[1]:
            valid_edges = np.unique(data[:, col : col + 2], axis=0)
            placeholders = np.any(valid_edges == placeholder, axis=1)
            valid_edges = valid_edges[np.invert(placeholders)]
            g.add_edges_from(valid_edges)
            col += 1
    return g


def convert_categorical_labels_to_numeric(
    data: np.ndarray, placeholder_label: str = PLACEHOLDER_LABEL_CAT
) -> Tuple[np.ndarray, dict]:
    """
    Take a string array and convert the values to enumerated integers.

    Parameters
    ----------
    data : np.array
        Data to be processed.
    placeholder_label : str
        label representing no present class.

    Returns
    -------
    conversion : np.array
        The converted integer array.
    mapping : dict
         Mapping from the original value to the new enumeration.
    """
    categories, indices, mapping = np.unique(
        data, return_index=True, return_inverse=True
    )
    label_to_int = dict(zip(categories, mapping[indices]))
    # map all undefined_labels to -1, implying an undefined output and remove it from the mapping
    if placeholder_label in categories:
        mapping[mapping == label_to_int[placeholder_label]] = PLACEHOLDER_LABEL_NUMERIC
        label_to_int[placeholder_label] = PLACEHOLDER_LABEL_NUMERIC
    return mapping.reshape(data.shape), label_to_int


def find_root(graph: nx.DiGraph) -> NODE_TYPE:
    """Take a graph and return one of its root nodes.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph for which the root should be found.

    Returns
    -------
    root : int or str
        The label of the root node.
    """
    current_node = list(graph.nodes)[0]
    successors = list(graph.predecessors(current_node))
    while len(successors) != 0:
        current_node = successors[0]
        successors = list(graph.predecessors(current_node))
    return current_node


def find_max_depth(graph: nx.DiGraph, root: NODE_TYPE) -> int:
    """Find the maximum depth of a DAG given a node as root.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph for which the root should be found.
    root : int or str
        Node from which the distance should be measured.

    Returns
    -------
    max_depth : int
        The number of nodes in the furthest path starting from the root node. A graph that only has the root would
        return 1.
    """
    # Queue root node and initialize max_depth
    queue = [root]
    max_depth = 0

    while True:

        # node_count indicates the number of nodes at current level
        node_count = len(queue)
        if node_count == 0:
            return max_depth

        max_depth += 1

        # dequeue all nodes from current level
        # and queue nodes from next level
        while node_count > 0:
            node = queue.pop(0)
            for successor in list(graph.successors(node)):
                queue.append(successor)
            node_count -= 1


def minimal_graph_depth(graph: nx.DiGraph) -> int:
    """
    Calculate the minimal depth in which all nodes can be hit.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph to be analyzed.

    Returns
    -------
    depth : int
        The minimal depth.
    """
    max_level = 1
    for level in minimal_per_node_depth(graph).values():
        max_level = max(level, max_level)
    return max_level


def minimal_per_node_depth(graph: nx.DiGraph) -> dict:
    """
    Calculate the minimal depth which is needed to hit a node, for all nodes.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph to be analyzed.

    Returns
    -------
    node_depth : dict
        A mapping for node : depth
    """
    root = find_root(graph)
    node_depth = {root: 1}
    for parent, successors in nx.bfs_successors(graph, root):
        parent_level = node_depth[parent]
        for node in successors:
            if node not in node_depth.keys():
                node_depth[node] = parent_level + 1
    return node_depth


def is_numeric_label(array: np.ndarray) -> bool:
    """Determine whether an array has a numerical label format.

    Supported formats are booleans, unsigned integers, signed integers and floats.

    Parameters
    ----------
    array : np.array
        The array to check.

    Returns
    -------
    result : bool
        True if the array is has a supported format, False otherwise
    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS


def flatten_labels(labels: np.ndarray, placeholder_label: int = -1) -> np.ndarray:
    """Flattens hierarchical labels to only the most specific label per entry.

    Expects the most specific label to be on the rightmost side of the label array for every entry.

    Parameters
    ----------
    labels : np.array
        2d label matrix, formatted to be row, column.
    placeholder_label : int or str
        value describing an undefined label in order to support uneven hierarchies and labels.

    Returns
    -------
    new_labels: np.array
        1d array of the most specific label for each row.
    """
    if len(labels.shape) < 2:
        return labels
    num_labels = len(labels)
    new_labels = np.zeros(num_labels, dtype=labels.dtype)
    for i, hierarchical_label in enumerate(labels):
        for label in hierarchical_label[::-1]:
            if label != placeholder_label:
                new_labels[i] = label
                break
    return new_labels
