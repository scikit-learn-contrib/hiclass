import numpy as np

from hiclass import DirectedAcyclicGraph


def test_add_node():
    n_rows = 3
    dag = DirectedAcyclicGraph(n_rows)
    dag.add_node("node1")
    dag.add_node("node2")
    dag.add_node("node1")
    dag.add_node("node2")
    assert 3 == len(dag.nodes)
    assert "root" in dag.nodes
    assert "node1" in dag.nodes
    assert "node2" in dag.nodes


def test_add_path():
    paths = np.array([
        ["a", "c", "d"],
        ["b", "c", "e"],
        ["a", "c", "f"],
        ["c", "", ""],
        ["a", "c", "d"],
        ["b", "c", "e"],
        ["a", "c", "f"],
        ["c", "", ""],
        ["", "", ""],
    ])
    rows = paths.shape[0]
    dag = DirectedAcyclicGraph(rows)
    for row in range(rows):
        path = paths[row, :]
        dag.add_path(path)
    assert 8 == len(dag.nodes)
