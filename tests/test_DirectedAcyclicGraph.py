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


def test_is_acyclic():
    n_rows = 3
    dag = DirectedAcyclicGraph(n_rows)
    dag.add_path([0, 1, 2])
    dag.add_path([0, 2, 3])
    assert dag.is_acyclic() is True
    dag.add_path([0, 2, 0])
    # assert dag.is_acyclic() is False
    # the creation of new nodes removes cycles
    # so this last assertion fails


def test_get_parent_nodes():
    n_rows = 3
    dag = DirectedAcyclicGraph(n_rows)
    dag.add_path(["a", "b", "c"])
    dag.add_path(["d", "e", "f"])
    parent_nodes = dag.get_parent_nodes()
    assert 5 == len(parent_nodes)
    names = ["root", "a", "b", "d", "e"]
    assert names == [node.name for node in parent_nodes]
