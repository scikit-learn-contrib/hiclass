from hiclass import Node


def test_add_successor():
    n_rows = 3
    name = "root"
    node = Node(n_rows, name)
    assert node.name == "root"
    successor1 = node.add_successor("node1")
    successor2 = node.add_successor("node2")
    assert successor1 == node.add_successor("node1")
    assert successor2 == node.add_successor("node2")
    assert n_rows == node.n_rows
    assert 2 == len(node.children)
