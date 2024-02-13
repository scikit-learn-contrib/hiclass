from hiclass.Node import Node


class DirectedAcyclicGraph:
    """
        Manages the directed acyclic graph used in HiClass.

        It tries to copy networkx API as much as possible,
        but extends it by adding support for multiple nodes with the same name,
        as long as they have different predecessors.
    """


    def __init__(self, n_rows):
        """
        Initialize a directed acyclic graph.

        Parameters
        ----------
        n_rows : int
            The number of rows in x and y, i.e., the features and labels matrices.
        """
        self.root = Node(n_rows, "root", True)
        self.nodes = {
            "root": self.root
        }

    def add_node(self, node_name):
        """
        Add a new as successor of the root node.

        Parameters
        ----------
        node_name : str
            The name of the node.
        """
        if node_name != "":
            new_node = self.root.add_successor(node_name)
            self.nodes[node_name] = new_node

    def add_path(self, nodes):
        """
        Add new nodes from a path.

        Parameters
        ----------
        nodes : np.ndarray
            The list with the path, e.g., [a b c] = a -> b -> c
        """
        successor = nodes[0]
        leaf = self.root.add_successor(successor)
        self.nodes[successor] = leaf
        index = 0
        while index < len(nodes) - 1 and nodes[index] != "":
            successor = nodes[index + 1]
            if successor != "":
                leaf = leaf.add_successor(successor)
                self.nodes[successor] = leaf
            index = index + 1

    def is_acyclic(self):
        visited = set()
        to_visit = [self.root]
        while len(to_visit) > 0:
            next = to_visit.pop(0)
            if next in visited:
                return False
            visited.add(next)
            to_visit.extend(next.successors.values())
        return True

    def get_parent_nodes(self):
        parent_nodes = []
        for node in self.nodes.values():
            # Skip only leaf nodes
            successors = node.successors.values()
            if len(successors) > 0:
                parent_nodes.append(node)
        return parent_nodes
