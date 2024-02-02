import numpy as np

class Node:
    """Manages data for an individual node in the hierarchy."""

    def __init__(self, n_rows, name):
        """
        Initialize an individual node.

        Parameters
        ----------
        n_rows : int
            The number of rows in x and y.
        """
        self.n_rows = n_rows
        self.mask = np.full(n_rows, True)
        self.children = dict()
        self.name = name

    def add_successor(self, successor_name):
        """
        Add a new successor.

        Parameters
        ----------
        node_name : str
            The name of the new successor.

        Returns
        -------
        successor : Node
            The new successor created.
        """
        if successor_name != "":
            if not successor_name in self.children:
                new_successor = Node(self.n_rows, successor_name)
                self.children[successor_name] = new_successor
                return new_successor
            else:
                return self.children[successor_name]
