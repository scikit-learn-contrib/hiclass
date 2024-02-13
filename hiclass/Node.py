import numpy as np

class Node:
    """Manages data for an individual node in the hierarchy."""

    def __init__(self, n_rows, name, default_mask):
        """
        Initialize an individual node.

        Parameters
        ----------
        n_rows : int
            The number of rows in x and y.
        name : str
            The name of the node.
        default_mask : Bool
            The default value of the mask, i.e., True or False.
        """
        self.n_rows = n_rows
        self.mask = np.full(n_rows, default_mask)
        self.successors = dict()
        self.name = name
        self.classifier = None

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
            if not successor_name in self.successors:
                new_successor = Node(self.n_rows, successor_name, False)
                self.successors[successor_name] = new_successor
                return new_successor
            else:
                return self.successors[successor_name]
