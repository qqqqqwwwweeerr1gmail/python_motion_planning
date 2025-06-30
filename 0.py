from math import sqrt
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

"""
@file: node.py
@breif: 2-dimension node data stucture
@author: Yang Haodong, Wu Maojia
@update: 2024.3.15
"""


class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost

    Examples:
        >>> from env import Node
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 5), 2, 8)
        >>> node3 = Node((2, 0), (1, 6), 3, 1)
        ...
        >>> node1 + node2
        >>> Node((2, 0), (2, 3), 3, 2)
        ...
        >>> node1 == node2
        >>> True
        ...
        >>> node1 != node3
        >>> True
    """

    def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h

    def __add__(self, node):
        assert isinstance(node, Node)
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h)

    def __eq__(self, node) -> bool:
        if not isinstance(node, Node):
            return False
        return self.current == node.current

    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        assert isinstance(node, Node)
        return self.g + self.h < node.g + node.h or \
            (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {})".format(self.current, self.parent, self.g, self.h)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def x(self) -> float:
        return self.current[0]

    @property
    def y(self) -> float:
        return self.current[1]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None


class Env(ABC):
    """
    Class for building 2-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        # >>> from python_motion_planning.utils import Env
        # >>> env = Env(30, 40)
    """

    def __init__(self, x_range: int, y_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range
        self.y_range = y_range
        # self.x_range = 150
        # self.y_range = 29

        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass


class Grid(Env):
    """
    Class for discrete 2-d grid map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def init(self) -> None:
        """Initialize the grid environment"""
        # Initialize obstacles (empty by default)
        self.obstacles = set()
        # Create KD-tree for obstacle checking
        if self.obstacles:
            self.obstacles_tree = cKDTree(np.array(list(self.obstacles)))
        else:
            self.obstacles_tree = None


    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        # allowed motions
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1), None, sqrt(2), None),
                        Node((0, 1), None, 1, None), Node((1, 1), None, sqrt(2), None),
                        Node((1, 0), None, 1, None), Node((1, -1), None, sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        # self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
        #                 Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
        #                 Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None)]

        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.init()



dpis = (499,500)

# grid_env = Grid(102, 102)
grid_env = Grid(dpis[0]+2, dpis[1]+2)
print(grid_env)


















