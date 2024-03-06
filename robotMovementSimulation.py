

#Enhanced Dynamic Robot Movement Simulation**


import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

# Node Class represents a state in the search tree.
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state  # Represents the current position of the agent in the grid.
        self.parent = parent  # Represents the node in the search tree that generated this node.
        self.action = action  # Indicates the action taken to reach this state.
        self.path_cost = path_cost  # Represents the cost from the start node to this node.

    # Comparison operator for the priority queue.
    def __lt__(self, other):
        return self.path_cost < other.path_cost
