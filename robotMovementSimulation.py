

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


#Calculate the Manhattan distance between two points a and b.
def heuristic(a, b):
    (x1, y1) = a  #a: Tuple representing the x and y coordinates of point a (e.g., (x1, y1))
    (x2, y2) = b  #Calculate the Manhattan distance between two points a and b.
    return abs(x1 - x2) + abs(y1 - y2) #Returns: - The Manhattan distance between points a and b.

# Environment Class represents the grid and handles state transitions.
class Environment:
    def __init__(self, grid, start, goal):
        self.grid = grid  # Represents the grid layout where 1 represents an obstacle and 0 is free space.
        self.initial = start  # Represents the starting position of the agent.
        self.goal = goal  # Represents the goal position the agent aims to reach.
        self.battery_level = 100  # Initializes the battery level to 100%.
        self.recharge_count = 0  # Initializes the recharge count to 0.

    # Returns the possible actions from a given state.
    def actions(self, state):
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        x, y = state

        # Eliminates impossible actions based on grid boundaries and obstacles.
        if x == 0 or self.grid[x - 1][y] == 1:
            possible_actions.remove('UP')
        if x == len(self.grid) - 1 or self.grid[x + 1][y] == 1:
            possible_actions.remove('DOWN')
        if y == 0 or self.grid[x][y - 1] == 1:
            possible_actions.remove('LEFT')
        if y == len(self.grid[0]) - 1 or self.grid[x][y + 1] == 1:
            possible_actions.remove('RIGHT')

        return possible_actions

# Returns the state resulting from taking a given action at a given state.
    def result(self, state, action):
        x, y = state

        if action == 'UP':
            new_state = (x - 1, y)
        elif action == 'DOWN':
            new_state = (x + 1, y)
        elif action == 'LEFT':
            new_state = (x, y - 1)
        elif action == 'RIGHT':
            new_state = (x, y + 1)

        # Updates the battery level.
        self.battery_level -= 10
        if self.battery_level <= 0:
            # Requires the robot to recharge before continuing.
            self.recharge_battery()

        return new_state

    # Recharges the battery level to 100%.
    def recharge_battery(self):
        self.battery_level = 100
        self.recharge_count += 1
    # Checks if the goal has been reached.
    def is_goal(self, state):
        return state == self.goal

    # Returns the current recharge count.
    def get_recharge_count(self):
        return self.recharge_count
