"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pd)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
Implemented maintaining similarity with the pseudocode for understanding.
Code can be significantly optimized by using a priority queue for U, etc.
Avoiding additional imports based on repository philosophy.
"""
import math
import numpy as np
from typing import Tuple

pause_time = 0.001
p_create_random_obstacle = 0


class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost

    def __repr__(self):
        return f"({self.x}, {self.y})"


def add_coordinates(node1: Node, node2: Node):
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node


def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y


class DStarLite:
    # Please adjust the heuristic function (h) if you change the list of
    # possible motions
    motions = [
        Node(1, 0, 1),
        Node(0, 1, 1),
        Node(-1, 0, 1),
        Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)),
        Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)),
        Node(-1, -1, math.sqrt(2))
    ]

    def __init__(self, map_size):
        # Ensure that within the algorithm implementation all node coordinates
        # are indices in the grid and extend
        # from 0 to abs(<axis>_max - <axis>_min)
        self.x_min_world = 0
        self.y_min_world = 0
        self.x_max = map_size[0]
        self.y_max = map_size[1]

        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = list()
        self.km = 0.0
        self.kold = 0.0
        self.rhs = list()
        self.g = list()
        self.detected_obstacles = list()

    def create_grid(self, val: float):
        grid = list()
        for _ in range(0, self.x_max):
            grid_row = list()
            for _ in range(0, self.y_max):
                grid_row.append(val)
            grid.append(grid_row)
        return grid

    def is_obstacle(self, node: Node):
        return any([compare_coordinates(node, obstacle)
                    for obstacle in self.detected_obstacles])

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            # Attempting to move from or to an obstacle
            return math.inf
        new_node = Node(node1.x - node2.x, node1.y - node2.y)
        detected_motion = list(filter(lambda motion:
                                      compare_coordinates(motion, new_node),
                                      self.motions))
        return detected_motion[0].cost

    def h(self, s: Node):
        # Cannot use the 2nd euclidean norm as this might sometimes generate
        # heuristics that overestimate the cost, making them inadmissible,
        # due to rounding errors etc (when combined with calculate_key)
        # To be admissible heuristic should
        # never overestimate the cost of a move
        # hence not using the line below
        # return math.hypot(self.start.x - s.x, self.start.y - s.y)

        # Below is the same as 1; modify if you modify the cost of each move in
        # motion
        # return max(abs(self.start.x - s.x), abs(self.start.y - s.y))
        return 1

    def calculate_key(self, s: Node):
        return (min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s)
                + self.km, min(self.g[s.x][s.y], self.rhs[s.x][s.y]))

    def is_valid(self, node: Node):
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def get_neighbours(self, u: Node):
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def pred(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def succ(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def init_first_run(self, start: Node, goal: Node):
        self.start.x = start.x - self.x_min_world
        self.start.y = start.y - self.y_min_world
        self.goal.x = goal.x - self.x_min_world
        self.goal.y = goal.y - self.y_min_world
        self.U = list()  # Would normally be a priority queue
        self.km = 0.0
        self.rhs = self.create_grid(math.inf)
        self.g = self.create_grid(math.inf)
        self.rhs[self.goal.x][self.goal.y] = 0
        self.U.append((self.goal, self.calculate_key(self.goal)))
        self.detected_obstacles = list()

    def update_vertex(self, u: Node):
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min([self.c(u, sprime) +
                                      self.g[sprime.x][sprime.y]
                                      for sprime in self.succ(u)])
        if any([compare_coordinates(u, node) for node, key in self.U]):
            self.U = [(node, key) for node, key in self.U
                      if not compare_coordinates(node, u)]
            self.U.sort(key=lambda x: x[1])
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key_pair1: Tuple[float, float],
                     key_pair2: Tuple[float, float]):
        return key_pair1[0] < key_pair2[0] or \
               (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        self.U.sort(key=lambda x: x[1])
        while (len(self.U) > 0 and
               self.compare_keys(self.U[0][1],
                                 self.calculate_key(self.start))) or \
                self.rhs[self.start.x][self.start.y] != \
                self.g[self.start.x][self.start.y]:
            self.kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            if self.compare_keys(self.kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x][u.y] > self.rhs[u.x][u.y]:
                self.g[u.x][u.y] = self.rhs[u.x][u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x][u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)
            self.U.sort(key=lambda x: x[1])

    def detect_changes(self, obs):
        self.detected_obstacles.clear()
        self.detected_obstacles.extend(obs)
        return obs

    def compare_paths(self, path1: list, path2: list):
        if len(path1) != len(path2):
            return False
        for node1, node2 in zip(path1, path2):
            if not compare_coordinates(node1, node2):
                return False
        return True

    def initialize(self, start, goal, obstacles):
        self.init_first_run(Node(x=start[0], y=start[1]), goal=Node(x=goal[0], y=goal[1]))
        self.last = self.start
        self.last_obs = None
        self.compute_shortest_path()

    def convert_obstacles(self, obstacles):
        return [Node(x=o[0], y=o[1]) for o in obstacles]

    def find_path(self, start_pos, goal_pos, obstacles):
        obstacles = self.convert_obstacles(obstacles)
        if self.g[self.start.x][self.start.y] == math.inf:
            # No path possible
            raise RuntimeError('No path possible')
        self.start = min(self.succ(self.start),
                         key=lambda sprime:
                         self.c(self.start, sprime) +
                         self.g[sprime.x][sprime.y])

        if self.last_obs is not None:
            self.last_obs.extend(obstacles)
            changed_vertices = self.detect_changes(self.last_obs)
        else:
            changed_vertices = self.detect_changes(obstacles)

        # New obstacle detected
        self.km += self.h(self.last)
        self.last = self.start
        for u in changed_vertices:
            if compare_coordinates(u, self.start):
                continue
            self.rhs[u.x][u.y] = math.inf
            self.g[u.x][u.y] = math.inf
            self.update_vertex(u)
        self.compute_shortest_path()

        direction = self.start
        self.last_obs = obstacles
        return np.array([direction.x, direction.y])
