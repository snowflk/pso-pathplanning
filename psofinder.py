"""
Metaheuristic Minimization Using Particle Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi
"""

import numpy as np
from util import make_velocity_vector, min_distance_to_obstacles
from pso import PSO


class PSOFinder:
    def __init__(self, n_waypoints=7, size=400, population=100, epochs=500, max_v=5):
        self.n_waypoints = n_waypoints
        self.max_v = max_v
        self.lb = np.concatenate([np.zeros(n_waypoints * 2), np.zeros(n_waypoints + 1)])
        self.ub = np.concatenate([np.ones(n_waypoints * 2) * size, np.ones(n_waypoints + 1) * max_v])
        self.opt = None
        self.robot = None
        self.population = population
        self.epochs = epochs
        self.last_pos = np.ones(2) * -1

    def set_robot(self, robot):
        self.robot = robot

    def compile_fitnessfunc(self, X, args):
        """
        W is the number of waypoints
        N is the number of particles
        There are W+1 segments, 2W coordinate values

        X is the solution (N particles, 2*W, W + 1)

        Obstacles Data (N obstacles, 5) (x vector + v vector + size)
        Robot Pos 2D point (x,y)
        Target 2D point (x,y)
        Robot size: int
        :param X:
        :param args:
        :return:
        """
        obstacles, target, robot_pos, robot_size = args
        n_obstacles = obstacles.shape[0]
        waypoints = X[:, :-self.n_waypoints - 1]
        V = X[:, -self.n_waypoints - 1:]
        n_points = waypoints.shape[1] // 2
        n_particles = waypoints.shape[0]

        # Make N copies, add a new axis for N particles, then combine them into a single batch
        obstacles = np.expand_dims(obstacles, axis=0)
        obstacles_x = np.tile(obstacles[:, :, :2], (n_particles, 1, 1)).reshape(-1, 2)
        obstacles_v = np.tile(obstacles[:, :, 2:4], (n_particles, 1, 1)).reshape(-1, 2)
        obstacles_size = np.tile(obstacles[:, :, -1].reshape(1, n_obstacles), (n_particles, 1)).reshape(-1)

        # Add current position and goal, construct the segments
        points_arr = [np.tile(robot_pos.reshape(1, 1, 2), (n_particles, 1, 1)),
                      waypoints.reshape(n_particles, n_points, 2),
                      np.tile(np.array(target).reshape(1, 1, 2), (n_particles, 1, 1))]
        points = np.concatenate(points_arr, axis=1)
        diff = np.diff(points, axis=1)
        s = np.sqrt(np.sum(np.square(diff), axis=-1))  # length of each segment (particles, segments)
        d_cost = np.sum(s, axis=1)
        t_seg = s / V  # time for each segment
        t_cost = np.sum(t_seg, axis=1)
        c_cost = np.zeros((n_particles, self.n_waypoints + 1))

        # cumulated values
        obs_dpos = np.zeros((n_particles * n_obstacles, 2))

        direction = diff / np.tile(np.linalg.norm(diff, axis=2).reshape(n_particles, self.n_waypoints + 1, 1),
                                   (1, 1, 2))
        angles = np.zeros((n_particles, self.n_waypoints))

        for i in range(self.n_waypoints + 1):
            p1 = np.tile(points[:, i, :].reshape(-1, 1, 2), (1, n_obstacles, 1)).reshape(-1, 2)
            p2 = np.tile(points[:, i + 1, :].reshape(-1, 1, 2), (1, n_obstacles, 1)).reshape(-1, 2)
            if i > 0:
                angles[:, i - 1] = np.degrees(np.arccos(np.einsum('ij,ij->i', direction[:, i - 1], direction[:, i])))

            t = np.tile(t_seg[:, i].reshape(-1, 1, 2), (1, n_obstacles, 2)).reshape(-1, 2)
            robot_v_ori = p2 - p1
            robot_v_ori2 = robot_v_ori / np.tile(np.linalg.norm(robot_v_ori, axis=1).reshape(-1, 1), (1, 2))
            robot_v = robot_v_ori2 * np.tile(V[:, i].reshape(n_particles, 1, 1), (1, n_obstacles, 2)).reshape(-1, 2)

            obs_pos = obstacles_x + obs_dpos
            obs_v = obstacles_v
            min_dist = min_distance_to_obstacles(p1, p2, robot_v, obs_pos, obs_v, obstacles_size, robot_size)
            mask = min_dist < 0

            collision_penalty = -min_dist * mask.astype(int) * 100000 + np.abs(1. / min_dist) * (~mask).astype(int)
            c_cost[:, i] = collision_penalty.reshape(n_particles, -1).sum(axis=1)
            obs_dpos += t * obstacles_v

        # smoothness_mask = angles > 60
        # smoothness_cost = np.sum(angles ** 1.1, axis=1)
        # print("C_COST", c_cost.sum(axis=(1, 2)).copy())
        cost = t_cost + c_cost.sum(axis=1) + d_cost  # + smoothness_cost
        return cost

    def calculate_path(self, obstacles, target, robot_pos, robot_size):
        fitness_func = self.compile_fitnessfunc
        args = [obstacles, target, robot_pos, robot_size]
        solution, info = PSO(fitness_func, LB=self.lb, UB=self.ub, nPop=self.population,
                             epochs=self.epochs, args=args)
        waypoints = solution[:-self.n_waypoints - 1]
        v = solution[-self.n_waypoints - 1:]
        return waypoints.reshape(-1, 2), v, info[0]
