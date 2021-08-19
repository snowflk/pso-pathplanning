"""
Local Path Planning using Particle Swarm Optimization

Copyright (c) 2021 Thien Phuc Tran
"""

import numpy as np
from util import min_distance_to_obstacles, make_velocity_vector
from pso import PSO


class PSOFinder:
    def __init__(self, n_waypoints=7, size=400, population=100, epochs=500, max_v=5):
        self.n_waypoints = n_waypoints
        self.max_v = max_v
        self.lb = np.concatenate([np.zeros(n_waypoints * 2), np.ones(n_waypoints + 1) * 0.1])
        self.ub = np.concatenate([np.ones(n_waypoints * 2) * size, np.ones(n_waypoints + 1) * max_v])
        self.opt = None
        self.population = population
        self.epochs = epochs
        self.last_pos = np.ones(2) * -1

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

        # Make N copies, add a new axis for N particles, then combine them into a single batch n_particles*n_obstacles
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
        # Compute length for each segment and compute the distance cost and time cost
        s = np.sqrt(np.sum(np.square(diff), axis=-1))  # length of each segment (particles, segments)
        t_seg = s / V  # time for each segment
        t_cost = np.sum(t_seg, axis=1)

        d_cost = np.sum(s, axis=1)

        # Compute collision cost (safety cost)
        c_cost = np.zeros((n_particles, self.n_waypoints + 1))

        # cumulated position shift of the obstacles after each segment
        obs_dpos = np.zeros((n_particles * n_obstacles, 2))

        # direction = diff / np.tile(np.linalg.norm(diff, axis=2).reshape(n_particles, self.n_waypoints + 1, 1),
        #                           (1, 1, 2))

        # angles = np.zeros((n_particles, self.n_waypoints))

        for i in range(self.n_waypoints + 1):
            p1 = np.tile(points[:, i, :].reshape(-1, 1, 2), (1, n_obstacles, 1)).reshape(-1, 2)
            p2 = np.tile(points[:, i + 1, :].reshape(-1, 1, 2), (1, n_obstacles, 1)).reshape(-1, 2)
            # if i > 0:
            #    angles[:, i - 1] = np.degrees(np.arccos(np.einsum('ij,ij->i', direction[:, i - 1], direction[:, i])))

            t = np.tile(t_seg[:, i].reshape(-1, 1, 2), (1, n_obstacles, 2)).reshape(-1, 2)
            robot_v_dir = p2 - p1
            robot_v_dir_unit = robot_v_dir / (
                    np.tile(np.linalg.norm(robot_v_dir, axis=1).reshape(-1, 1), (1, 2)) + 1e-4)
            robot_v = robot_v_dir_unit * np.tile(V[:, i].reshape(n_particles, 1, 1), (1, n_obstacles, 2)).reshape(-1, 2)

            obs_pos = obstacles_x + obs_dpos

            min_dist = min_distance_to_obstacles(p1, p2, robot_v, obs_pos, obstacles_v, obstacles_size + 2, robot_size)
            mask = min_dist < 0

            collision_penalty = -min_dist * mask.astype(int) * 100000 + np.abs(
                10 * 1. / (min_dist + 1e-4)) * (~mask).astype(int)
            c_cost[:, i] = collision_penalty.reshape(n_particles, -1).sum(axis=1)
            obs_dpos += t * obstacles_v

        # smoothness_mask = angles > 60
        # smoothness_cost = np.sum(angles * smoothness_mask.astype(int), axis=1) * 0.1
        # print("C_COST", c_cost.sum(axis=(1, 2)).copy())
        cost = t_cost * 0.5 + c_cost.sum(axis=1) + d_cost * 2  # + smoothness_cost
        return cost

    def calculate_path(self, obstacles, target, robot_pos, robot_size):
        fitness_func = self.compile_fitnessfunc
        args = [obstacles, target, robot_pos, robot_size]
        init_solution = [robot_pos for _ in range(self.n_waypoints)]
        init_solution.append(np.ones(self.n_waypoints + 1) * 0.1)
        init_solution[-1][-1] = self.max_v
        init_solution = np.concatenate(init_solution)
        solution, info = PSO(fitness_func, LB=self.lb, UB=self.ub, nPop=self.population,
                             epochs=self.epochs, args=args,
                             IntVar=[x + 1 for x in range(self.n_waypoints * 2)],
                             Xinit=init_solution)
        waypoints = solution[:-self.n_waypoints - 1]
        v = solution[-self.n_waypoints - 1:]

        return waypoints.reshape(-1, 2), v, info[0]


class PolarPSOFinder(PSOFinder):
    def __init__(self, n_waypoints=7, size=400, population=100, epochs=500, max_v=5, max_angle=60):
        super(PolarPSOFinder, self).__init__(n_waypoints, size, population, epochs, max_v)
        max_angle = np.deg2rad(max_angle)
        self.lb = np.concatenate(
            [-np.ones(n_waypoints) * max_angle, np.zeros(n_waypoints), np.ones(n_waypoints + 1) * 0.1])
        self.ub = np.concatenate(
            [np.ones(n_waypoints) * max_angle, np.ones(n_waypoints) * size, np.ones(n_waypoints + 1) * max_v])

    def _to_cartesian(self, X, robot_pos, target):
        Xz = np.copy(X)
        population = X.shape[0]
        robot_dir = np.tile((target - robot_pos).reshape(1, 2), (population, 1))
        acc_robot_pos = np.tile(robot_pos.reshape(1, 2), (population, 1))
        for i in range(self.n_waypoints):
            robot_angle = np.arctan2(robot_dir[:, 1], robot_dir[:, 0])
            # print("ANGLES", robot_angle[0] * 180 / np.pi)
            phi, r = X[:, i], X[:, self.n_waypoints + i]
            delta_x = np.swapaxes(np.array([np.cos(phi) * r, np.sin(phi) * r]), 0, 1)
            # print("X", delta_x[0])
            rotation_mat = np.transpose(np.array(
                [[np.cos(robot_angle), -np.sin(robot_angle)], [np.sin(robot_angle), np.cos(robot_angle)]]), (2, 0, 1))

            rotated_delta_x = np.zeros((population, 2))
            for j in range(population):
                rotated_delta_x[j, :] = rotation_mat[j] @ delta_x[j]
            # print(
            #    f"{i} | Angle: {robot_angle[0] * 180 / np.pi} | X {delta_x[0]} -> XR {rotated_delta_x[0]} | Phi: {phi[0]*180/np.pi}")
            Xz[:, 2 * i] = acc_robot_pos[:, 0] + rotated_delta_x[:, 0]
            Xz[:, 2 * i + 1] = acc_robot_pos[:, 1] + rotated_delta_x[:, 1]
            acc_robot_pos += rotated_delta_x
            robot_dir = acc_robot_pos
        # print("XZ", Xz[0])
        return Xz.copy()

    def compile_fitnessfunc(self, X, args):
        obstacles, target, robot_pos, robot_size = args
        X = self._to_cartesian(X, robot_pos, target)
        return super().compile_fitnessfunc(X, args)

    def calculate_path(self, obstacles, target, robot_pos, robot_size):
        fitness_func = self.compile_fitnessfunc
        args = [obstacles, target, robot_pos, robot_size]
        init_solution = [robot_pos for _ in range(self.n_waypoints)]
        init_solution.append(np.ones(self.n_waypoints + 1) * 0.1)
        init_solution[-1][-1] = self.max_v
        init_solution = np.concatenate(init_solution)
        solution, info = PSO(fitness_func, LB=self.lb, UB=self.ub, nPop=self.population,
                             epochs=self.epochs, args=args,
                             IntVar=[x + 1 for x in range(self.n_waypoints * 2)],
                             Xinit=init_solution)
        print("SOL", solution.shape)
        print("SOL2", np.expand_dims(solution, axis=0).shape)
        solution = self._to_cartesian(np.expand_dims(solution, axis=0), robot_pos, target)[0]
        waypoints = solution[:-self.n_waypoints - 1]
        v = solution[-self.n_waypoints - 1:]
        return waypoints.reshape(-1, 2), v, info[0]


if __name__ == '__main__':
    print(np.arctan2(-10, 10) * 180 / np.pi)
