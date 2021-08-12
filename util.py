import numpy as np


def distance_to_segment(p, a, b):
    # normalized tangent vectors
    d_ba = b - a
    mask = np.hypot(d_ba[:, 0], d_ba[:, 1]) != 0
    d = np.divide(d_ba, np.tile(np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1), (1, 2)))
    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c) * mask.astype(int) + np.linalg.norm(a) * (~mask).astype(int)


def make_velocity_vector(direction, speed):
    return direction * speed / np.linalg.norm(direction)


def min_distance_to_obstacles(from_pos, to_pos, v, ox, ov, os, robot_size):
    """
    Find mininum distance between the robot and each obstacles as the robot travels
    from "from_pos" to "to_pos" with the velocity "v".
    Each obstacle is also moving with the velocity "ov"
    If the value is negative, a collision has occurred.

    :param from_pos: starting position, 2D point
    :param to_pos: goal position, 2D point
    :param v: 2D vector of velocity
    :param ox: positions of obstacles (N obstacles, 2)
    :param ov: v vectors of obstacles (N obstacles, 2)
    :param os: obstacle size (N obstacles,)
    :param robot_size: int
    :return: (N obstacles, min distance)
    """
    n_obstacles = ox.shape[0]
    # Time for the robot to travel
    t = np.linalg.norm(to_pos - from_pos) / np.linalg.norm(v)
    # Change everything to the robot's coordinate system
    shifted_ox = ox - from_pos
    next_shifted_ox = shifted_ox + (ov - v) * t
    min_dist = distance_to_segment(np.zeros((n_obstacles, 2)), shifted_ox, next_shifted_ox)
    collision_dist = os + np.tile(np.array([robot_size]), (n_obstacles,))
    return min_dist - collision_dist
