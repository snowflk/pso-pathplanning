from abc import abstractmethod
from typing import Sequence
import numpy as np

from util import min_distance_to_obstacles
from dstarlite import DStarLite
from psofinder import PSOFinder
from world import DynamicObject


class PathPlanner:
    def __init__(self, robot_size=10, map_size=(400, 400), **kwargs):
        self._robot_size = robot_size
        self._map_size = map_size
        self._t = 0

    def tick_callback(self):
        self._t += 1

    @abstractmethod
    def make_navigation_command(self, current_pos, goal, obstacle_info: Sequence[DynamicObject], **kwargs):
        pass


class PSOPlanner(PathPlanner):
    def __init__(self, n_waypoints, max_velocity=10, population=100, epochs=500, **kwargs):
        super(PSOPlanner, self).__init__(**kwargs)
        self._pso = PSOFinder(n_waypoints=n_waypoints,
                              size=self._map_size[0],
                              population=population,
                              epochs=epochs,
                              max_v=max_velocity)
        self._max_v = max_velocity
        self._n_waypoints = n_waypoints
        self._population = population
        self._epochs = epochs

        self._obs_memory = None
        self._current_plan_wps = None
        self._current_plan_vs = None
        self._current_plan_idx = -1
        self._gave_next_step = False
        self._obs_memory_t = 0

    def should_replan(self, current_pos, current_v, obstacles):
        # Replan if there is nothing left to do
        if self._current_plan_wps is None or self._current_plan_idx >= len(self._current_plan_wps):
            return True

        # Check collision, replan if we'll collide on the way to the next waypoint
        next_wp = self._current_plan_wps[self._current_plan_idx]
        ox = obstacles[:, :2]
        ov = obstacles[:, 2:4]
        osize = obstacles[:, -1]
        dist_to_robot = min_distance_to_obstacles(current_pos, next_wp, current_v, ox, ov, osize, self._robot_size)
        if np.any(dist_to_robot < 0):
            return True
        return False

    def _on_waypoint_reached(self):
        self._current_plan_idx += 1
        self._gave_next_step = False

    def _get_current_nav_command(self):
        self._gave_next_step = True
        return {
            'next_pos': self._current_plan_wps[self._current_plan_idx],
            'next_v': self._current_plan_vs[self._current_plan_idx],
            'plan': self._current_plan_wps[self._current_plan_idx:]
        }

    def make_navigation_command(self, current_pos, goal, obstacle_info: Sequence[DynamicObject], current_v=0, **kwargs):
        if len(obstacle_info) == 0:
            return {
                'next_pos': goal,
                'next_v': self._max_v
            }
        obstacles = np.vstack([
            np.concatenate([obs.x, obs.v, [obs.size]]) for obs in obstacle_info
        ])

        if self._current_plan_wps is not None and np.linalg.norm(
                self._current_plan_wps[self._current_plan_idx] - current_pos) < 5:
            self._on_waypoint_reached()

        if not self.should_replan(current_pos, current_v, obstacles):
            if not self._gave_next_step:
                return self._get_current_nav_command()
            return None

        self._obs_memory = obstacles.copy()
        self._obs_memory_t = self._t

        waypoints, velocities, _ = self._pso.calculate_path(obstacles=obstacles,
                                                            target=goal,
                                                            robot_pos=current_pos,
                                                            robot_size=self._robot_size)
        self._current_plan_wps = np.vstack([waypoints, goal])
        self._current_plan_vs = velocities
        self._current_plan_idx = 0
        return self._get_current_nav_command()


class DSLPlanner(PathPlanner):
    def __init__(self, cell_size=10, **kwargs):
        super(DSLPlanner, self).__init__(**kwargs)
        self._cell_size = cell_size
        self._maxx = self._map_size[0] // self._cell_size
        self._maxy = self._map_size[1] // self._cell_size
        self._dsl = DStarLite(map_size=(self._maxx, self._maxy))
        self._init = False
        self._goal = None

    def make_navigation_command(self, current_pos, goal, obstacle_info: Sequence[DynamicObject], **kwargs):
        # Convert map into grid
        # Rescale all coordinates
        obs = list()
        for i in range(self._maxx):
            for j in range(self._maxy):
                pos = np.array([i * self._cell_size, j * self._cell_size])
                for inf in obstacle_info:
                    if inf.did_collide_pos(pos, self._robot_size + 10):
                        obs.append(np.array([i, j]))

        current_pos = (current_pos / self._cell_size).astype(int)
        goal = (goal / self._cell_size).astype(int)
        if not self._init:
            self._goal = goal
            self._init = True
            self._dsl.initialize(current_pos, goal, obs)
        # Get path and convert grid coordinates back to normal ones
        next_pos = self._dsl.find_path(current_pos, goal, obstacles=obs)
        return {
            'next_pos': next_pos.astype(np.float64) * self._cell_size
        }
