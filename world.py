from typing import List, Set

import numpy as np
from log import make_logger


class DynamicObject:
    def __init__(self, spawn_pos=None, velocity=None, size=5, color=None):
        self._v = np.array([0, 0], dtype=np.float64)
        self._x = np.array([0, 0], dtype=np.float64)
        if spawn_pos is not None:
            self._x = np.array(spawn_pos, dtype=np.float64)
        if velocity is not None:
            self._v = np.array(velocity, dtype=np.float64)
        self._size = size
        self.color = color

    @property
    def x(self):
        return self._x

    @property
    def v(self):
        return self._v

    def set_velocity(self, v):
        self._v = v

    def set_direction_and_velocity(self, target, v):
        veloc_dir = target - self._x
        veloc_dir = veloc_dir / (np.linalg.norm(veloc_dir) + 1e-9)
        self._v = veloc_dir * v
        self._logger.info(f"Next way point set: {self._next_waypoint}, velocity: {v}")

    @property
    def size(self):
        return self._size

    def step(self):
        self._x += self._v

    def did_collide_pos(self, other_pos, other_size, custom_size=None):
        r = custom_size or self._size
        return np.linalg.norm(self._x - other_pos) < r + other_size

    def did_collide(self, other, custom_range=None):
        r = custom_range or self._size
        other_size = other.size
        pos = other.x
        return self.did_collide_pos(pos, other_size, r)


class Robot(DynamicObject):
    def __init__(self, path_planner, goal, sensor_range, max_v=2, name='Noname', **kwargs):
        super(Robot, self).__init__(**kwargs)
        self._planner = path_planner
        self._goal = np.array(goal, dtype=np.float64)
        self._sensor_range = sensor_range
        self._nearby = set()
        self._next_waypoint = None
        self._max_v = max_v
        self._name = name
        self._logger = make_logger(name)
        self._current_plan = None
        self._dead = False
        self._goal_reached = False

    def notify_sensor(self, nearby_objs):
        self._nearby.clear()
        self._nearby = self._nearby.union(nearby_objs)

    def set_direction_and_velocity(self, target, v):
        if v > self._max_v:
            v = self._max_v
            self._logger.warn(f"Attempt to set velocity too high. Allowed: {self._max_v}, Actual: {v}")
        super(Robot, self).set_direction_and_velocity(target, v)

    @property
    def name(self):
        return self._name

    def get_plan(self):
        return self._current_plan

    def _should_replan(self):
        return self._planner.should_replan(self._nearby)

    def step(self):
        if self.is_dead() or self.goal_reached():
            return
        self._planner.tick_callback()
        dist_to_goal = np.linalg.norm(self.goal - self.x)
        if dist_to_goal < 2:
            self._goal_reached = True
            self._logger.info("Reached goal")
            return
        # Check whether way point is reached
        if self._next_waypoint is not None:
            dist_remaining = np.linalg.norm(self._next_waypoint - self._x)
            self._logger.info(f"{dist_remaining:.2f}m more to reach waypoint {self._next_waypoint}")
            if dist_remaining < 2:
                self._next_waypoint = None
                self._logger.info(f"Target way point reached")

        # If the planner thinks we need replanning, or target waypoint already reached
        # Plan the path
        # Stick to the old plan if nav command is null
        nav_command = self._planner.make_navigation_command(self.x, self.goal, self._nearby,
                                                            current_v=self.v)
        if nav_command is not None:
            self._next_waypoint = nav_command['next_pos']
            self._current_plan = nav_command.get('plan', None)
            next_veloc = nav_command.get('next_v', self._max_v)
            self.set_direction_and_velocity(self._next_waypoint, next_veloc)
        else:
            if self._next_waypoint is None:
                self._logger.info(f"Could not find any path and doesn't know what to do")
                return
            self._logger.info(f"Nothing new, stick to the old command")
        # Prevent overshooting
        dist_remaining = np.linalg.norm(self._next_waypoint - self._x)
        if dist_remaining < np.linalg.norm(self.v):
            self._logger.info("Slow down to prevent overshooting")
            self.set_direction_and_velocity(self._next_waypoint, dist_remaining)
        super().step()

    def dead(self):
        self.set_velocity(np.array([0, 0]))
        self._dead = True

    def is_dead(self):
        return self._dead

    def goal_reached(self):
        return self._goal_reached

    @property
    def goal(self):
        return self._goal

    def get_sensor_range(self):
        return self._sensor_range

    def can_detect_using_sensor(self, other):
        return self.did_collide(other, custom_range=self._sensor_range)


class Obstacle(DynamicObject):
    def __init__(self, **kwargs):
        super(Obstacle, self).__init__(**kwargs)


class World:
    def __init__(self, robots, obstacles, map_size=100):
        self._robots: Set[Robot] = set(robots)
        self._obstacles: Set[Obstacle] = set(obstacles)
        self._all: Set[DynamicObject] = self._obstacles.union(self._robots)
        self._logger = make_logger('World')
        self._map_size = map_size

    def step(self):
        # Notify the robots about their surrounding (fake sensor)
        for r in self._robots:
            sensed = set()
            for other in self._all:
                if r != other and r.can_detect_using_sensor(other):
                    sensed.add(other)
                if r != other and r.did_collide(other):
                    self._logger.info(f"{r.name} crashed")
                    r.dead()
            r.notify_sensor(sensed)

        for obj in self._all:
            obj.step()
            # Rebound
            if isinstance(obj, Obstacle):
                new_v = obj.v.copy()
                for i in range(2):
                    if obj.x[i] < obj.size or obj.x[i] > self._map_size - obj.size:
                        new_v[i] *= -1
                obj.set_velocity(new_v)

    def get_objects(self):
        return self._all
