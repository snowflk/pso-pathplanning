import os
import glob
import json
import numpy as np
from pygame import gfxdraw
from typing import List

import pygame
from world import Robot, Obstacle, DynamicObject, World
from path_planners import PSOPlanner, DSLPlanner

from log import make_logger

logger = make_logger()

ROBOT_COLOR = (0, 255, 0)
DEAD_ROBOT_COLOR = (100, 100, 100)
OBS_COLOR = (255, 0, 0)
BORDER_COLOR = (50, 50, 50)
SENSOR_RANGE_COLOR = '#dddddd'
WIN_WIDTH, WIN_HEIGHT = 600, 600
LOG_DIR = './logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)


def make_planner(planner_type, robot_cfg, cfg):
    map_size = (cfg['width'], cfg['height'])

    if planner_type == 'PSO' or planner_type == 'PPSO':
        return PSOPlanner(n_waypoints=3,
                          max_velocity=cfg['max_v'],
                          population=30,
                          map_size=map_size,
                          epochs=3000,
                          robot_size=robot_cfg['size'],
                          use_polar=planner_type == 'PPSO',
                          max_angle=60, log_dir=LOG_DIR)
    cell_size = cfg['max_v'] / np.sqrt(2)
    return DSLPlanner(robot_size=robot_cfg['size'],
                      cell_size=cell_size,
                      map_size=map_size)


def make_world(config, planner_type):
    robots = list()
    obstacles = list()
    for idx, robot_cfg in enumerate(config['robots']):
        planner = make_planner(planner_type, robot_cfg, config)
        pos = robot_cfg['start_x'], robot_cfg['start_y']
        goal = robot_cfg['goal_x'], robot_cfg['goal_y']
        size = robot_cfg['size']
        name = robot_cfg.get('name', f'Robot {idx+1}')
        sensor_range = robot_cfg['sensor_range']
        color = robot_cfg['color'] or ROBOT_COLOR
        if isinstance(color, str):
            color = tuple([int(x) for x in color.split(',')])
        robots.append(Robot(path_planner=planner,
                            spawn_pos=np.array(pos),
                            size=size,
                            max_v=config['max_v'],
                            sensor_range=sensor_range,
                            goal=goal,
                            name=name,
                            color=color))
    for obs_cfg in config['obstacles']:
        pos = obs_cfg['start_x'], obs_cfg['start_y']
        velocity = obs_cfg.get('v_x', 0), obs_cfg.get('v_y', 0)
        size = obs_cfg['size']
        color = obs_cfg.get('color', OBS_COLOR)
        if isinstance(color, str):
            color = tuple([int(x) for x in color.split(',')])
        if np.sum(np.abs(velocity)) == 0:
            color = BORDER_COLOR
        obstacles.append(Obstacle(spawn_pos=pos,
                                  size=size,
                                  velocity=velocity,
                                  color=color))

    world = World(robots=robots, obstacles=obstacles)
    return world


class SimObject:
    def __init__(self, model: DynamicObject, color: str, scale=1):
        self._model = model
        self._color = color
        self._scale = scale
        self._trail = []

    def draw(self, canvas):
        pos = np.round(self._model.x * self._scale).astype(int)
        size = np.round(self._model.size * self._scale).astype(int)
        veloc = self._model.v * self._scale

        # Draw object body
        gfxdraw.filled_circle(canvas, pos[0], pos[1], size, BORDER_COLOR)
        gfxdraw.filled_circle(canvas, pos[0], pos[1], size - 2, self._color)
        gfxdraw.aacircle(canvas, pos[0], pos[1], size - 2, self._color)
        gfxdraw.aacircle(canvas, pos[0], pos[1], size, BORDER_COLOR)
        if isinstance(self._model, Robot):
            if self._model.is_dead():
                gfxdraw.filled_circle(canvas, pos[0], pos[1], size - 2, DEAD_ROBOT_COLOR)
            pygame.draw.circle(canvas,
                               color=SENSOR_RANGE_COLOR,
                               center=pos,
                               radius=self._model.get_sensor_range() * self._scale,
                               width=1)

        # Draw direction pointer
        if veloc is None or np.linalg.norm(veloc) == 0:
            return
        veloc_dir = veloc * size / np.linalg.norm(veloc)
        pygame.draw.aaline(canvas,
                           start_pos=pos, end_pos=pos + veloc_dir,
                           color=BORDER_COLOR)

    def draw_plan(self, canvas):
        for t in self._trail:
            size = 2 if isinstance(self._model, Robot) else 1
            gfxdraw.filled_circle(canvas, t[0], t[1], size + 1, BORDER_COLOR)
            gfxdraw.filled_circle(canvas, t[0], t[1], size, self._color)
        pos = (self._model.x * self._scale).astype(int)
        if isinstance(self._model, Robot):
            plan = self._model.get_plan()
            if plan is not None:
                pygame.draw.aalines(canvas, points=np.vstack([pos, plan * self._scale]), closed=False,
                                    color='#333333')

    def get_model(self):
        return self._model

    def tick(self):
        self._trail.append(np.round(self._model.x * self._scale).astype(int))


class Simulation:
    TITLE = 'Path Planning Simulation'
    BG_COLOR = '#ffffff'

    DEFAULT_HEIGHT = 400
    DEFAULT_WIDTH = 400
    FPS = 24
    TICK = 1000 // FPS

    def __init__(self, config_file, planner_type, image_suffix=None, image_dir=None):
        self._config_file = config_file
        self._planner_type = planner_type
        self._read_config()
        self._objs: List[SimObject] = list()

        self._field_width = self._config['width'] or self.DEFAULT_WIDTH
        self._field_height = self._config['height'] or self.DEFAULT_HEIGHT
        self._screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self._clock = pygame.time.Clock()
        pygame.display.set_caption(self.TITLE)
        pygame.font.init()
        self._font = pygame.font.Font(pygame.font.get_default_font(), 30)
        self._image_counter = 0
        self._image_suffix = image_suffix
        self._image_dir = image_dir
        # Construct world
        self._size_scale = WIN_WIDTH / self._field_width
        self._world_model = make_world(self._config, self._planner_type)
        self._obj_models = self._world_model.get_objects()
        for obj in self._obj_models:
            sim_obj = SimObject(model=obj, color=obj.color, scale=self._size_scale)
            self._objs.append(sim_obj)

    def _read_config(self):
        with open(self._config_file) as f:
            self._config = json.load(f)
        logger.info(f"Loaded config from {os.path.abspath(self._config_file)}")

    def _tick(self):
        self._world_model.step()
        for obj in self._objs:
            obj.tick()

    def _draw(self):
        self._screen.fill(self.BG_COLOR)
        # Draw robot plan first as the first layer
        for obj in self._objs:
            obj.draw_plan(self._screen)
        # Draw objects on top of the plan
        for obj in self._objs:
            obj.draw(self._screen)
            if isinstance(obj.get_model(), Robot):
                goal_pos = obj.get_model().goal * self._size_scale
                t = self._font.render('x', True, obj._color)
                self._screen.blit(t, tuple(goal_pos))

        pygame.display.update()

    def _screenshot(self):
        self._image_counter += 1
        filename = os.path.join(self._image_dir, f"{self._image_suffix}_{self._image_counter}.jpeg")
        pygame.image.save(self._screen, filename)

    def run(self):
        while True:
            pygame.event.get()
            pygame.time.delay(self.TICK)
            self._clock.tick(self.TICK)
            self._tick()
            self._draw()
            if self._image_suffix is not None:
                self._screenshot()


if __name__ == '__main__':
    config_file = os.getenv('CFG', './config/obs_rich_multi.json')
    planner_type = os.getenv('PLANNER', 'PPSO')
    image_out_dir = os.getenv('IMG_OUT', 'experiments')
    env_name = os.path.splitext(os.path.basename(config_file))[0]
    image_suffix = f"{planner_type}_{env_name}"
    image_out_dir = os.path.join(image_out_dir, image_suffix)
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(image_out_dir, f"{image_suffix}_*.jpeg")):
        os.remove(f)

    Simulation(config_file=config_file,
               planner_type=planner_type,
               image_suffix=image_suffix,
               image_dir=image_out_dir).run()
