import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from env.car_racing import CarRacing, STATE_H, STATE_W
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, trajectory_color_max_speed, trajectory_thickness, draw_for_last):
        super(CustomEnvWrapper, self).__init__(env)
        self.trajectory_color_max_speed = trajectory_color_max_speed
        self.trajectory_thickness = trajectory_thickness
        self.draw_for_last = draw_for_last

        self.frames = []
        self.car_states = []
        self.observation_space = Box(low=0, high=255, shape=(*self.observation_space.shape[:2], 1), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = rgb2gray(obs)

        self.frames = [obs]
        self.car_states = [info]

        return obs, info

    def step(self, action):
        modified_action = action
        observation, reward, truncated, done, info = self.env.step(modified_action)
        obs = rgb2gray(observation)
        self.draw_trajectory_on_frame_angle_based(obs, self.car_states[-self.draw_for_last:])

        self.frames.append(obs)
        self.car_states.append(info)

        return obs, reward, truncated, done, info

    def interpolate_color(self, x):
        x = max(min(x, self.trajectory_color_max_speed), 0)
        fraction = x / self.trajectory_color_max_speed
        r = g = b = int(255 * fraction)

        return r, g, b

    def draw_trajectory_on_frame_angle_based(self, frame, car_states):
        height, width = frame.shape[:2]
        half_width = int(width / 2)
        quarter_height = int(height / 4)

        new_positions = [(0., 0.)]

        world_center = car_states[-1]

        angle = -world_center['heading']
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        world_center_coords = np.dot(rotation_matrix, np.array([world_center['x'], world_center['y']]))
        for state in reversed(car_states[:-1]):
            coords_in_world_center = np.dot(rotation_matrix, np.array([state['x'], state['y']]))
            new_positions.append((coords_in_world_center[0] - world_center_coords[0],
                                  coords_in_world_center[1] - world_center_coords[1]))

        points_inside = np.array([
            (np.clip(half_width + (STATE_W / 100.) * item[0], -1, 401).astype(np.int32),
             np.clip(height - quarter_height - (STATE_H / 100.) * item[1], -1, 401).astype(np.int32))
            for item in new_positions])

        for index, item in enumerate(points_inside):
            if index == len(points_inside) - 1:
                break
            cv2.line(frame, item, points_inside[index + 1],
                     self.interpolate_color(car_states[len(points_inside) - index - 1]['speed']),
                     self.trajectory_thickness)
