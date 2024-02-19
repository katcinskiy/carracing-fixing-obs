import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import ffmpeg
import cv2

from env.car_racing import CarRacing

from tqdm.auto import tqdm

from PIL import Image

from gymnasium.spaces import Box

import numpy as np


def create_video_from_frames(frames, output_video_path, fps=24):
    # Check if the frame list is empty
    if not frames:
        print("The frame list is empty. No video will be created.")
        return

    # Determine the size of the first frame
    height, width = frames[0].shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert frames to the correct color format if necessary
        if frame.ndim == 2 or frame.shape[2] == 1:
            # If the frame is grayscale, convert it to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            # If the frame has an alpha channel, convert it to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Ensure the frame size matches the video size
        if (frame.shape[0] != height) or (frame.shape[1] != width):
            print("Frame size does not match video size. Resizing frame.")
            frame = cv2.resize(frame, (width, height))

        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]


frames = []


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomEnvWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(*self.observation_space.shape[:2], 1), dtype=np.uint8)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return rgb2gray(observation[0]), observation[1]

    def step(self, action):
        modified_action = action
        observation, reward, truncated, done, info = self.env.step(modified_action)
        return rgb2gray(observation), reward, truncated, done, info


if __name__ == '__main__':
    env = make_vec_env(lambda: CustomEnvWrapper(CarRacing(continuous=False, verbose=True, render_mode='human')),
                       n_envs=1)

    model = PPO("CnnPolicy", env, verbose=1)

    obs = env.reset()
    for _ in tqdm(range(1000)):
        frames.append(obs[0])
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render()

    create_video_from_frames(frames, 'output.mp4')
