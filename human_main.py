import pygame
from stable_baselines3.common.env_util import make_vec_env
from tqdm.auto import tqdm

from env.car_racing import CarRacing
from utils import create_video_from_frames
from wrapper.env_wrapper import CustomEnvWrapper

human_action = [0]


def get_human_action():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                human_action[0] = 2
            if event.key == pygame.K_RIGHT:
                human_action[0] = 1
            if event.key == pygame.K_UP:
                human_action[0] = 3
            if event.key == pygame.K_DOWN:
                human_action[0] = 4

        if event.type == pygame.KEYUP:
            human_action[0] = 0


if __name__ == '__main__':
    env = make_vec_env(lambda: CustomEnvWrapper(CarRacing(continuous=False, verbose=True, render_mode='human')),
                       n_envs=1)

    obs = env.reset()
    for _ in tqdm(range(500)):
        get_human_action()
        action = human_action
        bs, rewards, dones, info = env.step(action)

    create_video_from_frames(env.envs[0].frames, 'output.mp4')
