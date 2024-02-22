import os

import gymnasium as gym
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from stable_baselines3 import PPO, A2C, DDPG, SAC, DQN, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from callbacks.eval_callback import EvalCallback
from conf.config_models import AppConfig
from env.car_racing import CarRacing, set_obs_width_height
from wrapper.env_wrapper import CustomEnvWrapper

cs = ConfigStore.instance()
cs.store(name="base_config", node=AppConfig)


def make_algorithm(algo_name, **kwargs):
    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
        "DDPG": DDPG,
        "SAC": SAC,
        "DQN": DQN,
        "TD3": TD3
    }

    if algo_name in algorithms:
        algo_class = algorithms[algo_name]
        return algo_class(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: AppConfig) -> None:
    folders_to_create = [
        cfg.log_dir,
        cfg.eval.video_save_path,
        cfg.eval.model_save_path,
        cfg.algo_cfg.tensorboard_log
    ]

    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    yaml_config = OmegaConf.to_yaml(cfg)
    with open(cfg.log_dir + '/config.yaml', 'w') as f:
        f.write(yaml_config)

    print(
        f"###########################################################################\nRunning experiment with config: \n\n{yaml_config}\n\n###########################################################################")

    set_obs_width_height(cfg.env.obs_width, cfg.env.obs_height)

    if cfg.env.wrapper.use:
        base_env_fun = lambda: CustomEnvWrapper(
            gym.wrappers.TimeLimit(
                CarRacing(
                    verbose=False,
                    domain_randomize=cfg.env.domain_randomize,
                    continuous=cfg.env.continuous
                ),
                cfg.env.max_episode_steps
            ),
            trajectory_color_max_speed=cfg.env.wrapper.trajectory_color_max_speed,
            trajectory_thickness=cfg.env.wrapper.trajectory_thickness,
            draw_for_last=cfg.env.wrapper.draw_for_last,
        )

        eval_env = CustomEnvWrapper(
            gym.wrappers.TimeLimit(
                CarRacing(
                    verbose=False,
                    domain_randomize=cfg.env.domain_randomize,
                    continuous=cfg.env.continuous,
                    render_mode='rgb_array'
                ),
                cfg.env.max_episode_steps
            ),
            trajectory_color_max_speed=cfg.env.wrapper.trajectory_color_max_speed,
            trajectory_thickness=cfg.env.wrapper.trajectory_thickness,
            draw_for_last=cfg.env.wrapper.draw_for_last,
        )
    else:
        base_env_fun = lambda: gym.wrappers.TimeLimit(
            CarRacing(
                verbose=False,
                domain_randomize=cfg.env.domain_randomize,
                continuous=cfg.env.continuous
            ),
            cfg.env.max_episode_steps
        )

        eval_env = gym.wrappers.TimeLimit(
            CarRacing(
                verbose=False,
                domain_randomize=cfg.env.domain_randomize,
                continuous=cfg.env.continuous,
                render_mode='rgb_array'
            ),
            cfg.env.max_episode_steps
        )

    if cfg.train.n_envs == 1:
        env = make_vec_env(base_env_fun, n_envs=1)
    else:
        env = make_vec_env(
            base_env_fun,
            n_envs=cfg.train.n_envs,
            vec_env_cls=SubprocVecEnv
        )

    algo = make_algorithm(cfg.algo_name, env=env, **OmegaConf.to_container(cfg.algo_cfg, resolve=True))

    algo.learn(
        total_timesteps=cfg.train.total_timesteps,
        tb_log_name=cfg.name,
        callback=[
            EvalCallback(
                eval_env=eval_env,
                **OmegaConf.to_container(cfg.eval, resolve=True)
            ),
        ],
        progress_bar=True)


if __name__ == '__main__':
    my_app()
