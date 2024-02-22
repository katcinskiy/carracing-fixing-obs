import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
from utils import create_video_from_frames
import logging
import os
import torch
from stable_baselines3.common.logger import Video

logger = logging.getLogger(__name__)


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, video_save_path, model_save_path, eval_freq, n_eval_episodes,
                 deterministic,
                 verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.video_save_path = video_save_path
        self.model_save_path = model_save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

        self._best_mean_reward = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            logger.info("Starting evaluation at episode {}".format(self.num_timesteps))
            all_rewards = []
            all_obs_frames = []
            all_orig_frames = []
            for i in tqdm(range(self.n_eval_episodes), desc="Evaluating"):
                obs = self.eval_env.reset()[0]
                orig_frames = []
                obs_frames = []
                done = False
                truncated = False
                cur_reward = 0
                while not done and not truncated:
                    action, _ = self.model.predict(obs[np.newaxis, :], deterministic=self.deterministic)
                    obs, reward, done, truncated, _ = self.eval_env.step(action[0])
                    # frame = np.flip(np.rot90(self.eval_env.render(), 3), axis=1)
                    frame = self.eval_env.render()
                    orig_frames.append(frame)
                    obs_frames.append(np.repeat(obs, 3, axis=2))
                    cur_reward += reward

                all_orig_frames.append(orig_frames)
                all_obs_frames.append(obs_frames)
                all_rewards.append(cur_reward)

                # (0, 2, 1) - No
                #
                self.logger.record(
                    f"eval/original_video_{i}",
                    Video(torch.ByteTensor(np.array(orig_frames)[np.newaxis, ...]).permute((0, 1, 4, 2, 3)), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                self.logger.record(
                    f"eval/obs_video_{i}",
                    Video(torch.ByteTensor(np.array(obs_frames)[np.newaxis, ...]).permute((0, 1, 4, 2, 3)), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )

                orig_video_path = self.video_save_path + f'/orig/{self.num_timesteps:07}/{i:02}_{cur_reward:.1f}.mp4'
                obs_video_path = self.video_save_path + f'/obs/{self.num_timesteps:07}/{i:02}_{cur_reward:.1f}.mp4'
                create_video_from_frames(orig_frames, orig_video_path)
                create_video_from_frames(obs_frames, obs_video_path)

            mean_reward = np.sum(all_rewards) / self.n_eval_episodes

            if self._best_mean_reward is None or mean_reward > self._best_mean_reward:
                model_path = self.model_save_path + f"{self.num_timesteps:07}_{mean_reward:.1f}" + '.zip'
                self.model.save(model_path)
                logger.info(f"Found best model with reward {mean_reward:.1f}, saved to {model_path}")

                self._best_mean_reward = mean_reward

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/max_reward", np.max(all_rewards))
            self.logger.record("eval/min_reward", np.min(all_rewards))

        return True
